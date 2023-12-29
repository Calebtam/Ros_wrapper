/**
* 
* Adapted from ORB-SLAM3: Examples/ROS/src/ros_stereo_inertial.cc
*
*/

#include "common.h"
#include <yaml-cpp/yaml.h>
#include "options/VioManagerOptions.h"
#include "state/State.h"
#include "state/Propagator.h"
#include "state/StateHelper.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "init/InertialInitializer.h"
#include "static/StaticInitializer.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace ov_core;
using namespace ov_msckf;
using namespace ov_init;

std::shared_ptr<ov_core::TrackBase> trackFEATS;

class Noise : public NoiseManager
{
public:

  virtual void print()  // 虚函数
  {
    PRINT_DEBUG("  - gyroscope_noise_density: %.6f\n", sigma_w);
    PRINT_DEBUG("  - accelerometer_noise_density: %.5f\n", sigma_a);
    PRINT_DEBUG("  - gyroscope_random_walk: %.7f\n", sigma_wb);
    PRINT_DEBUG("  - accelerometer_random_walk: %.6f\n", sigma_ab);
  }
  virtual void SetTopic(string str) = 0;   // 纯虚函数
  virtual string GetTopic()=0;
  virtual void SetExtrinsic(Eigen::Matrix4d T) = 0;   // 纯虚函数 Extrinsic
  virtual Eigen::Matrix4d GetExtrinsic()=0;
};
class ImuIntrics : public Noise
{
public:
    ImuIntrics(){};
    string topic;
    Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
    void print() {
        std::cout << "ImuIntrics topic: " << topic << std::endl;
        Noise::print();
        std::cout << "T_CtoI: " << std::endl << T_CtoI << std::endl;
    }
    void SetTopic(string str){
        topic = str;
    }
    string GetTopic(){
        return topic;
    }
    void SetExtrinsic(Eigen::Matrix4d T){
        T_CtoI = T;
    }
    Eigen::Matrix4d GetExtrinsic(){
        return T_CtoI;
    }
};

class ImuGrabber
{
public:
    ImuGrabber(string imu_topic): mImuTopic(imu_topic){};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
    string mImuTopic;
};
class RtkGrabber
{
public:
    RtkGrabber(ImuGrabber *pImuGb, std::shared_ptr<ov_msckf::State> state): mpImuGb(pImuGb), mState(state), thread_init_running(false), thread_init_success(false){ }

    void GrabRtk(const sensor_msgs::NavSatFixPtr & msg);
    bool try_to_initialize();
    void SyncWithImu();
    void pubPose(Eigen::Matrix<double, 3, 3> R, Eigen::Matrix<double, 3, 1> t, ros::Time msg_time);

    std::shared_ptr<ov_msckf::Propagator> propagator;
    // std::shared_ptr<ov_init::InertialInitializer> initializer;
    std::shared_ptr<StaticInitializer> init_static;
    std::atomic<bool> thread_init_running, thread_init_success;

    ros::Publisher pub_pose_out;

    queue<sensor_msgs::NavSatFixPtr> RtkBuf;
    std::mutex mBufMutex;
    ImuGrabber *mpImuGb;
    std::shared_ptr<ov_msckf::State> mState;
    
};

void load_noise(const std::shared_ptr<cv::FileStorage> config, const std::shared_ptr<ov_core::YamlParser> &parser, std::vector<std::shared_ptr<Noise>> &imu_noises) {

    int num;
    parser->parse_config("imu_num", num); // might be redundant
    std::vector<std::string> imuTopic;
    for (int i = 0; i < num; i++) {
        imuTopic.push_back(std::string("relative_config_imu") + std::to_string(i));
        string imuType;
        config->root()[imuTopic.at(i)]>>imuType;
        if (parser != nullptr) {
            std::shared_ptr<Noise> imu = make_shared<ImuIntrics>();
            parser->parse_external(imuTopic.at(i), "imu0", "gyroscope_noise_density", imu->sigma_w);
            parser->parse_external(imuTopic.at(i), "imu0", "gyroscope_random_walk", imu->sigma_wb);
            parser->parse_external(imuTopic.at(i), "imu0", "accelerometer_noise_density", imu->sigma_a);
            parser->parse_external(imuTopic.at(i), "imu0", "accelerometer_random_walk", imu->sigma_ab);
            string str;
            parser->parse_external(imuTopic.at(i), "imu0", "rostopic", str);
            imu->SetTopic(str);

            Eigen::Matrix4d T;
            parser->parse_external(imuTopic.at(i), "imu0", "T_cam_imu", T);
            imu->SetExtrinsic(T);
            // parser->parse_external(imuTopic.at(i), "imu0", "T_imu_cam", imu->T_CtoI);

            imu->sigma_w_2 = std::pow(imu->sigma_w, 2);
            imu->sigma_wb_2 = std::pow(imu->sigma_wb, 2);
            imu->sigma_a_2 = std::pow(imu->sigma_a, 2);
            imu->sigma_ab_2 = std::pow(imu->sigma_ab, 2);
            imu_noises.push_back(imu);
            std::cout << "------------------------" << i+1 << "/" << num << "---------------------------" << std::endl;
            std::cout << "imu  ==  "  << imuType << std::endl;
            imu->print();
            std::cout << "------------------------------------------------------" << std::endl;
        }
    }
    
}

void RtkGrabber::GrabRtk(const sensor_msgs::NavSatFixPtr &img_msg)
{
    mBufMutex.lock();
    // if (!RtkBuf.empty())
    while (RtkBuf.size() > 20){
        RtkBuf.pop();
    } 
    RtkBuf.push(img_msg);
    mBufMutex.unlock();
}
void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)

{
    mBufMutex.lock();
    while (imuBuf.size() > 50000){
        imuBuf.pop();
    } 
    imuBuf.push(imu_msg);
    // std::cout << "GrabImu callback "  << this->mImuTopic  << " " << imu_msg->header.stamp << std::endl;
    mBufMutex.unlock();
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Stereo_Inertial");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (argc > 1)
    {
        ROS_WARN ("Arguments supplied via command line are ignored.");
    }

    ros::NodeHandle node_handler;
    std::string node_name = ros::this_node::getName();
    image_transport::ImageTransport image_transport(node_handler);

    std::string settings_file;
    node_handler.param<std::string>(node_name + "/settings_file", settings_file, "file_not_set");
    std::cout << "settings_file path is "  << settings_file << std::endl;

    auto config = std::make_shared<cv::FileStorage>(settings_file, cv::FileStorage::READ);
    if (!config->isOpened()) {
      config = nullptr;
      return 0;
    }
    if (!config->isOpened()) {
      printf("unable to open the configuration file!\n%s\n" , settings_file.c_str());
      std::exit(EXIT_FAILURE);
    }

    // Load the config
    auto parser = std::make_shared<ov_core::YamlParser>(settings_file);

    // Verbosity
    std::string verbosity = "SILENT";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    auto params = std::make_shared<VioManagerOptions>();
    params->print_and_load(parser);

    verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    std::vector<std::shared_ptr<Noise>> imu_noises;
    load_noise(config, parser, imu_noises);

    double gravity_mag = 9.81;
    parser->parse_config("gravity_mag", gravity_mag);

    int num;
    parser->parse_config("imu_num", num);
    if(imu_noises.size() < num){
        printf("only one imu can use!\n\n");
        std::exit(EXIT_FAILURE);
    }

    // Our rtk topic
    std::string topic_rtk;
    parser->parse_config("rtk_topic", topic_rtk);
    PRINT_DEBUG("[SERIAL]: RTK: %s\n", topic_rtk.c_str());
    
    ImuGrabber imu(imu_noises.at(0)->GetTopic());
    std::shared_ptr<State> state = std::make_shared<State>(params->state_options);
    printf("make_shared state!\n\n");



    RtkGrabber igb(&imu, state);

    igb.propagator = std::make_shared<Propagator>(*imu_noises.at(0), gravity_mag);
    // igb.initializer = std::make_shared<ov_init::InertialInitializer>(params->init_options, trackFEATS->get_feature_database());
    auto imu_data = std::make_shared<std::vector<ov_core::ImuData>>();
    auto mtx_imu_data = std::make_shared<std::mutex>();
    igb.init_static = std::make_shared<StaticInitializer>(params->init_options, imu_data, mtx_imu_data);
    igb.pub_pose_out = node_handler.advertise<geometry_msgs::PoseStamped> ("/out_pose", 30);

    ros::Subscriber sub_imuCh0 = node_handler.subscribe(imu_noises.at(0)->GetTopic(), 1000, &ImuGrabber::GrabImu, &imu);
    ros::Subscriber sub_rtk = node_handler.subscribe(topic_rtk, 1000, &RtkGrabber::GrabRtk, &igb);

    auto sync_thread = std::make_shared<thread>(&RtkGrabber::SyncWithImu, &igb);
    std::cout << "start === " << std::endl;
    ros::spin();

    return 0;
}

void RtkGrabber::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    double lastTime = -1.0;
    bool  is_initialized_vio = false;

    std::chrono::milliseconds tSleep(3000);
    std::this_thread::sleep_for(tSleep);

    while(1)
    {
        // std::cout << "start === 1" << std::endl;
        if (!mpImuGb->imuBuf.empty())
        {
            { // IMU
                mpImuGb->mBufMutex.lock();
                while(!mpImuGb->imuBuf.empty())
                {
                    ov_core::ImuData message;
                    message.timestamp = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    message.data = mpImuGb->imuBuf.front();
                    this->propagator->feed_imu(message, lastTime);        // 输入imu数据

                    if (!is_initialized_vio) 
                        init_static->feed_imu(message, lastTime);

                    mpImuGb->imuBuf.pop();
                }
                mpImuGb->mBufMutex.unlock();
            }
            // std::cout << "imu === " << std::endl;
        }

        if(RtkBuf.empty() || RtkBuf.size() < 3 || thread_init_running){
            // std::cout << "rtk size === " << RtkBuf.size() << std::endl;
            continue;
        }
        
        if (!try_to_initialize()) {
            std::chrono::milliseconds tSleep(100);
            std::this_thread::sleep_for(tSleep);
            // Failure detection
            // last_no_slam_feature_time = message.timestamp;
            // is_initialized_vio = try_to_initialize();
            continue;
        }
        // if (!is_initialized_vio) 
        //     continue;
        std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
        if(!RtkBuf.empty() && RtkBuf.size() >= 2)
        {
            
            double tRtk = 0;
            tRtk = RtkBuf.front()->header.stamp.toSec();

            { // Rtk
                this->mBufMutex.lock(); 
                sensor_msgs::NavSatFixPtr rtk_msg = RtkBuf.front();
                RtkBuf.pop();
                this->mBufMutex.unlock();
            }  

            // // rtk 比 最新IMU还要新, 不能估计这个过程，返回去等
            // if(tRtk > mpImuGb->imuBuf.back()->header.stamp.toSec())
            // {
            //     std::chrono::milliseconds tSleep(5);
            //     std::this_thread::sleep_for(tSleep);
            //     continue;
            // }
            
            std::cout << std::fixed << " "  << tRtk << std::endl;

            // if (!mpImuGb->imuBuf.empty())
            // {
                // std::cout << "++++++++++++++++++++++++++++++++++++++++ " << std::endl;
            //     { // Rtk
            //         this->mBufMutex.lock(); 
            //         sensor_msgs::NavSatFixPtr rtk_msg = RtkBuf.front();
            //         RtkBuf.pop();
            //         this->mBufMutex.unlock();
            //     }    
            //     { // IMU
            //         mpImuGb->mBufMutex.lock();
            //         // 把这个周期的IMU数据全都放进去
            //         // while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tRtk)
            //         while(!mpImuGb->imuBuf.empty())
            //         {
            //             ov_core::ImuData message;
            //             message.timestamp = mpImuGb->imuBuf.front()->header.stamp.toSec();
            //             message.data = mpImuGb->imuBuf.front();

            //             this->propagator->feed_imu(message);        // 输入imu数据
                        
            //             mpImuGb->imuBuf.pop();
            //         }
            //         mpImuGb->mBufMutex.unlock();
            //     }
                if (mState->_timestamp != tRtk) {
                    this->propagator->propagate_and_clone(mState, tRtk);
                    lastTime = tRtk;
                }
                if(mState->_clones_IMU.find(tRtk) != mState->_clones_IMU.end())
                {
                    std::cout << std::endl << 
                    mState->_clones_IMU.at(tRtk)->Rot() << std::endl << 
                    mState->_clones_IMU.at(tRtk)->pos().transpose() << std::endl;
                }
                // // Main algorithm runs here
                // Sophus::SE3f Tcw = mpSLAM->TrackMonocular(im, tIm, vImuMeas);
                // Sophus::SE3f Twc = Tcw.inverse();
                pubPose(mState->_clones_IMU.at(tRtk)->Rot(), mState->_clones_IMU.at(tRtk)->pos(), (ros::Time)mState->_timestamp);
                // publish_ros_camera_pose(Twc, msg_time);
                // publish_ros_tf_transform(Twc, world_frame_id, cam_frame_id, msg_time);
                // publish_ros_tracked_mappoints(mpSLAM->GetTrackedMapPoints(), msg_time);
                
                // 
                ov_msckf::StateHelper::marginalize_old_clone(mState);
            // }
        }


        std::chrono::milliseconds tSleep(1);
        std::this_thread::sleep_for(tSleep);
    }
}
void RtkGrabber::pubPose(Eigen::Matrix<double, 3, 3> R, Eigen::Matrix<double, 3, 1> t, ros::Time msg_time) 
{
    Eigen::Vector3d global_t;
    Eigen:: Quaterniond global_q;

    global_q = R;
    global_t = t;

    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time(0);
    pose.header.frame_id = "world";
    pose.pose.position.x = global_t.x();
    pose.pose.position.y = global_t.y();
    pose.pose.position.z = global_t.z();
    pose.pose.orientation.x = global_q.x();
    pose.pose.orientation.y = global_q.y();
    pose.pose.orientation.z = global_q.z();
    pose.pose.orientation.w = global_q.w();
    pub_pose_out.publish(pose);

    // tf::Transform transform_w_c;
    // transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
    // tf::Quaternion q;
    // q = global_q;

    static tf::TransformBroadcaster tf_broadcaster;
    tf::Transform transform_w_c;
    tf::Quaternion tmp_q(global_q.x(), global_q.y(), global_q.z(), global_q.w());
    transform_w_c.setRotation(tmp_q);
    transform_w_c.setOrigin(tf::Vector3(global_t.x(), global_t.y(), global_t.z() ));
    tf_broadcaster.sendTransform(tf::StampedTransform(transform_w_c,  ros::Time::now(), "world", "camera")); 

}
bool RtkGrabber::try_to_initialize() 
{
    // std::cout << "try_to_initialize [              ]" << std::endl;


    // If the thread was a success, then return success!
    if (thread_init_success) {
        //  std::cout << " init success " << std::endl;
        return true;
    }

    if(RtkBuf.empty()){
        return false;
    }
    // std::cout << "try_to_initialize [==            ]" << std::endl;
    std::vector<double> tRtk;
    {
        this->mBufMutex.lock(); 
        // std::queue<sensor_msgs::NavSatFixPtr> temp = RtkGrabber::RtkBuf; // 复制队列
        while (!RtkBuf.empty())
        {
            sensor_msgs::NavSatFixPtr& item = RtkBuf.front();
            tRtk.push_back(item->header.stamp.toSec());
            RtkBuf.pop();
        }
        this->mBufMutex.unlock();
    }
    // std::cout << "try_to_initialize [====          ]" << std::endl;

    // tRtk = RtkBuf.front()->header.stamp.toSec();

    // Run the initialization in a second thread so it can go as slow as it desires
    thread_init_running = true;
    std::thread thread([&] {
        // Returns from our initializer
        double timestamp;
        Eigen::MatrixXd covariance;
        std::vector<std::shared_ptr<ov_type::Type>> order;

        // std::cout << "try_to_initialize [=====         ]" << std::endl;
        bool success = init_static->initialize(timestamp, covariance, order, mState->_imu, false);
        
        if (success) 
        {

            // Set our covariance (state should already be set in the initializer)
            StateHelper::set_initial_covariance(mState, covariance, order);

            // Set the state time
            mState->_timestamp = timestamp;

            PRINT_INFO(GREEN "[init]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, mState->_imu->quat()(0), mState->_imu->quat()(1),
                        mState->_imu->quat()(2), mState->_imu->quat()(3));
            PRINT_INFO(GREEN "[init]: bias gyro = %.4f, %.4f, %.4f\n" RESET, mState->_imu->bias_g()(0), mState->_imu->bias_g()(1),
                        mState->_imu->bias_g()(2));
            PRINT_INFO(GREEN "[init]: velocity = %.4f, %.4f, %.4f\n" RESET, mState->_imu->vel()(0), mState->_imu->vel()(1), mState->_imu->vel()(2));
            PRINT_INFO(GREEN "[init]: bias accel = %.4f, %.4f, %.4f\n" RESET, mState->_imu->bias_a()(0), mState->_imu->bias_a()(1),
                        mState->_imu->bias_a()(2));
            PRINT_INFO(GREEN "[init]: position = %.4f, %.4f, %.4f\n" RESET, mState->_imu->pos()(0), mState->_imu->pos()(1), mState->_imu->pos()(2));

            // Now we have initialized we will propagate the state to the current timestep
            // In general this should be ok as long as the initialization didn't take too long to perform
            // Propagating over multiple seconds will become an issue if the initial biases are bad
            for (size_t i = 0; i < tRtk.size(); i ++) 
            {
                if(tRtk.at(i) > mState->_timestamp)
                {
                    propagator->propagate_and_clone(mState, tRtk.at(i));
                    StateHelper::marginalize_old_clone(mState);
                }
            }
            std::cout << std::endl;
            PRINT_DEBUG(YELLOW "[init]: moved the state forward %.2f seconds\n" RESET, mState->_timestamp - timestamp);
            thread_init_success = true;
            // std::cout << "try_to_initialize [===========   ]" << std::endl;
        } 
        // Finally, mark that the thread has finished running
        thread_init_running = false;
    });
    // If we are single threaded, then run single threaded
    // Otherwise detach this thread so it runs in the background!
    // if (!params->use_multi_threading_subs) {
        // thread.join();
    // } else {
        thread.detach();
    // }
    
    if(thread_init_success)
    {
        std::cout << "try_to_initialize [==============]" << std::endl;
        return true;
    }
    else
        return false;
}