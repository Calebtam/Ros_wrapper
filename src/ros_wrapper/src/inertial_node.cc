/**
* 
* Adapted from ORB-SLAM3: Examples/ROS/src/ros_stereo_inertial.cc
*
*/

#include "common.h"
#include <yaml-cpp/yaml.h>
// #include "opencv_yaml_parse.h"
// #include "core/VioManager.h"
// #include "options/VioManagerOptions.h"
// #include "ros/ROS1Visualizer.h"
#include "options/VioManagerOptions.h"
#include "State.h"
#include "track/TrackBase.h"
#include "track/TrackKLT.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace ov_core;
using namespace ov_msckf;

std::shared_ptr<ov_core::TrackBase> trackFEATS;
std::shared_ptr<ov_msckf::VioManagerOptions> params;

class ImuGrabber
{
public:
    ImuGrabber(string imu_topic): mImuTopic(imu_topic){};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
    string mImuTopic;
};

class ImuSyn
{
public:
    ImuSyn(ImuGrabber *pCh0, ImuGrabber *pIcm): mpCh0(pCh0), mpIcm(pIcm){}

    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf, imgBuf;
    std::mutex mBufMutexLeft,mBufMutexRight,mBufMuteximg;
    // ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpCh0;
    ImuGrabber *mpIcm;
};

class ImuIntrics : public ov_msckf::NoiseManager
{
public:
    ImuIntrics(){};
    string topic;
    Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
    void print() {
        std::cout << "ImuIntrics topic: " << topic << std::endl;
        ov_msckf::NoiseManager::print();
        std::cout << "T_CtoI: " << std::endl << T_CtoI << std::endl;
    }
};
void load_noise(const std::shared_ptr<cv::FileStorage> config, const std::shared_ptr<ov_core::YamlParser> &parser, std::vector<ImuIntrics> &imu_noises) {

    int num;
    parser->parse_config("imu_num", num); // might be redundant
    std::vector<std::string> imuTopic;
    for (int i = 0; i < num; i++) {
        imuTopic.push_back(std::string("relative_config_imu") + std::to_string(i));
        string imu;
        config->root()[imuTopic.at(i)]>>imu;
        if (parser != nullptr) {
            ImuIntrics imu_noise;
            parser->parse_external(imuTopic.at(i), "imu0", "rostopic", imu_noise.topic);
            parser->parse_external(imuTopic.at(i), "imu0", "gyroscope_noise_density", imu_noise.sigma_w);
            parser->parse_external(imuTopic.at(i), "imu0", "gyroscope_random_walk", imu_noise.sigma_wb);
            parser->parse_external(imuTopic.at(i), "imu0", "accelerometer_noise_density", imu_noise.sigma_a);
            parser->parse_external(imuTopic.at(i), "imu0", "accelerometer_random_walk", imu_noise.sigma_ab);
            parser->parse_external(imuTopic.at(i), "imu0", "T_imu_cam", imu_noise.T_CtoI);
            imu_noise.sigma_w_2 = std::pow(imu_noise.sigma_w, 2);
            imu_noise.sigma_wb_2 = std::pow(imu_noise.sigma_wb, 2);
            imu_noise.sigma_a_2 = std::pow(imu_noise.sigma_a, 2);
            imu_noise.sigma_ab_2 = std::pow(imu_noise.sigma_ab, 2);
            imu_noises.push_back(imu_noise);
            std::cout << "------------------------" << i+1 << "/" << num << "---------------------------" << std::endl;
            std::cout << "imu  ==  "  << imu << std::endl;
            imu_noise.print();
            std::cout << "------------------------------------------------------" << std::endl;
        }
    }
    
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
    std::shared_ptr<ov_core::YamlParser> parser = std::make_shared<ov_core::YamlParser>(settings_file);

    // Verbosity
    std::string verbosity = "DEBUG";
    ov_core::Printer::setPrintLevel(verbosity);

    std::vector<ImuIntrics> imu_noises;
    load_noise(config, parser, imu_noises);

    if(imu_noises.size() < 2){
        printf("only one imu can use!\n%s\n");
        std::exit(EXIT_FAILURE);
    }

    ImuGrabber imuCh0(imu_noises.at(0).topic), imuIcm(imu_noises.at(1).topic);
    ImuSyn igb(&imuCh0, &imuIcm);

    ros::Subscriber sub_imuCh0 = node_handler.subscribe(imu_noises.at(0).topic, 1000, &ImuGrabber::GrabImu, &imuCh0);
    ros::Subscriber sub_imuIcm = node_handler.subscribe(imu_noises.at(1).topic, 1000, &ImuGrabber::GrabImu, &imuIcm);

    // setup_ros_publishers(node_handler, image_transport);

    auto sync_thread = std::make_shared<thread>(&ImuSyn::SyncWithImu, &igb);
    std::cout << "start === " << std::endl;
    ros::spin();

    return 0;
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    // std::cout << "GrabImu callback "  << this->mImuTopic  << " " << imu_msg->header.stamp << std::endl;
    mBufMutex.unlock();
    return;
}
void ImuSyn::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    while(1)
    {
        // cv::Mat imLeft, imRight;
        // double tImLeft = 0, tImRight = 0;
        // // if (!imgLeftBuf.empty()&&!imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())
        // if (0)
        // {
 
            
        //     std::chrono::milliseconds tSleep(1);
        //     std::this_thread::sleep_for(tSleep);
        // }
        // else if(0)
        // {
    

        //     std::chrono::milliseconds tSleep(1);
        //     std::this_thread::sleep_for(tSleep);
        // }
        std::chrono::milliseconds tSleep(100);
        std::this_thread::sleep_for(tSleep);
    }
}

