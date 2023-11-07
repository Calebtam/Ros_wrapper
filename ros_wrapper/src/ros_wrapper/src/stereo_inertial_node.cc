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
// using namespace ov_type;
// using namespace ov_msckf;

// std::shared_ptr<ov_msckf::VioManager> sys;
// std::shared_ptr<ov_msckf::ROS1Visualizer> viz;
std::shared_ptr<ov_core::TrackBase> trackFEATS;
std::shared_ptr<ov_msckf::VioManagerOptions> params;

class ImgWriter
{
public:
    ImgWriter(const string Path): DataSavePath(Path){}

    void WriterImg(const cv::Mat &img_msg);
    void WriterImg(const cv::Mat &img_msg1, const cv::Mat &img_msg2);
    string DataSavePath;
    long factory_id = 0;
};

class ImuGrabber
{
public:
    ImuGrabber(){};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ImuGrabber *pImuGb): mpImuGb(pImuGb){}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    void GrabImage(const sensor_msgs::ImageConstPtr &img_msg);

    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    cv_bridge::CvImageConstPtr GetImagePtr(const sensor_msgs::ImageConstPtr &img_msg);

    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf, imgBuf;
    std::mutex mBufMutexLeft,mBufMutexRight,mBufMuteximg;
    // ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;
};

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

    std::string cam1Topic, cam2Topic, imuTopic, camTopic;
    config->root()["cam0rostopic"]>>cam1Topic;
    config->root()["cam1rostopic"]>>cam2Topic;
    config->root()["imu0rostopic"]>>imuTopic;
    config->root()["camrostopic"]>>camTopic;
    std::cout << "cam0rostopic path is "  << cam1Topic << std::endl;
    std::cout << "cam1rostopic path is "  << cam2Topic << std::endl;
    std::cout << "imu0rostopic path is "  << imuTopic << std::endl;
    std::cout << "camrostopic path is "  << camTopic << std::endl;

    ImuGrabber imugb;
    ImageGrabber igb(&imugb);


    // Maximum delay, 5 seconds * 200Hz = 1000 samples
    ros::Subscriber sub_imu = node_handler.subscribe(imuTopic, 1000, &ImuGrabber::GrabImu, &imugb); 
    ros::Subscriber sub_img = node_handler.subscribe(camTopic, 100, &ImageGrabber::GrabImage, &igb);
    ros::Subscriber sub_img_left = node_handler.subscribe(cam1Topic, 100, &ImageGrabber::GrabImageLeft, &igb);
    ros::Subscriber sub_img_right = node_handler.subscribe(cam2Topic, 100, &ImageGrabber::GrabImageRight, &igb);

    std::shared_ptr<ov_core::YamlParser> parser = std::make_shared<ov_core::YamlParser>(settings_file);

    // Verbosity
    std::string verbosity = "DEBUG";
    ov_core::Printer::setPrintLevel(verbosity);

    auto params = std::make_shared<ov_msckf::VioManagerOptions>();
    params->print_and_load(parser);

    // Create the state!!
    std::shared_ptr<ov_msckf::State> state = std::make_shared<ov_msckf::State>(params->state_options);

    // Timeoffset from camera to IMU
    Eigen::VectorXd temp_camimu_dt;
    temp_camimu_dt.resize(1);
    temp_camimu_dt(0) = params->calib_camimu_dt;
    state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
    state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

    // Loop through and load each of the cameras
    state->_cam_intrinsics_cameras = params->camera_intrinsics;
    for (int i = 0; i < state->_options.num_cameras; i++) {
        state->_cam_intrinsics.at(i)->set_value(params->camera_intrinsics.at(i)->get_value());
        state->_cam_intrinsics.at(i)->set_fej(params->camera_intrinsics.at(i)->get_value());
        state->_calib_IMUtoCAM.at(i)->set_value(params->camera_extrinsics.at(i));
        state->_calib_IMUtoCAM.at(i)->set_fej(params->camera_extrinsics.at(i));
    }

    // 内参 
    trackFEATS = std::shared_ptr<ov_core::TrackBase>(
        new TrackKLT(state->_cam_intrinsics_cameras, 50, state->_options.max_aruco_features, params));

    // setup_ros_publishers(node_handler, image_transport);

    auto sync_thread = std::make_shared<thread>(&ImageGrabber::SyncWithImu, &igb);
    std::cout << "start === " << std::endl;
    ros::spin();

    return 0;
}
void ImgWriter::WriterImg(const cv::Mat &img_msg1, const cv::Mat &img_msg2)
{
    factory_id++;
    stringstream stream;
    stream << factory_id;
    string result;
    stream >> result;
    string path = DataSavePath + "/L_" + result + ".jpg";
    imwrite(path, img_msg1);
    path = DataSavePath + "/R_" + result + ".jpg";
    imwrite(path, img_msg2);
}
void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // std::cout << "GrabImageLeft callback "  << std::endl;

    mBufMuteximg.lock();
    if (!imgBuf.empty())
        imgBuf.pop();
    imgBuf.push(img_msg);
    mBufMuteximg.unlock();
}
void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
    // std::cout << "GrabImageLeft callback "  << std::endl;

    mBufMutexLeft.lock();
    if (!imgLeftBuf.empty())
        imgLeftBuf.pop();
    imgLeftBuf.push(img_msg);
    mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg)
{
    // std::cout << "GrabImageRight callback "  << std::endl;
    mBufMutexRight.lock();
    if (!imgRightBuf.empty())
        imgRightBuf.pop();
    imgRightBuf.push(img_msg);
    mBufMutexRight.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    
    if(cv_ptr->image.type()==0)
    {
        return cv_ptr->image.clone();
    }
    else
    {
        std::cout << "Error type" << std::endl;
        return cv_ptr->image.clone();
    }
}
cv_bridge::CvImageConstPtr ImageGrabber::GetImagePtr(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    
    if(cv_ptr->image.type()==0)
    {
        return cv_ptr;
    }
    else
    {
        std::cout << "Error type" << std::endl;
        return cv_ptr;
    }
}
void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // std::cout << "GrabImu callback "  << std::endl;
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    
    return;
}

void ImageGrabber::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    ImgWriter imWriter_("/home/tam/DataSet/img");
    while(1)
    {
        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        if (!imgLeftBuf.empty()&&!imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())
        {
            tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            tImRight = imgRightBuf.front()->header.stamp.toSec();

            this->mBufMutexRight.lock();
            while((tImLeft-tImRight)>maxTimeDiff && imgRightBuf.size()>1)
            {
                imgRightBuf.pop();
                tImRight = imgRightBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRight.unlock();

            this->mBufMutexLeft.lock();
            while((tImRight-tImLeft)>maxTimeDiff && imgLeftBuf.size()>1)
            {
                imgLeftBuf.pop();
                tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexLeft.unlock();

            if((tImLeft-tImRight)>maxTimeDiff || (tImRight-tImLeft)>maxTimeDiff)
            {
                // std::cout << "big time difference" << std::endl;
                continue;
            }
            if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;


            ov_core::CameraData message;
            message.timestamp = imgLeftBuf.front()->header.stamp.toSec();
            message.sensor_ids.push_back(0);
            message.sensor_ids.push_back(1);

            this->mBufMutexLeft.lock();
            imLeft = GetImage(imgLeftBuf.front());
            ros::Time msg_time = imgLeftBuf.front()->header.stamp;
            message.images.push_back(imLeft);
            imgLeftBuf.pop();
            this->mBufMutexLeft.unlock();

            this->mBufMutexRight.lock();
            imRight = GetImage(imgRightBuf.front());
            message.images.push_back(imRight);
            imgRightBuf.pop();
            this->mBufMutexRight.unlock();

            // cv::imshow("imLeft", imLeft);
            // cv::imshow("imRight", imRight);
            // cv::waitKey(0); 

            imWriter_.WriterImg(imLeft, imRight);

            trackFEATS->feed_new_camera(message);

            cv::waitKey(100);           
            
            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
        else if(!imgBuf.empty())
        {
            cv::Mat im;
            tImLeft = imgBuf.front()->header.stamp.toSec();

            this->mBufMuteximg.lock();
            im = GetImage(imgBuf.front());
            imgBuf.pop();
            this->mBufMuteximg.unlock();
            // cv::imshow("imLeft", im(cv::Rect(0, 0, im.cols, im.rows/2)));
            // cv::imshow("imRight", im(cv::Rect(0, im.rows/2, im.cols, im.rows/2)));
             
            ov_core::CameraData message;
            message.timestamp = imgLeftBuf.front()->header.stamp.toSec();
            message.sensor_ids.push_back(0);
            message.sensor_ids.push_back(1);
            message.images.push_back(im(cv::Rect(0, 0, im.cols, im.rows/2)));
            message.images.push_back(im(cv::Rect(0, im.rows/2, im.cols, im.rows/2)));

            trackFEATS->feed_new_camera(message);

            cv::waitKey(2);        

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

