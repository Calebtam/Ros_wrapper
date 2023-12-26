/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_ROS1VISUALIZER_H
#define OV_MSCKF_ROS1VISUALIZER_H
#ifdef USE_ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>
#endif
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>


#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>

#ifdef PC_VISUALIZATION
#include "Viewer.h"
#endif
#include "core/VioManager.h"

namespace ov_core {
class YamlParser;
struct CameraData;
} // namespace ov_core


namespace ov_msckf {

class VioManager;
class Simulator;

/**
 * @brief Helper class that will publish results onto the ROS framework.
 *
 * Also save to file the current total state and covariance along with the groundtruth if we are simulating.
 * We visualize the following things:
 * - State of the system on TF, pose message, and path
 * - Image of our tracker
 * - Our different features (SLAM, MSCKF, ARUCO)
 * - Groundtruth trajectory if we have it
 */
class ROS1Visualizer {

public:
  /**
   * @brief Default constructor
   * @param nh ROS node handler
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
#ifdef USE_ROS
  ROS1Visualizer(std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr, std::shared_ptr<ros::NodeHandle> nh = nullptr);
#else
  ROS1Visualizer(std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr);
#endif
  /**
   * @brief Will setup ROS subscribers and callbacks
   * @param parser Configuration file parser
   */
  void setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser);

  /**
   * @brief Will visualize the system if we have new things
   */
  void visualize();

  /**
   * @brief Will publish our odometry message for the current timestep.
   * This will take the current state estimate and get the propagated pose to the desired time.
   * This can be used to get pose estimates on systems which require high frequency pose estimates.
   */
  void visualize_odometry(double timestamp);

  /**
   * @brief After the run has ended, print results
   */
  void visualize_final();

  /// Callback for inertial information
  void callback_inertial(const sensor_msgs::Imu::ConstPtr &msg);

  /// Callback for monocular cameras information
  void callback_monocular(const sensor_msgs::ImageConstPtr &msg0, int cam_id0);

  /// Callback for synchronized stereo camera information
  void callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0, int cam_id1);

  void callback_pub_reset(bool isReset) { 
    std::cout << "******* send start  ******  " << isReset << std::endl;
//    system("curl -H \"content-Type: application/json\" -H \"Content-Length: 0\" -X POST http://localhost:8080/pause");
    std::cout << "******* send finish ******  " << std::endl;
  }

  void setCallbackForPoseMsg(pose_msg_callback_fun_ptr callback){
    pose_callback_fun = callback;
    _app->pose_callback_fun = callback;
  }

  void setCallBackForVioDistance(std::function<void(const float)> callback){
    odometry_distance_callback_fun = callback;
  }

  void query_vio_status();
  void set_vo_reset_callback(std::function<void()> callback){
    vo_reset_callback_fun = callback;
  }

  void reset_vo(){
    vo_reset = true;
  }
  void set_vo_status_callback(std::function<void(int)> callback){
    vo_status_callback_fun = callback;
  }
  void stopSlamSystem() {
    m_processImgFlag = false;
    image_processing_thread_cv.notify_all();
    if(image_processing_thread){
      if(image_processing_thread->joinable()){
        image_processing_thread->join();
        delete image_processing_thread;
      }
    }
  }

  protected:
  /// Publish the current state
  void publish_state();

  /// Publish the active tracking image
  void publish_images();

  /// Publish current features
  void publish_features();

  /// Publish groundtruth (if we have it)
  void publish_groundtruth();

  /// Publish loop-closure information of current pose and active track information
  void publish_loopclosure_information();

  /// Reset visualizer
  void reset_visualizer();

  /// Image processing
  void image_processing();

  /// Global node handler
#ifdef USE_ROS
  std::shared_ptr<ros::NodeHandle> _nh;
#endif
  /// Vio options
//  VioManagerOptions params;
  /// Core application of the filter system
  std::shared_ptr<VioManager> _app;
  /// Simulator (is nullptr if we are not sim'ing)
  std::shared_ptr<Simulator> _sim;
#ifdef USE_ROS
  // Our publishers
  image_transport::Publisher it_pub_tracks, it_pub_loop_img_depth, it_pub_loop_img_depth_color;
  ros::Publisher pub_poseimu, pub_odomimu, pub_pathimu;
  ros::Publisher pub_points_msckf, pub_points_slam, pub_points_aruco, pub_points_sim;
  ros::Publisher pub_loop_pose, pub_loop_point, pub_loop_extrinsic, pub_loop_intrinsics;
  std::shared_ptr<tf::TransformBroadcaster> mTfBr;

  // Our subscribers and camera synchronizers
  ros::Subscriber sub_imu;
  std::vector<ros::Subscriber> subs_cam;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  std::vector<std::shared_ptr<message_filters::Synchronizer<sync_pol>>> sync_cam;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam;
#endif
  /// vio status
  int vio_status = 0;
  int last_vio_status = 1;
  std::mutex vio_mutex;
  int current_vision_status = 0;
  // For path viz
  unsigned int poses_seq_imu = 0;
  std::vector<geometry_msgs::PoseStamped> poses_imu;
  bool m_processImgFlag = true;
  // Groundtruth infomation
#ifdef USE_ROS
  ros::Publisher pub_pathgt, pub_posegt;
#endif
  double summed_mse_ori = 0.0;
  double summed_mse_pos = 0.0;
  double summed_nees_ori = 0.0;
  double summed_nees_pos = 0.0;
  size_t summed_number = 0;

  // Start and end timestamps
  bool start_time_set = false;
  boost::posix_time::ptime rT1, rT2;

  // Thread atomics
  std::atomic<bool> thread_update_running;

  /// Queue up camera measurements sorted by time and trigger once we have
  /// exactly one IMU measurement with timestamp newer than the camera measurement
  /// This also handles out-of-order camera measurements, which is rare, but
  /// a nice feature to have for general robustness to bad camera drivers.
  std::deque<ov_core::CameraData> camera_queue;
  std::mutex camera_queue_mtx;
  std::thread* image_processing_thread = nullptr;
  std::mutex image_processing_thread_mtx;
  std::condition_variable image_processing_thread_cv;
  std::mutex imu_camera_syn_mtx;
  double latest_imu_msg_timestamp = -1;

  // Last camera message timestamps we have received (mapped by cam id)
  std::map<int, double> camera_last_timestamp;

  // Last timestamp we visualized at
  double last_visualization_timestamp = 0;
  double last_visualization_timestamp_image = 0;

  // Our groundtruth states
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;

  // For path viz
  unsigned int poses_seq_gt = 0;
  std::vector<geometry_msgs::PoseStamped> poses_gt;
  bool publish_global2imu_tf = true;
  bool publish_calibration_tf = true;

  // Files and if we should save total state
  bool save_total_state = false;
  std::ofstream of_state_est, of_state_std, of_state_gt;

  //for fusion
//  pose_msg_callback_fun_ptr pose_callback_fun = nullptr;   /// c++ interface callback function
  pose_msg_callback_fun_ptr pose_callback_fun = nullptr;
  std::function<void()> vo_reset_callback_fun = nullptr;
  std::atomic<bool> vo_reset;
  std::function<void(int)> vo_status_callback_fun = nullptr;
  //
  std::function<void(const float)> odometry_distance_callback_fun = nullptr;
public:
  //for visualization
#ifdef PC_VISUALIZATION
  std::shared_ptr<ov_visualization::SdViewer> viewer_ptr = nullptr;
#endif
  /// frame frequency control related
  int camera_frequency = 20;
  int interval_num = 1;
};

} // namespace ov_msckf

#endif // OV_MSCKF_ROS1VISUALIZER_H
