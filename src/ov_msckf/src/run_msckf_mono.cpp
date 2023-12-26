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
#ifdef USE_ROS
#include <ros/ros.h>
#endif
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <memory>

#include "core/VioManager.h"
#include "options/VioManagerOptions.h"
#include "ros/ROS1Visualizer.h"
#include "utils/dataset_reader.h"
#include <signal.h>
using namespace ov_msckf;
#ifndef USE_ROS
using namespace ob_slam;
#endif

std::shared_ptr<VioManager> sys;
std::shared_ptr<ROS1Visualizer> viz;

void signal_callback_handler(int signum) {
    viz->visualize_final();
    PRINT_INFO(" Ctr + C Interrupt !!");
    exit(signum);
}

// Main function
int main(int argc, char **argv) {

  signal(SIGINT, signal_callback_handler);
  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }
#ifdef USE_ROS
  // Launch our ros node
  ros::init(argc, argv, "ros1_serial_msckf");
  auto nh = std::make_shared<ros::NodeHandle>("~");
//  nh->param<std::string>("config_path", config_path, config_path);
#endif
  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
#ifdef USE_ROS
  parser->set_node_handler(nh);
#endif
  // Verbosity
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  auto params = std::make_shared<VioManagerOptions>();
  params->print_and_load(parser);
  // params->num_opencv_threads = 0; // uncomment if you want repeatability
  // params->use_multi_threading_pubs = 0; // uncomment if you want repeatability
  params->use_multi_threading_subs = false;
  sys = std::make_shared<VioManager>(params);
#ifdef USE_ROS
  viz = std::make_shared<ROS1Visualizer>(sys, nh);
#else
  viz = std::make_shared<ROS1Visualizer>(sys);
#endif
  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "[SERIAL]: unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Our imu topic
  std::string topic_imu;
//  nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  PRINT_DEBUG("[SERIAL]: imu: %s\n", topic_imu.c_str());

  // Our camera topics
  std::vector<std::string> topic_cameras;
  for (int i = 0; i < params->state_options.num_cameras; i++) {
    std::string cam_topic;
//    nh->param<std::string>("topic_camera" + std::to_string(i), cam_topic, "/cam" + std::to_string(i) + "/image_raw");
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
    topic_cameras.emplace_back(cam_topic);
    PRINT_DEBUG("[SERIAL]: cam: %s\n", cam_topic.c_str());
  }

  // Location of the ROS bag we want to read in
  std::string path_to_bag;
  parser->parse_config("path_bag", path_to_bag);
  PRINT_DEBUG("[SERIAL]: ros bag path is: %s\n", path_to_bag.c_str());

  // Get our start location and how much of the bag we want to play
  // Make the bag duration < 0 to just process to the end of the bag
  double bag_start, bag_durr;
#ifdef USE_ROS
  nh->param<double>("bag_start", bag_start, 0);
  nh->param<double>("bag_durr", bag_durr, -1);
#else
  parser->parse_config("bag_start", bag_start);
  parser->parse_config("bag_durr", bag_durr);
#endif
  PRINT_DEBUG("[SERIAL]: bag start: %.1f\n", bag_start);
  PRINT_DEBUG("[SERIAL]: bag duration: %.1f\n", bag_durr);

  //===================================================================================

  // Load rosbag here, and find messages we can play
  rosbag::Bag bag;
  bag.open(path_to_bag, (uint32_t)rosbag::bagmode::BagMode::Read);

  // We should load the bag as a view
  // Here we go from beginning of the bag to the end of the bag
  rosbag::View view_full;
  rosbag::View view;

  // Start a few seconds in from the full view time
  // If we have a negative duration then use the full bag length
  view_full.addQuery(bag);
  auto time_init = view_full.getBeginTime();
#ifdef USE_ROS
  time_init += ros::Duration(bag_start);
  auto time_finish = (bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(bag_durr);
#else
  time_init += ob_slam::Duration(bag_start);
  auto time_finish = (bag_durr < 0) ? view_full.getEndTime() : time_init + ob_slam::Duration(bag_durr);
#endif
  PRINT_DEBUG("time start = %.6f\n", time_init.toSec());
  PRINT_DEBUG("time end   = %.6f\n", time_finish.toSec());
  view.addQuery(bag, time_init, time_finish);

  // Check to make sure we have data to play
  if (view.size() == 0) {
    PRINT_ERROR(RED "[SERIAL]: No messages to play on specified topics.  Exiting.\n" RESET);
#ifdef USE_ROS
    ros::shutdown();
#endif
    return EXIT_FAILURE;
  }

  // loop through rosbag
  rosbag::View::iterator iter, next_iter;
  for (iter = view.begin(), next_iter = view.begin(); iter != view.end(); iter++) {
#ifdef PC_VISUALIZATION
    while (viz->viewer_ptr->mbPausePlayBag) {
      usleep(5000);
    }
#endif
    if (iter->getTopic() == topic_imu) {
      viz->callback_inertial(iter->instantiate<sensor_msgs::Imu>());
    }
//    for (int i = 0; i < params->state_options.num_cameras; i++) {
    if (iter->getTopic() == topic_cameras.at(0)) {
      if (iter->isType<sensor_msgs::Image>()) {
        viz->callback_monocular(iter->instantiate<sensor_msgs::Image>(), 0);
      }else{
        const auto image_ptr = iter->instantiate<ob_slam::sensor_msgs::CompressedImage>();
        cv::Mat mat = cv::imdecode(image_ptr->data, cv::IMREAD_GRAYSCALE);
        sensor_msgs::ImagePtr msg = nullptr;
        msg = cv_bridge::CvImage(image_ptr->header, "mono8", mat).toImageMsg();
        viz->callback_monocular(msg, 0);
      }
    }
//    }
    if (++next_iter != view.end()) {
      //预留帧间时间间隔
      double sleep_time = next_iter->getTime().toSec() - iter->getTime().toSec();
//                sleep_time = sleep_time / 0.5;
      struct timespec ts;
      ts.tv_sec = (long)sleep_time;
      ts.tv_nsec = (long)((sleep_time - (long)sleep_time) * 1e9);
      nanosleep(&ts, 0);
    }
  }
  PRINT_WARNING(RED "[SERIAL]: play bag finished\n" RESET);

  // Final visualization
  viz->visualize_final();

  // Done!
  return EXIT_SUCCESS;
}
