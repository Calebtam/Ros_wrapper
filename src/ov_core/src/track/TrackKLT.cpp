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

#include "TrackKLT.h"

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

using namespace ov_core;

void TrackKLT::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != params_ptr->masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - params_ptr->masks.size() = %zu\n" RESET, params_ptr->masks.size());
    std::exit(EXIT_FAILURE);
    // return;
  }

  // Preprocessing steps that we do not parallelize
  // NOTE: DO NOT PARALLELIZE THESE!
  // NOTE: These seem to be much slower if you parallelize them...
#ifdef TIME_STATISTICS
  rT1 = boost::posix_time::microsec_clock::local_time();
#endif
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images[msg_id], img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images[msg_id], img);
    } else {
      img = message.images[msg_id];
    }

    // Extract image pyramid
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

    // Save!
    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  if (num_images == 1) {
    // feed_monocular(message, 0);
  } else if (num_images == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
  } 
  // else if (!use_stereo) {
  //   parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
  //                   for (int i = range.start; i < range.end; i++) {
  //                     feed_monocular(message, i);
  //                   }
  //                 }));
  // } else {
  //   PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
  //   std::exit(EXIT_FAILURE);
  // }
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);

  std::cout << BLUE << "============================================================" << RESET << std::endl;
  std::cout << std::endl << std::fixed << std::setprecision(6) << "message time  " << message.timestamp << std::endl;

  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Get our image objects for this image
  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);
  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = params_ptr->masks.at(msg_id_left);
  cv::Mat mask_right = params_ptr->masks.at(msg_id_right);

  cv::Mat kp_im_show1 = img_curr.at(cam_id_left);
  cv::cvtColor(kp_im_show1, kp_im_show1, cv::COLOR_GRAY2BGR);
  cv::Mat kp_im_show2 = img_curr.at(cam_id_right);
  cv::cvtColor(kp_im_show2, kp_im_show2, cv::COLOR_GRAY2BGR);

#ifdef TIME_STATISTICS
  rT2 = boost::posix_time::microsec_clock::local_time();
#endif
  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    // Track into the new image
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, cam_id_left, cam_id_right, good_left, good_right,
                             good_ids_left, good_ids_right);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    return;
  }
   std::cout << GREEN << "============================================================" << RESET << std::endl;
  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
#ifdef TIME_STATISTICS
  int pts_before_detect = (int)pts_last[cam_id_left].size();
#endif
  auto pts_left_old = pts_last[cam_id_left];
  auto pts_right_old = pts_last[cam_id_right];
  auto ids_left_old = ids_last[cam_id_left];
  auto ids_right_old = ids_last[cam_id_right];
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old,
                           ids_right_old);
#ifdef TIME_STATISTICS
  rT3 = boost::posix_time::microsec_clock::local_time();
#endif
  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll, mask_rr;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

  // Lets track temporally
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                                     is_left ? pts_left_old : pts_right_old, is_left ? pts_left_new : pts_right_new,
                                     is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                     is_left ? mask_ll : mask_rr);
                  }
                }));
#ifdef TIME_STATISTICS
  rT4 = boost::posix_time::microsec_clock::local_time();
#endif
  //===================================================================================
  //===================================================================================

  // left to right matching
  // TODO: we should probably still do this to reject outliers
  // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
  // std::vector<uchar> mask_lr;
  // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
#ifdef TIME_STATISTICS
  rT5 = boost::posix_time::microsec_clock::local_time();
#endif
  //===================================================================================
  //===================================================================================

  // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty() && mask_rr.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  int num_of_points = 0;
  int nn_count = 0, ll_count = 0, rr_count = 0;
  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > img_left.cols ||
        (int)pts_left_new.at(i).pt.y > img_left.rows)
      continue;
    // See if we have the same feature in the right
    bool found_right = false;
    size_t index_right = 0;
    for (size_t n = 0; n < ids_right_old.size(); n++) {
      if (ids_left_old.at(i) == ids_right_old.at(n)) {
        found_right = true;
        index_right = n;
        break;
      }
    }
    // If it is a good track, and also tracked from left to right
    // Else track it as a mono feature in just the left image
    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows)
        continue;
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(index_right));
      good_ids_left.push_back(ids_left_old.at(i));
      good_ids_right.push_back(ids_right_old.at(index_right));
      num_of_points++;
      cv::circle(kp_im_show1, pts_left_new.at(i).pt, 5/1.5 * fontsize, cv::Scalar(0, 255, 0), cv::FILLED); //BGR
      cv::line(kp_im_show1, pts_right_new.at(index_right).pt, pts_left_new.at(i).pt, cv::Scalar(0, 255, 0), 2/1.5*fontsize);
      cv::line(kp_im_show1, pts_left_new.at(i).pt, pts_left_old.at(i).pt, cv::Scalar(0, 255, 255), 2/1.5*fontsize);

      cv::circle(kp_im_show2, pts_right_new.at(index_right).pt, 5/1.5 * fontsize, cv::Scalar(0, 255, 0), cv::FILLED); //BGR
      cv::line(kp_im_show2, pts_left_new.at(i).pt, pts_right_new.at(index_right).pt, cv::Scalar(0, 255, 0), 2/1.5*fontsize);
      cv::line(kp_im_show2, pts_right_new.at(index_right).pt, pts_right_old.at(index_right).pt, cv::Scalar(0, 255, 255), 2/1.5*fontsize);
      nn_count++;
      // PRINT_DEBUG("adding to stereo - %u , %u\n", ids_left_old.at(i), ids_right_old.at(index_right));
    } 
    else if (mask_ll[i]) 
    {
      good_left.push_back(pts_left_new.at(i));
      good_ids_left.push_back(ids_left_old.at(i));
      cv::circle(kp_im_show1, pts_left_new.at(i).pt, 5/1.5 * fontsize, cv::Scalar(0, 255, 255), cv::FILLED); //BGR
      cv::line(kp_im_show1, pts_left_new.at(i).pt, pts_left_old.at(i).pt, cv::Scalar(0, 255, 255), 2/1.5*fontsize);
      ll_count++;
      // PRINT_DEBUG("adding to left - %u \n",ids_left_old.at(i));
    } 
    if(!mask_ll[i])
    {
      std::stringstream buf;
      buf << ids_left_old[i];
      // cv::putText(kp_im_show1, buf.str(), cv::Point2f(pts_left_old.at(i).pt.x-20,pts_left_old.at(i).pt.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.4*fontsize, cv::Scalar(0, 0, 255), 2/1.5*fontsize);
      cv::circle(kp_im_show1, pts_left_old.at(i).pt, 5/1.5 * fontsize, cv::Scalar(0, 0, 255), cv::FILLED); //BGR
      cv::line(kp_im_show1, pts_left_new.at(i).pt, pts_left_old.at(i).pt, cv::Scalar(0, 0, 255), 2/1.5*fontsize);
    }
    if(!mask_rr[index_right])
    {
      std::stringstream buf;
      buf << ids_right_old[index_right];
      // cv::putText(kp_im_show2, buf.str(), cv::Point2f(pts_right_old.at(index_right).pt.x-20,pts_right_old.at(index_right).pt.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.6*fontsize, cv::Scalar(0, 0, 255), 2/1.5*fontsize);
      cv::circle(kp_im_show2, pts_right_old.at(index_right).pt, 5/1.5 * fontsize, cv::Scalar(0, 0, 255), cv::FILLED); //BGR
      cv::line(kp_im_show2, pts_right_new.at(index_right).pt, pts_right_old.at(index_right).pt, cv::Scalar(0, 0, 255), 2/1.5*fontsize);
    }
  }

  // Loop through all right points
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // See if we have the same feature in the right
    bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
    // If it has not already been added as a good feature, add it as a mono track
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new.at(i));
      good_ids_right.push_back(ids_right_old.at(i));
      cv::circle(kp_im_show2, pts_right_new.at(i).pt, 5/1.5 * fontsize, cv::Scalar(0, 255, 255), cv::FILLED); //BGR
      cv::line(kp_im_show2, pts_right_new.at(i).pt, pts_right_old.at(i).pt, cv::Scalar(0, 255, 255), 2/1.5*fontsize);
      // PRINT_DEBUG("adding to right - %u \n", ids_right_old.at(i));
      rr_count++;
    }
    if(!mask_rr[i])
    {
      std::stringstream buf;
      buf << ids_right_old[i];
      // cv::putText(kp_im_show1, buf.str(), cv::Point2f(pts_right_old.at(i).pt.x-20,pts_right_old.at(i).pt.y-10), cv::FONT_HERSHEY_SIMPLEX, 0.4*fontsize, cv::Scalar(0, 0, 255), 2/1.5*fontsize);
      cv::circle(kp_im_show1, pts_right_old.at(i).pt, 5/1.5 * fontsize, cv::Scalar(0, 0, 255), cv::FILLED); //BGR
    }
  }
  cv::resize(kp_im_show1,kp_im_show1,cv::Size(kp_im_show1.cols/fontsize,kp_im_show1.rows/fontsize));
  cv::resize(kp_im_show2,kp_im_show2,cv::Size(kp_im_show2.cols/fontsize,kp_im_show2.rows/fontsize));
  cv::putText(kp_im_show1, "n/l/r:" + std::to_string((int)nn_count) + "/" + std::to_string((int)ll_count) + "/" + std::to_string((int)rr_count), 
            cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(193, 73, 255), 1.8);
  cv::putText(kp_im_show2, "n/l/r:" + std::to_string((int)nn_count) + "/" + std::to_string((int)ll_count) + "/" + std::to_string((int)rr_count), 
            cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(193, 73, 255), 1.8);
  imshow("Left Frame Track ",kp_im_show1); 
  imshow("Right Frame Track ",kp_im_show2); 
  cv::waitKey (2);

  bool print = false;
  if(print)
  std::cout << "\033[34m" << "[TrakckKLT] database add ";

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
    if(print)
      std::cout << " " << good_ids_left.at(i);
  }
  for (size_t i = 0; i < good_right.size(); i++) {
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
    if(print)
      std::cout << " " << good_ids_right.at(i);
  }
  if(print)
    std::cout << "\033[0m" << std::endl;
    
  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
  }
#ifdef TIME_STATISTICS
  rT6 = boost::posix_time::microsec_clock::local_time();

  //  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id_left].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", (rT5 - rT4).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT6 - rT5).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT6 - rT1).total_microseconds() * 1e-6);
#endif
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();


  cv::Mat kp_im_show1 = img_curr.at(cam_id_left);
  cv::cvtColor(kp_im_show1, kp_im_show1, cv::COLOR_GRAY2BGR);
  cv::Mat kp_im_show2 = img_curr.at(cam_id_right);
  cv::cvtColor(kp_im_show2, kp_im_show2, cv::COLOR_GRAY2BGR);
  int stereo_count = 0, left_count = 0, right_count = 0;
  int left_need = 0, right_need = 0;
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
      cv::rectangle(kp_im_show1, pt1, pt2, cv::Scalar(75,75,75), cv::FILLED);
      cv::rectangle(kp_im_show1, pt1, pt2, cv::Scalar(255));
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();
  left_need = num_features - (int)pts1.size();
  // LEFT: if we need features we should extract them in the current frame
  // LEFT: we will also try to track them from this frame over to the right frame
  // LEFT: in the case that we have two features that are the same, then we should merge them
  if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts0_ext;
    // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid0.cols; x++) {
      for (int y = 0; y < grid_2d_grid0.rows; y++) {
        if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);
    left_count = pts0_ext.size();

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) 
    {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
    }

    // TODO: Project points from the left frame into the right frame
    // TODO: This will not work for large baseline systems.....
    // TODO: If we had some depth estimates we could do a better projection
    // TODO: Or project and search along the epipolar line??
    std::vector<cv::KeyPoint> kpts1_new;
    std::vector<cv::Point2f> pts1_new;

    // cv::Point2f un_pts0 = camera_calib.at(cam_id_left)->undistort_cv(pts0_new.at(i));
    // cv::Point2f un_pts1 = camera_calib.at(cam_id_right)->undistort_cv(pts1_new.at(i));
    // Eigen::Vector3d un_p1 = K_C1 * Eigen::Vector3d(un_pts0.x, un_pts0.y, 1);
    // Eigen::Vector3d un_p2 = K_C2 * Eigen::Vector3d(un_pts1.x, un_pts1.y, 1);
    // Eigen::Vector3d abc = un_p1.transpose() * F12;

    kpts1_new = kpts0_new;
    pts1_new = pts0_new;

    // If we have points, do KLT tracking to get the valid projections into the right image
    if (!pts0_new.empty()) {

      // Do our KLT tracking from the left to the right frame of reference
      // Note: we have a pretty big window size here since our projection might be bad
      // Note: but this might cause failure in cases of repeated textures (eg. checkerboard)
      std::vector<uchar> mask;
      // perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
      std::vector<float> error;
      cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

      // int pyr_levels = 3;
      // cv::Size win_size = cv::Size(15, 15);
      cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, cv::Size(15, 15), 3, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);

      // Loop through and record only ones that are valid
      for (size_t i = 0; i < pts0_new.size(); i++) {

        // Check to see if the feature is out of bounds (oob) in either image
        bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img0pyr.at(0).cols || (int)pts0_new.at(i).y < 0 ||
                         (int)pts0_new.at(i).y >= img0pyr.at(0).rows);
        bool oob_right = ((int)pts1_new.at(i).x < 0 || (int)pts1_new.at(i).x >= img1pyr.at(0).cols || (int)pts1_new.at(i).y < 0 ||
                          (int)pts1_new.at(i).y >= img1pyr.at(0).rows);

        // Check to see if it there is already a feature in the right image at this location
        //  1) If this is not already in the right image, then we should treat it as a stereo
        //  2) Otherwise we will treat this as just a monocular track of the feature
        // TODO: we should check to see if we can combine this new feature and the one in the right
        // TODO: seems if reject features which overlay with right features already we have very poor tracking perf
        if (!oob_left && !oob_right && mask[i] == 1) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          kpts1_new.at(i).pt = pts1_new.at(i);

          cv::Point2f un_pts0 = camera_calib.at(cam_id_left)->undistort_cv(pts0_new.at(i));
          cv::Point2f un_pts1 = camera_calib.at(cam_id_right)->undistort_cv(pts1_new.at(i));
          Eigen::Vector3d un_p1 = K_C1 * Eigen::Vector3d(un_pts0.x, un_pts0.y, 1);
          Eigen::Vector3d un_p2 = K_C2 * Eigen::Vector3d(un_pts1.x, un_pts1.y, 1);

          Eigen::Vector3d abc = un_p1.transpose() * F12;
          const double num = abc.x() * un_p2.x() + abc.y() * un_p2.y() + abc.z();
          const double den = abc.x() * abc.x() + abc.y() * abc.y();

          double dsqr;
          if(den!=0)
              dsqr = num*num/den;
          else
              dsqr = 1000;


          std::stringstream buf;
          buf << dsqr;

          if (dsqr < 5.992 && un_p1.x() > un_p2.x() && std::abs(un_p1.y() - un_p2.y() < 5.992))  //3.84 5.992  7.81   10.64 15.51
          // if(1)
          {  
            // cv::putText(kp_im_show1, buf.str(), cv::Point2f(pts0_new.at(i).x-20,pts0_new.at(i).y-10), cv::FONT_HERSHEY_SIMPLEX, 0.4*fontsize, cv::Scalar(0, 255, 255), 1*fontsize);
            cv::circle(kp_im_show1, pts0_new.at(i), 3 * fontsize, cv::Scalar(0, 255, 0), cv::FILLED); //BGR
            cv::line(kp_im_show1, pts1_new.at(i), pts0_new.at(i), cv::Scalar(0, 255, 255), 1*fontsize);
            cv::circle(kp_im_show2, pts1_new.at(i), 3 * fontsize, cv::Scalar(0, 255, 0), cv::FILLED); //BGR

            // append the new uv coordinate
            pts0.push_back(kpts0_new.at(i));
            pts1.push_back(kpts1_new.at(i));
            // move id forward and append this new point
            size_t temp = ++currid;
            ids0.push_back(temp);
            ids1.push_back(temp);
            stereo_count++;
          }
          else
          {
            // update the uv coordinates
            kpts0_new.at(i).pt = pts0_new.at(i);
            // append the new uv coordinate
            pts0.push_back(kpts0_new.at(i));
            // move id forward and append this new point
            size_t temp = ++currid;
            ids0.push_back(temp);
            // cv::putText(kp_im_show1, buf.str(), cv::Point2f(pts0_new.at(i).x-20,pts0_new.at(i).y-10), cv::FONT_HERSHEY_SIMPLEX, 0.4*fontsize, cv::Scalar(210, 210, 0), 1*fontsize);
            cv::circle(kp_im_show1, pts0_new.at(i), 3 * fontsize, cv::Scalar(210, 210, 0), cv::FILLED); //BGR
            // cv::line(kp_im_show1, kpts1_new.at(i).pt, kpts0_new.at(i).pt, cv::Scalar(128, 128, 0), 1*fontsize);
            // cv::line(kp_im_show1, pts1_new.at(i), pts0_new.at(i), cv::Scalar(210, 210, 0), 1*fontsize);
          }

        } else {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
        }
      }
      // cv::resize(kp_im_show1,kp_im_show1,cv::Size(kp_im_show1.cols/fontsize,kp_im_show1.rows/fontsize));
      // cv::putText(kp_im_show1, "Tra/Det/Req:" + std::to_string((int)count) + "/" + std::to_string((int)pts0_ext.size()) + "/" + std::to_string((int)num_needed), cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
      //   cv::Scalar(0, 235, 0), 1.5);
      // imshow("Left Point Track ",kp_im_show1); 
      // cv::resize(kp_im_show2,kp_im_show2,cv::Size(kp_im_show2.cols/fontsize,kp_im_show2.rows/fontsize));
      // imshow("Right Point",kp_im_show2);
    }
  }

  // RIGHT: Now summarise the number of tracks in the right image
  // RIGHT: We will try to extract some monocular features if we have the room
  // RIGHT: This will also remove features if there are multiple in the same location
  cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid1(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  cv::Mat mask1_updated = mask0.clone();
  it0 = pts1.begin();
  it1 = ids1.begin();
  while (it0 != pts1.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x1);
    int y_grid = std::floor(kpt.pt.y / size_y1);
    if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    // NOTE: if it is *not* a stereo point, then we will not delete the feature
    // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }

    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img1pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img1pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // RIGHT: if we need features we should extract them in the current frame
  // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  int num_featsneeded_1 = num_features - (int)pts1.size();
  right_need = num_featsneeded_1;
  if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts1_ext;
    // Grider_FAST::perform_griding(img1pyr.at(0), mask1, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask1_grid;
    cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid1.cols; x++) {
      for (int y = 0; y < grid_2d_grid1.rows; y++) {
        if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts1_ext;
    Grider_GRID::perform_griding(img1pyr.at(0), mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);
    right_count = pts1_ext.size();
    // Now, reject features that are close a current feature
    for (auto &kpt : pts1_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
  // 193, 73, 255
  cv::resize(kp_im_show1,kp_im_show1,cv::Size(kp_im_show1.cols/fontsize,kp_im_show1.rows/fontsize));
  cv::putText(kp_im_show1, "Tra/Det/Req:" + std::to_string((int)stereo_count) + "/" + std::to_string((int)left_count) + "/" + std::to_string((int)left_need), 
              cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(193, 73, 255), 1.8);
  imshow("Left Point Track ",kp_im_show1);

  cv::resize(kp_im_show2,kp_im_show2,cv::Size(kp_im_show2.cols/fontsize,kp_im_show2.rows/fontsize));
  cv::putText(kp_im_show2, "Tra/Det/Req:" + std::to_string((int)stereo_count) + "/" + std::to_string((int)right_count) + "/" + std::to_string((int)right_need), 
            cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(193, 73, 255), 1.8);
  imshow("Right Point",kp_im_show2);
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out) {
//  auto T0 = boost::posix_time::microsec_clock::local_time();
  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }
//  auto T1 = boost::posix_time::microsec_clock::local_time();
  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.01);//tune: change 30 to 15
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);//time cost! 50%
//  auto T2 = boost::posix_time::microsec_clock::local_time();
  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.99, mask_rsc);//time cost! 45%
//  auto T3 = boost::posix_time::microsec_clock::local_time();
  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
//  auto T4 = boost::posix_time::microsec_clock::local_time();
//  PRINT_ALL("[matching]: %.4f seconds to calculate1\n", (T1 - T0).total_microseconds() * 1e-6);
//  PRINT_ALL("[matching]: %.4f seconds to calculate2\n", (T2 - T1).total_microseconds() * 1e-6);
//  PRINT_ALL("[matching]: %.4f seconds to calculate3\n", (T3 - T2).total_microseconds() * 1e-6);
//  PRINT_ALL("[matching]: %.4f seconds to calculate4\n", (T4 - T0).total_microseconds() * 1e-6);
}
