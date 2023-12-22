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

#ifndef OV_CORE_TRACK_KLT_H
#define OV_CORE_TRACK_KLT_H

#include "TrackBase.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
namespace ov_core {

/**
 * @brief KLT tracking of features.
 *
 * This is the implementation of a KLT visual frontend for tracking sparse features.
 * We can track either monocular cameras across time (temporally) along with
 * stereo cameras which we also track across time (temporally) but track from left to right
 * to find the stereo correspondence information also.
 * This uses the [calcOpticalFlowPyrLK](https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp)
 * OpenCV function to do the KLT tracking.
 */
class TrackKLT : public TrackBase {

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * @param fast_threshold FAST detection threshold
   * @param gridx size of grid in the x-direction / u-direction
   * @param gridy size of grid in the y-direction / v-direction
   * @param minpxdist features need to be at least this number pixels away from each other
   */
  explicit TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, std::shared_ptr<ov_msckf::VioManagerOptions>& params)
      : TrackBase(cameras, numfeats, numaruco, params), threshold(params->fast_threshold), grid_x(params->grid_x), grid_y(params->grid_y),
        min_px_dist(params->min_px_dist) {

            Eigen::Matrix4d T_C1toC0 = Eigen::Matrix4d::Identity();

            Eigen::Matrix3d t12x,k;  // tc1c2 SkewSymmetricMatrix
            Eigen::Matrix4d Tc1c2; 
            

            // t12x << 0  , 0        , 0           ,
            //         0  , 0        ,  -1 * 0.095 ,
            //         0  , 0.095 ,             0  ;

            // Tc1c2 << 1,  0,  0, 0.095, 
            //         0,  1,  0,    0,
            //         0,  0,  1,    0, 
            //         0,  0,  0,    1;

            // // cv::Matx<double, 3, 3> K1 = cameras[0]->get_K();
            // // cv::Matx<double, 3, 3> K2 = cameras[0]->get_K();
            // // cv::eigen2cv(k, K1);
            // // Eigen::Matrix3d k;
            // k << 428.3537922602939, 0,               422.67752703029004, 
            //       0,              428.3201454661401, 236.60599791251943,
            //       0,              0,                  1 ;
            // (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
            Eigen::MatrixXd KK1_C1 = params->camera_intrinsics[0]->get_value();
            K_C1 << KK1_C1(0),           0,               KK1_C1(2), 
                    0,              KK1_C1(1),         KK1_C1(3),
                    0,              0,                  1 ;
            // cv::Matx<double, 3, 3> KK2 = params->camera_intrinsics[1]->get_K();
            Eigen::MatrixXd KK2_C2 = params->camera_intrinsics[1]->get_value();
            K_C2 << KK2_C2(0),           0,               KK2_C2(2), 
                    0,              KK2_C2(1),         KK2_C2(3),
                    0,              0,                  1 ;

            Eigen::Matrix4d T_C1_C2 = Eigen::Matrix4d::Identity();
            T_C1_C2.block(0, 0, 3, 3) = ov_core::quat_2_Rot(params->camera_extrinsics.at(10).block(0, 0, 4, 1));
            T_C1_C2.block(0, 3, 3, 1) = params->camera_extrinsics.at(10).block(4, 0, 3, 1);
            Eigen::Matrix3d t_c1_c2_x = ov_core::skew_x(T_C1_C2.block(0, 3, 3, 1));
            E12 = t_c1_c2_x * T_C1_C2.block(0,0,3,3);
            F12 = K_C1.transpose().inverse() * E12 * K_C2.inverse();
            
            std::cout << " K_C1 " << std::endl << K_C1 << std::endl;
            std::cout << " K_C2 " << std::endl << K_C2 << std::endl;
            // std::cout << " K_C1.transpose().inverse() " << std::endl << K_C1.transpose().inverse() << std::endl;
            // std::cout << " K_C2.inverse() " << std::endl << K_C2.inverse() << std::endl;
            std::cout << " T_C1_C2 " << std::endl << T_C1_C2.matrix() << std::endl;
            std::cout << " E12 = tx * R = " << std::endl << E12 << std::endl;
            std::cout << " F12 = " << std::endl << F12 << std::endl;

            // T_C1toC0.block(0, 0, 3, 3) = ov_core::quat_2_Rot(params->camera_extrinsics.at(10).block(0, 0, 4, 1));
            // T_C1toC0.block(0, 3, 3, 1) = params->camera_extrinsics.at(10).block(4, 0, 3, 1);
            // // T_C1toC0.block(0, 0, 3, 3) = ov_core::quat_2_Rot(params->camera_extrinsics.at(10).block(0, 0, 4, 1)).transpose();
            // // T_C1toC0.block(0, 3, 3, 1) = -T_C1toC0.block(0, 0, 3, 3) * params->camera_extrinsics.at(10).block(4, 0, 3, 1);
            // cv::eigen2cv(T_C1toC0, mat_44d);



            // cv::Size c1_imageSize(cameras[0]->w(), cameras[0]->h());
            // cv::Size c2_imageSize(cameras[1]->w(), cameras[1]->h());
            // //    当alpha=1时，所有像素均保留，但存在黑色边框
            // //    当alpha=0时，损失最多的像素，没有黑色边框
            // //    const double alpha = 1;
            // const double alpha = 1;  //

            // cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(cameras[0]->get_K(), cameras[0]->get_D(), c1_imageSize, alpha, c1_imageSize, 0);
            // initUndistortRectifyMap(cameras[0]->get_K(), cameras[0]->get_D(), cv::Mat(), NewCameraMatrix, c1_imageSize, CV_16SC2, c1_map1, c1_map2);
            // cv::Mat NewCameraMatrix1 = getOptimalNewCameraMatrix(cameras[1]->get_K(), cameras[1]->get_D(), c2_imageSize, alpha, c2_imageSize, 0);
            // initUndistortRectifyMap(cameras[1]->get_K(), cameras[1]->get_D(), cv::Mat(), NewCameraMatrix1, c2_imageSize, CV_16SC2, c2_map1, c2_map2);
        }

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right, int a);
  /**
   * @brief Detects new features in the current image
   * @param img0pyr image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param pts0 vector of currently extracted keypoints in this image
   * @param ids0 vector of feature ids for each currently extracted keypoint
   *
   * Given an image and its currently extracted features, this will try to add new features if needed.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   * Passed images should already be grayscaled.
   */
  void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                   std::vector<size_t> &ids0);

  /**
   * @brief Detects new features in the current stereo pair
   * @param img0pyr left image we will detect features on (first level of pyramid)
   * @param img1pyr right image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param mask1 mask which has what ROI we do not want features in
   * @param cam_id_left first camera sensor id
   * @param cam_id_right second camera sensor id
   * @param pts0 left vector of currently extracted keypoints
   * @param pts1 right vector of currently extracted keypoints
   * @param ids0 left vector of feature ids for each currently extracted keypoint
   * @param ids1 right vector of feature ids for each currently extracted keypoint
   *
   * This does the same logic as the perform_detection_monocular() function, but we also enforce stereo contraints.
   * So we detect features in the left image, and then KLT track them onto the right image.
   * If we have valid tracks, then we have both the keypoint on the left and its matching point in the right image.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   */
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1, int & data);
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1, float a);
  /**
   * @brief KLT track between two images, and do RANSAC afterwards
   * @param img0pyr starting image pyramid
   * @param img1pyr image pyramid we want to track too
   * @param pts0 starting points
   * @param pts1 points we have tracked
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param mask_out what points had valid tracks
   *
   * This will track features from the first image into the second image.
   * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
   * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
   */
  void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);

  // Parameters for our FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // How many pyramid levels to track
  int pyr_levels = 3;
  cv::Size win_size = cv::Size(15, 15);

  // Last set of image pyramids
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;

  // cv::Mat c1_map1, c1_map2;
  // cv::Mat c2_map1, c2_map2;
  Eigen::Matrix3d F12, E12;
  Eigen::Matrix3d K_C1, K_C2;   // F Matrix
  float fontsize = 1;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
