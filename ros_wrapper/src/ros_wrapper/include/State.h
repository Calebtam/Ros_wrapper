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

#ifndef OV_MSCKF_STATE_H
#define OV_MSCKF_STATE_H

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "cam/CamBase.h"
#include "feat/FeatureDatabase.h"
#include "options/StateOptions.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "types/Type.h"
#include "types/Vec.h"

namespace ov_msckf {

/**
 * @brief State of our filter
 *
 * This state has all the current estimates for the filter.
 * This system is modeled after the MSCKF filter, thus we have a sliding window of clones.
 * We additionally have more parameters for online estimation of calibration and SLAM features.
 * We also have the covariance of the system, which should be managed using the StateHelper class.
 */
class State {

public:
  /**
   * @brief Default Constructor (will initialize variables to defaults)
   * @param options_ Options structure containing filter options
   */
  State(StateOptions &options_);

  ~State() {}

  /**
   * @brief Will return the timestep that we will marginalize next.
   * As of right now, since we are using a sliding window, this is the oldest clone.
   * But if you wanted to do a keyframe system, you could selectively marginalize clones.
   * @return timestep of clone we will marginalize
   */
  double margtimestep() {
    std::lock_guard<std::mutex> lock(_mutex_state);
    double time = INFINITY;
    for (const auto &clone_imu : _clones_IMU) {
      if (clone_imu.first < time) {
        time = clone_imu.first;
      }
    }
    return time;
  }

  /**
   * @brief Will return the timestep that we will marginalize next.
   * As of right now, this is the keyframe whose tracked map features are lowest as well as lower than threshold or
   * the max_keyframe_size has been reached.
   * @return timestep of clone we will marginalize
   */
  std::vector<double> marg_keyframe_timestep(const std::shared_ptr<ov_core::FeatureDatabase>& databse) {
    double time = INFINITY;
    //condition 1: marg the keyframe whose tracked map features are lowest. implement in FeatureDatabase
    std::vector<double> marg_times;
    marg_times = databse->get_marg_keyframes();
    if(marg_times.size() > 0)
      return marg_times;
    //condition 2: marg oldeest keyframe if the max_keyframe_size has been reached
    for (const auto &kf_imu : _keyframes_IMU) {
      if (kf_imu.first < time) {
        time = kf_imu.first;
        marg_times.push_back(time);
      }
    }
    return marg_times;
  }

  /**
   * @brief Calculates the current max size of the covariance
   * @return Size of the current covariance matrix
   */
  int max_covariance_size() { return (int)_Cov.rows(); }

  /// Mutex for locking access to the state
  std::mutex _mutex_state;

  /// Current timestamp (should be the last update time!)
  double _timestamp = -1;

  /// Struct containing filter options
  StateOptions _options;

  /// Pointer to the "active" IMU state (q_GtoI, p_IinG, v_IinG, bg, ba)
  std::shared_ptr<ov_type::IMU> _imu;

  /// Map between imaging times and clone poses (q_GtoIi, p_IiinG)
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> _clones_IMU;

  /// Map between imaging times and keyframe poses (q_GtoIi, p_IiinG), not used now
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> _keyframes_IMU;

  /// Our current set of SLAM features (3d positions), key is Unique ID of this feature
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> _features_SLAM;

  /// Our current set of map features (3d positions), key is Unique ID of this feature, not used now
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> _features_Map;

  /// Time offset base IMU to camera (t_imu = t_cam + t_off)
  std::shared_ptr<ov_type::Vec> _calib_dt_CAMtoIMU;

  /// Calibration poses for each camera (R_ItoC, p_IinC)
  std::unordered_map<size_t, std::shared_ptr<ov_type::PoseJPL>> _calib_IMUtoCAM;

  /// Camera intrinsics
  std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> _cam_intrinsics;

  /// Camera intrinsics camera objects
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> _cam_intrinsics_cameras;

private:
  // Define that the state helper is a friend class of this class
  // This will allow it to access the below functions which should normally not be called
  // This prevents a developer from thinking that the "insert clone" will actually correctly add it to the covariance
  friend class StateHelper;

  /// Covariance of all active variables
  Eigen::MatrixXd _Cov;

  /// Vector of variables, order is _imu, _calib_dt_CAMtoIMU, _calib_IMUtoCAM, _cam_intrinsics, imu clone poses and slam features.
  /// imu clone poses and slam features are out of order.
  std::vector<std::shared_ptr<ov_type::Type>> _variables;
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_H