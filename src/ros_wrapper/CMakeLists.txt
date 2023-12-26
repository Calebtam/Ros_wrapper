cmake_minimum_required(VERSION 2.8)
project(wrapper)

# Change this to your installation of ORB-SLAM3. Default is ~/
# set(ORB_SLAM3_DIR
#    $ENV{HOME}/Packages/ORB_SLAM3
# )

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package (catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   cv_bridge
   image_transport
   tf
   sensor_msgs
   message_generation
)

find_package(OpenCV REQUIRED)
# MESSAGE("OPENCV VERSION:")
MESSAGE("OPENCV VERSION:" ${OpenCV_VERSION})

find_package(Eigen3 REQUIRED)
# MESSAGE("Eigen3 VERSION:")
MESSAGE("Eigen3 VERSION:" ${Eigen3_VERSION})

find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

catkin_package (
   CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge image_transport tf sensor_msgs message_runtime
   LIBRARIES {PROJECT_NAME}
)

include_directories(
   ${PROJECT_SOURCE_DIR}/include
   ${EIGEN3_INCLUDE_DIR}
   ${catkin_INCLUDE_DIRS}
   ${Pangolin_INCLUDE_DIRS}
   ${Sophus_INCLUDE_DIRS}
)

include_directories(${PROJECT_SOURCE_DIR}/../ov_core/src)
MESSAGE("include_directories:")
MESSAGE(${PROJECT_SOURCE_DIR})
# Build ROS nodes

set(LIBS
   ${EIGEN3_LIBS}
   ${catkin_LIBRARIES}
   ${Pangolin_LIBRARIES}
   ${OpenCV_LIBS}
   ${Sophus_LIBS}
)

# # mono node
# add_executable (${PROJECT_NAME}_mono
#    src/mono_node.cc
#    src/common.cc
# )
# target_link_libraries(${PROJECT_NAME}_mono
#    ${LIBS}
# )


# # stereo node
# add_executable (${PROJECT_NAME}_stereo
#    src/stereo_node.cc
#    src/common.cc
# )
# target_link_libraries(${PROJECT_NAME}_stereo
#    ${LIBS}
# )


# # mono-inertial node
# add_executable (${PROJECT_NAME}_mono_inertial
#    src/mono_inertial_node.cc
#    src/common.cc
# )
# target_link_libraries(${PROJECT_NAME}_mono_inertial
#    ${LIBS}
# )


# 前端测试 node
add_executable (${PROJECT_NAME}_stereo
   src/stereo_node.cc
   src/later/common.cc
   src/later/State.cpp
   ../ov_core/src/track/TrackBase.cpp
   ../ov_core/src/track/TrackKLT.cpp
   ../ov_core/src/options/VioManagerOptions.cpp
)
target_link_libraries(${PROJECT_NAME}_stereo
   ${LIBS} ov_core_lib
)

# IMU测试 node
add_executable (${PROJECT_NAME}_inertial
   src/inertial_node.cc
   src/later/common.cc
   src/later/State.cpp
   ../ov_core/src/track/TrackBase.cpp
   ../ov_core/src/track/TrackKLT.cpp
   ../ov_core/src/options/VioManagerOptions.cpp
)
target_link_libraries(${PROJECT_NAME}_inertial
   ${LIBS} ov_msckf_lib ov_core_lib
)

# # RGB-D node
# add_executable (${PROJECT_NAME}_rgbd
#    src/rgbd_node.cc
#    src/common.cc
# )
# target_link_libraries(${PROJECT_NAME}_rgbd
#    ${LIBS}
# )