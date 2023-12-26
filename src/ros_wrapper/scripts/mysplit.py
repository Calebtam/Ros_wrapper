#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu

def imu_callback(imu_data):
   # Publish the split messages to two different topics
    pub1.publish(imu_data)
    pub2.publish(imu_data)

if __name__ == "__main__":
   
    rospy.init_node('imu_splitter')

    # Create publishers for the two topics
    pub1 = rospy.Publisher('/imu0', Imu, queue_size=10)
    pub2 = rospy.Publisher('/imu1', Imu, queue_size=10)

    # Subscribe to the IMU topic
    rospy.Subscriber('/camera/imu', Imu, imu_callback)

    # rospy.Node('split').run()
    rospy.spin()

