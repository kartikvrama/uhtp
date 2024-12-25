#! /usr/bin/env python
import os
import yaml
import rospy, rospkg

from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, RegionOfInterest

def camera_info_from_yaml(folder):
    with open(os.path.join(folder, 'config/calib_params.yaml'), 'r') as f:
        params = yaml.load(f)
    msg = CameraInfo()
    msg.header = Header()
    msg.header.frame_id = "rgb_camera_link"    
    msg.height = params['image_height']
    msg.width = params['image_width']
    msg.distortion_model= "rational_polynomial"

    msg.K = params['camera_matrix']['data']
    msg.D = params['distortion_coefficients']['data'] + [0.0, 0.0, 0.0]
    msg.R = params['rectification_matrix']['data']
    msg.P = params['projection_matrix']['data']
    msg.roi = RegionOfInterest()  
    
    return msg
    

if __name__ == '__main__':
    rospy.init_node('publish_intrinsic_calibration', anonymous=False)
    pub = rospy.Publisher('/rgb/new_camera_info', CameraInfo, queue_size=1)
    rate = rospy.Rate(100)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('comanip_description')

    msg = camera_info_from_yaml(package_path)
    rospy.loginfo('Publishing new intrinsic parameters!')

    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()
    rospy.loginfo('Shutting down...')

