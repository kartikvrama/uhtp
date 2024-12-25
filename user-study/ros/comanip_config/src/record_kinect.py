#!/usr/bin/env python

import rospy
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

FPS = 5.0

class RecordKinect:

    def __init__(self):
        self.bridge = CvBridge()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, FPS, (1080,720))

        rospy.init_node('RecordKinect')
        self.rate = rospy.Rate(FPS)
        rospy.Subscriber('/rgb/image_raw', Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        saveimg = cv2.resize(cv_image, (1080, 720), interpolation=cv2.INTER_LINEAR)
        self.out.write(saveimg)            

        try:
            rospy.loginfo('Recording...')
            self.rate.sleep()
        except rospy.ROSInterruptException as e:
            self.out.release()
            print('Video released!')            

if __name__ == '__main__':
    RecordKinect()
    rospy.spin()
