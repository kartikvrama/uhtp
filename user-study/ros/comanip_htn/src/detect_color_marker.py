#!/usr/bin/env python

import cv2
import json
import rospy
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

'''
    LEFT AND RIGHT ARE FROM THE USER'S REFERENCE POINT!
'''

PARAMS = {'image_topic': '/front/rgb/image_raw',
          'color_right_marker_coords': [[466, 1185],
                                  [545, 1185],
                                  [424, 1262],
                                  [509, 1260]],
          'color_left_marker_coords': [[580, 1180],
                                   [663, 1180],
                                   [547, 1260],
                                   [634, 1259]],
           'moving_average_window': 5,
           'distance_threshold': 0.1,
           'color_right_rgb_template': [0.15811912, 0.14189593, 0.13626949],#[0.50216269, 0.46661743, 0.41908685],
           'color_left_rgb_template': [0.21875445, 0.19369292, 0.18960134] #[0.29824702, 0.54730765, 0.5478398],
        #    'color_right_rgb_template_file': '/home/comanip/adapcomanip_ws/src/comanip_htn/config/avg_color_right_rgb.npy',
        #    'color_left_rgb_template_file': '/home/comanip/adapcomanip_ws/src/comanip_htn/config/avg_color_left_rgb.npy',
           }

def draw_parallelogram_mask(image_dim, ul, ur, ll, lr):
    ''' 
        Given x, y coordinates of 4 corners of object, 
        make a parallolgram mask '''
    
    lower_d0 = min(ul[1], ur[1])
    upper_d0 = max(ll[1], lr[1])

    # Equation: x = m*y + c
    m_left = 1.0*(ul[0] - ll[0])/(ul[1] - ll[1])
    c_left = ul[0] - ul[1]*m_left

    m_right = 1.0*(ur[0] - lr[0])/(ur[1] - lr[1])
    c_right = ur[0] - ur[1]*m_right

    mask = np.zeros(image_dim[:2], dtype='int32')

    for i in range(lower_d0, upper_d0 + 1):
        d1_left = int(m_left*i + c_left)
        d1_right = int(m_right*i + c_right) + 1
        mask[i, d1_left:d1_right] = 1

    return mask

class DetectColorMarker:

    publish_msg = String()

    def __init__(self, params):

        self.params = params
        self.bridge = CvBridge()

        img = self.get_image()
        self.image_dim = img.shape

        self.color_right_mask = draw_parallelogram_mask(self.image_dim[:2], *params['color_right_marker_coords'])
        self.color_left_mask = draw_parallelogram_mask(self.image_dim[:2], *params['color_left_marker_coords'])

        # Detecting marker
        self.color_right_rgb_template = np.array(params['color_right_rgb_template']).reshape(-1, 3)
        self.color_left_rgb_template = np.array(params['color_left_rgb_template']).reshape(-1, 3)

        self.ma_window = params['moving_average_window']
        self.dist_threshold = params['distance_threshold']

        self.maqueue = np.empty([0, 2])

        self.publisher = rospy.Publisher('color_marker_detector', String, queue_size=1)
        self.rate = rospy.Rate(5)

        rospy.loginfo('Color marker detector active!')

        rospy.Subscriber(params['image_topic'], Image, self.callback)
        rospy.spin()

    def get_image(self):
        msg = rospy.wait_for_message(self.params['image_topic'], Image)
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.loginfo(e)
        return img

    def plot_masked(self, img):
        color_rightmask = np.tile(self.color_right_mask[:, :, np.newaxis], (1,1,3))
        color_leftmask = np.tile(self.color_left_mask[:, :, np.newaxis], (1,1,3))

        color_rightmask_img = (255*color_rightmask*(img.astype('float32')/255)).astype('uint8')
        color_leftmask_img = (255*color_leftmask*(img.astype('float32')/255)).astype('uint8')

        cv2.imshow('imgs', np.hstack([color_rightmask_img, color_leftmask_img])[1000:])
        cv2.waitKey(0)

    def calculate_avg_rgb(self):
        avg_color_right_rgb = np.zeros(3)
        avg_color_left_rgb = np.zeros(3)

        for i in range(4):
            img = self.get_image()
            print('Approve this image?')
            self.plot_masked(img)

            color_rightrgb, color_leftrgb = self.get_avg_rgb(img)
            avg_color_right_rgb += color_rightrgb
            avg_color_left_rgb += color_leftrgb

            raw_input('Done with {}: Next?'.format(i+1))
            print('')

        avg_color_right_rgb /= 4
        avg_color_left_rgb /= 4

        return avg_color_right_rgb, avg_color_left_rgb

    def get_avg_rgb(self, img):
        color_rightmask = np.tile(self.color_right_mask[:, :, np.newaxis], (1,1,3))
        color_leftmask = np.tile(self.color_left_mask[:, :, np.newaxis], (1,1,3))

        color_rightmask_img = color_rightmask*(img.astype('float32'))
        color_leftmask_img = color_leftmask*(img.astype('float32'))

        return np.mean(color_rightmask_img, axis=(0, 1)), np.mean(color_leftmask_img, axis=(0, 1))

    def callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.loginfo(e)

        color_rightrgb, color_leftrgb = self.get_avg_rgb(img)

        d_color_right = np.linalg.norm(color_rightrgb - self.color_right_rgb_template)
        d_color_left = np.linalg.norm(color_leftrgb - self.color_left_rgb_template)

        self.maqueue = np.vstack([self.maqueue, [[d_color_right, d_color_left]]])

        if len(self.maqueue) > self.ma_window:
            self.maqueue = np.delete(self.maqueue, 0, 0)
            avg_dist = np.average(self.maqueue, axis=0)

            # Detecting black square
            result = (avg_dist > self.dist_threshold).astype('int32').astype('str')

            # Detecting colored square
            # result = (avg_dist < self.dist_threshold).astype('int32').astype('str')

            result_dict = {'color_right': result[0], 'color_left': result[1]}

            self.publish_msg.data = json.dumps(result_dict)
            self.publisher.publish(self.publish_msg)
            self.rate.sleep()

        # rospy.loginfo('color_left dist: {:6.2f}, color_right dist: {:6.2f}'.format(d_color_left, d_color_right))


def main():
    rospy.init_node('DetectColorMarker')
    DetectColorMarker(PARAMS)

if __name__ == '__main__':
    main()