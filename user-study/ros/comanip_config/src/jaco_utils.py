#!/usr/bin/env python

import rospy

import numpy as np
from math import pi
from tf import TransformListener
from tf.transformations import euler_from_quaternion

from kinova_msgs.srv import HomeArm

DXMAX = 0.4
DTHETAMAX = 0.1 

class JacoControl():
    def __init__(self):
        self.tf = TransformListener()
        rospy.wait_for_service('/j2s7s300_driver/in/home_arm')
        self.homecall = rospy.ServiceProxy('/j2s7s300_driver/in/home_arm', HomeArm)

    def get_current_pose(self):
        self.tf.waitForTransform("/j2s7s300_link_base", "/j2s7s300_ee_link", rospy.Time(0), rospy.Duration(1))
        self.curpos, self.curquat = self.tf.lookupTransform("/j2s7s300_link_base", "/j2s7s300_ee_link", rospy.Time(0))
        self.curort = euler_from_quaternion(self.curquat)
