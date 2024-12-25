#!/usr/bin/env python

import tf
import rospy
import roslib
import actionlib

import numpy as np
from math import pi
from copy import deepcopy

from tf import TransformListener
from tf.transformations import euler_from_quaternion, euler_matrix, quaternion_matrix, quaternion_from_matrix#, quaternion_from_euler

from geometry_msgs.msg import Pose
from comanip_htn.msg import DetectDrillAction, DetectDrillResult, DetectDrillFeedback
from visualization_msgs.msg import MarkerArray


class DetectDrill:

    ETA = 0.05
    HANDTIP_LEFT = 9
    HANDTIP_RIGHT = 16
    DIST_THRESHOLD = 0.05

    _result = DetectDrillResult()
    _feedback = DetectDrillFeedback()

    def __init__(self):
        self.listener = tf.TransformListener()
        self.server = actionlib.SimpleActionServer('detectdrill', DetectDrillAction, self.execute, False)
        self.server.register_preempt_callback(self.kill)
        self.server.start()
        rospy.loginfo('DetectDrill service ready!')

    def get_hand_pos(self):
        hand_pos = -np.inf*np.ones((2,3))
        skeleton_msg = rospy.wait_for_message('/front/body_tracking_data', MarkerArray)

        if len(skeleton_msg.markers) == 32:
            hand_markers = [skeleton_msg.markers[self.HANDTIP_LEFT], skeleton_msg.markers[self.HANDTIP_RIGHT]]

            for i, m in enumerate(hand_markers):
                if m.pose.position.x < 1:
                    hand_pos[i, :] = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])

        return hand_pos

    def get_marker_pos(self):
        markerA, _ = self.listener.lookupTransform('front_depth_camera_link', 'ARmarker_15', 
                                                   rospy.Time(0))
        markerB, _ = self.listener.lookupTransform('front_depth_camera_link', 'ARmarker_25', 
                                                   rospy.Time(0))

        return np.array(markerA), np.array(markerB)

    def return_result(self, done, pred):
        pred = str(pred)
        if not done:
            self._feedback.pred = pred
            self.server.publish_feedback(self._feedback)

        else:
            self._result.pred = pred
            self.server.set_succeeded(self._result)

    def kill(self):
        rospy.loginfo('Cancelling current goal...')
        self.killflag = True
        # self._result.pred = pred
        # self.server.set_aborted(self._result)

    def execute(self, goal):
        self.killflag = False
        rospy.loginfo('Drill detection action called')

        distA = 1000
        distB = 1000

        count = 0
        pred = None

        while count < goal.count and not self.killflag:
            markerA, markerB = self.get_marker_pos()

            if sum(markerA) > -1000 or sum(markerB) > -1000:
                hand_pos = self.get_hand_pos()

                if sum(hand_pos.ravel()) != -np.inf:                    
                    dA = min(np.linalg.norm(hand_pos[0] - markerA),
                             np.linalg.norm(hand_pos[1] - markerA))
                    dB = min(np.linalg.norm(hand_pos[0] - markerB),
                             np.linalg.norm(hand_pos[1] - markerB))

                    if sum(markerA) > -1000:
                        if distA == 1000:
                            distA = dA
                        else:
                            distA = (1-self.ETA)*distA + self.ETA*dA

                    if sum(markerB) > -1000:
                        if distB == 1000:
                            distB = dB
                        else:
                            distB = (1-self.ETA)*distB + self.ETA*dB

                    count += 1                    

            if count > 2:
                d = distA - distB
                pred = None

                if d > self.DIST_THRESHOLD:
                    pred = 'B'
                elif d < -self.DIST_THRESHOLD:
                    pred = 'A'

                if count % 10 == 0
                    self.return_result(False, pred)

        if not self.killflag:
            self.return_result(True, pred)

            rospy.loginfo('Goal executed')

        else:
            self._result.pred = str(pred)
            self.server.set_aborted(self._result)

            rospy.loginfo('Goal cancelled')

        print('Count {}, Marker A: {}, Marker B: {}, Result: {}'.format(count, distA, distB, pred))

if __name__ == '__main__':
    rospy.init_node('DetectDrill')
    server = DetectDrill()
    rospy.spin()