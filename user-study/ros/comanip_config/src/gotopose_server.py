#!/usr/bin/env python

import roslib
import rospy
import actionlib

import numpy as np
from math import pi
from copy import deepcopy

import moveit_commander
import moveit_msgs.msg

from tf import TransformListener
from tf.transformations import euler_from_quaternion, euler_matrix, quaternion_matrix, quaternion_from_matrix#, quaternion_from_euler

from geometry_msgs.msg import Pose

from jaco_utils import JacoControl
from comanip_config.msg import GoToPoseAction
from robotiq_85_msgs.msg import GripperCmd, GripperStat

class JacoPosControl(JacoControl):
    def __init__(self):
        JacoControl.__init__(self)
        self.rate = rospy.Rate(100)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander('arm')

        self.get_current_pose()
    
    def goto_pose(self, pos, quat, max_vel_scaling=0.75):
        R_0_t = quaternion_matrix(quat)
        R_t_g = euler_matrix(0, pi/2, 0)
        quat = quaternion_from_matrix(np.matmul(R_0_t, R_t_g))

        pose_goal = Pose()
        pose_goal.position.x = pos[0]
        pose_goal.position.y = pos[1]
        pose_goal.position.z = pos[2]

        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]

        self.group.allow_replanning(True)
        self.group.set_pose_target(pose_goal)
        self.group.set_max_velocity_scaling_factor(max_vel_scaling)
        self.group.go()
        self.group.stop()

class Jaco2fingerGripper():
    def __init__(self):
        self.publisher = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=1)

    def get_state(self):
        gripper_msg = rospy.wait_for_message('/gripper/stat', GripperStat)
        return gripper_msg.position

    def close(self, close, speed=0.5, T=2.0):
        target = 0.085*(1-int(close))
        current_gripper_pos = self.get_state()
        # print(target - current_gripper_pos, target, current_gripper_pos)
        if abs(target - current_gripper_pos) < 1e-3:
            rospy.loginfo('Gripper position remains the same')
        else:
            gripper_cmd = GripperCmd()
            gripper_cmd.speed = speed
            gripper_cmd.force = 100.0
            gripper_cmd.position = target
            start = rospy.get_time()
            while (rospy.get_time() - start)<T:
                self.publisher.publish(gripper_cmd)
            rospy.sleep(0.05)
            # gripper_cmd.speed = 0.0
            # gripper_cmd.force = 0.0
            # gripper_cmd.position = self.get_state()
            # start = rospy.get_time()
            # # while (rospy.get_time() - start) < 0.01:
            # self.publisher.publish(gripper_cmd)

class GoToPoseServer:
    def __init__(self):
        self.moveit_controller = JacoPosControl()
        self.gripper_controller = Jaco2fingerGripper()
        self.server = actionlib.SimpleActionServer('gotopose', GoToPoseAction, self.execute, False)
        self.server.start()
        rospy.loginfo('GoToPose service ready!')

    def execute(self, goal):
        pos = [goal.pose.position.x, goal.pose.position.y, goal.pose.position.z]
        quat = [goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w]
        rospy.loginfo('Going to pose...')
        self.moveit_controller.goto_pose(pos, quat)
 
        gripper_flag = int(goal.is_close)
        self.gripper_controller.close(gripper_flag)
 
        self.server.set_succeeded()       

if __name__ == '__main__':
    rospy.init_node('GoToPose_server')
    server = GoToPoseServer()
    rospy.spin()
