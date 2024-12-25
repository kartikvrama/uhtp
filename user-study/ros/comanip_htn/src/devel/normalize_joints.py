#!/usr/bin/env python

import rospy
import actionlib
from math import pi

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal


def moveto(client, positions, duration=10.0):
    trajectory = JointTrajectory()
    trajectory.joint_names = ["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3", 
                                "j2s7s300_joint_4", "j2s7s300_joint_5", "j2s7s300_joint_6", "j2s7s300_joint_7"]
    trajectory.points.append(JointTrajectoryPoint())
    trajectory.points[0].positions = positions
    trajectory.points[0].velocities = [0.0 for _ in positions]
    trajectory.points[0].accelerations = [0.0 for _ in positions]
    trajectory.points[0].time_from_start = rospy.Duration(duration)
    follow_goal = FollowJointTrajectoryGoal()
    follow_goal.trajectory = trajectory
    client.send_goal(follow_goal)
    client.wait_for_result()
    rospy.loginfo("Done...")


def main():
    rospy.loginfo("Start client")
    rospy.init_node('normalize_joints')
    client = actionlib.SimpleActionClient("jaco_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    client.wait_for_server()
    rospy.loginfo("Client ready")

    positions = [0, 2.841815257065569, 6.283185307179586, 
                 pi/6, pi, 5*pi/6, 5.031392535766523]
    moveto(client, positions, 2)


if __name__ == '__main__':
    main()