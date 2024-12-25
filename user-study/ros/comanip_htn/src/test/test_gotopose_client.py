#!/usr/bin/env python

import rospy
import actionlib

from math import pi
from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from comanip_config.msg import GoToPoseAction, GoToPoseGoal

if __name__ == '__main__':
    rospy.init_node('GoToPose_client', anonymous=True)
    client = actionlib.SimpleActionClient('gotopose', GoToPoseAction)
    client.wait_for_server()
    print('ActionLib active!')

#    msg = rospy.wait_for_message('aruco_single/marker', Marker)
#    pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
#    quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

#    goal = GoToPoseGoal()
#    goal.pose.position.x = pos[0]-0.005
#    goal.pose.position.y = pos[1]
#    goal.pose.position.z = 0.1

#    quat = quaternion_from_euler(0, 0, 0)

#    goal.pose.orientation.x = quat[0]
#    goal.pose.orientation.y = quat[1]
#    goal.pose.orientation.z = quat[2]
#    goal.pose.orientation.w = quat[3]

#    goal.is_close = True


    goal = GoToPoseGoal()
    goal.pose.position.x = -0.05
    goal.pose.position.y = -0.5
    goal.pose.position.z = 0.05

    quat = quaternion_from_euler(0, 0, 0)

    goal.pose.orientation.x = quat[0]
    goal.pose.orientation.y = quat[1]
    goal.pose.orientation.z = quat[2]
    goal.pose.orientation.w = quat[3]

    goal.is_close = False

    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(0))
    print('finished')
