#!/usr/bin/env python

import os
import yaml
import json
from copy import deepcopy

import numpy as np
import rospy, rospkg
import actionlib

from math import pi
from std_srvs.srv import Empty
from comanip_htn.msg import GoToObjAction, GoToObjGoal
from rail_manipulation_msgs.msg import SegmentedObject, SegmentedObjectList

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

def is_close_object(query, target):
    query_center = np.array([query.center.x, query.center.y, query.center.z])
    target_center = np.array([target.center.x, target.center.y, target.center.z])
    return np.linalg.norm(query_center - target_center) < 5e-2

if __name__ == '__main__':
    rospy.init_node('GoToObj_client', anonymous=True)
    client = actionlib.SimpleActionClient('gotoobj', GoToObjAction)
    client.wait_for_server()

    joint_traj_client = actionlib.SimpleActionClient("jaco_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    joint_traj_client.wait_for_server()
    print('Joint actionlib active')

    standby_state = np.array([-pi/4, pi + pi/6, 2*pi, 2*pi/6+pi/4, pi, pi-pi/6-pi/4, pi/4])
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('comanip_htn')
    
    with open(os.path.join(package_path, 'config/objects.yaml'), 'r') as cfile:
        fullconfig = yaml.load(cfile)

    # config = fullconfig['Battery']
    # config = fullconfig['ScrewBox']

    def goto_joint(joint_states, duration=5.0):
        trajectory = JointTrajectory()
        trajectory.joint_names = ["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3", "j2s7s300_joint_4", "j2s7s300_joint_5", "j2s7s300_joint_6", "j2s7s300_joint_7"]
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = joint_states
        trajectory.points[0].velocities = [0.0 for _ in joint_states]
        trajectory.points[0].accelerations = [0.0 for _ in joint_states]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory
        joint_traj_client.send_goal(follow_goal)
        joint_traj_client.wait_for_result()

    print('Waiting for segmentation')
    rospy.wait_for_service('/rail_segmentation/segment')
    segmentcall = rospy.ServiceProxy('/rail_segmentation/segment', Empty)

    print('Segmenting')
    segmentcall()

    object_list = deepcopy(rospy.wait_for_message('rail_segmentation/segmented_objects', SegmentedObjectList))

    namedobject_list = ['ShellLR', 'ShellUR', 'ShellLL', 'ShellUL']


    for itr in range(4):
        config = fullconfig[namedobject_list[itr]]
        goal = GoToObjGoal()
        goal.json_config = json.dumps(config)
        goal.object_list = object_list
        goal.is_close = True

        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(0))
        result = client.get_result()
        id = result.object_id

        # '''
        removed_obj = object_list.objects.pop(id)
        for i, obj in enumerate(object_list.objects):
            if is_close_object(removed_obj, obj):
                object_list.objects.pop(i)
        print('{} done'.format(itr))
        raw_input('wait')
        goto_joint(standby_state, duration=3.0)

