import os
import yaml
import json
from math import pi
from copy import deepcopy

import rospy
import rospkg
import numpy as np

import actionlib
from std_srvs.srv import Empty

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from comanip_htn.msg import GoToObjAction, GoToObjGoal
from rail_segmentation.srv import RemoveObject, RemoveObjectRequest
from rail_manipulation_msgs.msg import SegmentedObject, SegmentedObjectList


def calc_obj_dist(query, target):
    query_center = np.array([query.center.x, query.center.y])
    target_center = np.array([target.center.x, target.center.y])
    return np.linalg.norm(query_center - target_center)


def is_close_object(query, target):
    query_center = np.array([query.center.x, query.center.y, query.center.z])
    target_center = np.array([target.center.x, target.center.y, target.center.z])
    return np.linalg.norm(query_center - target_center) < 5e-2


class CallServRobot:
    standby_state = np.array([pi/5, pi, 
                              6.283185307179586, pi/6+pi/4, 
                              pi, 3*pi/4-pi/4, pi/2])
    _remove_obj_req = RemoveObjectRequest()

    def __init__(self):
        self.joint_traj_client = actionlib.SimpleActionClient("jaco_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        self.joint_traj_client.wait_for_server()
        # msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
        # current_position = np.array(msg.position[:7])

        self.goto_standby()

        # TODO: ROS node should be able to get this from ros params
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('comanip_htn')        
        with open(os.path.join(package_path, 'config/objects.yaml'), 'r') as cfile:
            self.config = yaml.load(cfile)

        self.client = actionlib.SimpleActionClient('gotoobj', GoToObjAction)
        self.client.wait_for_server()

        rospy.loginfo('Waiting for segment service ...')
        rospy.wait_for_service('/rail_segmentation/segment')
        rospy.loginfo('Service active!')
        self.segmentcall = rospy.ServiceProxy('/rail_segmentation/segment', Empty)
        self.segmentcall()

        self.remove_object_srv = rospy.ServiceProxy('rail_segmentation/remove_object', RemoveObject)

        self.object_list = deepcopy(rospy.wait_for_message('rail_segmentation/segmented_objects', SegmentedObjectList))
        self.standby_obj_list = []

        #TODO: Remove this somehow
        self.filter_unwanted_objects()

        if len(self.object_list.objects) < 12:
            rospy.loginfo('Segmentation failed to detect all objects, exiting')
            raise KeyError
        else:
            rospy.loginfo('{} objects segmented'.format(len(self.object_list.objects)))

        # TODO: Better solution for fixed position objects
        self.obj_prefixes = ['UL', 'LR', 'UR', 'LL']

        rospy.loginfo('Table segmented!')

    def goto_standby(self):
        msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
        current_position = np.array(msg.position[:7])
        if np.linalg.norm(current_position - self.standby_state) > 1e-3:
            self.goto_joint(self.standby_state, duration=3.0)

    def pop_object(self, i):
        obj = self.object_list.objects.pop(i)
        self._remove_obj_req.index = i
        self.remove_object_srv(self._remove_obj_req)
        return obj

    def filter_unwanted_objects(self):
        for i, object in enumerate(self.object_list.objects):
            if object.center.x < -0.7 or (object.center.x**2 + object.center.y**2)**0.5<0.1:

                # Remove any object that crosses workspace boundary or lies on robot
                # _ = self.object_list.objects.pop(i)
                self.pop_object(i)

    def goto_joint(self, joint_states, duration=5.0):
        trajectory = JointTrajectory()
        trajectory.joint_names = ["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3", "j2s7s300_joint_4", "j2s7s300_joint_5", "j2s7s300_joint_6", "j2s7s300_joint_7"]
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = joint_states
        trajectory.points[0].velocities = [0.0 for _ in joint_states]
        trajectory.points[0].accelerations = [0.0 for _ in joint_states]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory
        rospy.loginfo('Sending robot to home position')
        self.joint_traj_client.send_goal(follow_goal)
        self.joint_traj_client.wait_for_result()

    def call_robot(self, object, wait=False):
        if object == 'Shell':
            pfx = self.obj_prefixes.pop(0)
            object += pfx
        print('Collecting {}'.format(object))

        # Save current object
        self.current_object = object

        config = self.config[object]
        if len(self.object_list.objects) == 0:
            rospy.loginfo('No more segmented objects!')
            raise NotImplementedError
            # rospy.loginfo('No more valid objects, resegmenting...')
            # self.segmentcall()
            # self.object_list = deepcopy(rospy.wait_for_message('rail_segmentation/segmented_objects', SegmentedObjectList))

        goal = GoToObjGoal()
        goal.json_config = json.dumps(config)
        goal.object_list = deepcopy(self.object_list)
        goal.is_close = True

        self.client.send_goal(goal)
        self.running_goal = goal

        if wait:
            self.client.wait_for_result(rospy.Duration.from_sec(0))
            rospy.sleep(0.1)
            return self.robot_success()

    def robot_success(self):
        result = self.client.get_result()

        # DEBUG
        # print('Result: {}'.format(result))

        if result is None:
            return None

        elif result.done == 1:
            # Object id must be >= 0 if result is True
            assert result.object_id >= 0
            remid = result.object_id

            print('Obj RGB: {:6.3f} {:6.3f} {:6.3f}'.format(*self.object_list.objects[remid].rgb))

            #DEBUG
            # print(self.current_object)

            # Finding specific shell object from list
            if self.current_object[:5] == 'Shell':
                shell_object = SegmentedObject()
                shell_object.center.x = self.config[self.current_object]['fixed_position'][0]
                shell_object.center.y = self.config[self.current_object]['fixed_position'][1]
                shell_object.center.z = self.config[self.current_object]['center_height']
                closest_dist = 1e2
                closest_id = -1
                for i, obj in enumerate(self.object_list.objects):
                    d = calc_obj_dist(shell_object, obj)
                    if d < closest_dist:
                        closest_dist = d
                        closest_id = i
                remid = closest_id

            removed_obj = self.pop_object(remid) #self.object_list.objects.pop(remid)
            #TODO: Remove object from rail segmentation list as well

            # #DEBUG
            # if self.current_object[:5] == 'Shell':
            #     print(remid)
            #     print('removed')
            #     print(removed_obj.center)
            #     print('actual')
            #     print(shell_object.center)
            #     raw_input('wait')

            # Removing duplicate segments from the same object
            for i, obj in enumerate(self.object_list.objects):
                if is_close_object(removed_obj, obj):
                    #DEBUG
                    print('Eliminating {} from object list'.format(i))
                    self.pop_object(i)
                    # _ = self.object_list.objects.pop(i)

            # If this grasp was a second attempt, reappend objects from standby list
            if len(self.standby_obj_list) > 0:
                self.object_list.objects = self.object_list.objects + self.standby_obj_list
                # #DEBUG
                # for o in self.object_list.objects:
                #     print(type(o))
                self.standby_obj_list = []
            return bool(result.done)

        elif result.done == 0:
            # Result is False, need to detect if gripper failure
            if result.object_id >= 0:
                stbid = result.object_id
                
                # Save failed object as standby object
                standby_obj = self.object_list.objects.pop(stbid) #temporary popping
                self.standby_obj_list.append(standby_obj)

                # Update running goal with trimmed object list
                self.running_goal.object_list = deepcopy(self.object_list)
                self.client.send_goal(self.running_goal)

                # Continue with another goal
                return None

            # Different issue: return False
            else:
                return bool(result.done)
