#!/usr/bin/env python

import json
import roslib
import rospy
import actionlib

import numpy as np
from math import pi
from copy import deepcopy

from tf.transformations import quaternion_from_euler, euler_from_quaternion

from std_msgs.msg import Header
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, PointCloud2, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from gotopose_server import JacoPosControl, Jaco2fingerGripper
from rail_manipulation_msgs.msg import SegmentedObject, SegmentedObjectList
from comanip_config.msg import GoToPoseAction, GoToPoseGoal, GoToObjAction, GoToObjResult


class TableSegmentation():

    def __init__(self):
        rospy.loginfo('Waiting for segment service ...')
        rospy.wait_for_service('/rail_segmentation/segment')
        rospy.loginfo('Service active!')
        # try:
        #     self.clearsrv = rospy.ServiceProxy('/rail_segmentation/clear', Empty)
        #     self.clearsrv()
        #     self.callsrv = rospy.ServiceProxy('/rail_segmentation/segment', Empty)
        # except rospy.ServiceException as e:
        #     rospy.loginfo('Service call failed: %s'%e)

    def calculate_distance(self, c1, c2):
        sqd = (c1-c2)**2
        rbar = 0.5*(c1[0] + c2[0])
        dist = 255*((2+rbar)*sqd[0] + 4*sqd[2] + (2+255/256-rbar)*sqd[3])**0.5
        # dist = np.linalg.norm(c1, c2)
        return dist

    def get_centroid_orientation(self, rgb, object_list, max_dimension):
        rgb = np.array(rgb)
        
        N = len(object_list.objects)
        rospy.loginfo('Number of segmented_objects: {}'.format(N))
        d = np.inf

        # select_obj_ids = []
        # select_objs = []
        obj_distmetric = []
        obj_rgbmetric = []
        obj_max_dimension = []
        # debug_dist_history = []

        select_obj_id = 0
        if N > 1:

            for i, obj in enumerate(object_list.objects):
                rospy.loginfo('Obj {:3d}, Dimensions in wxhxd: {:6.3f} x {:6.3f} x {:6.3f}, RGB: {:6.3f}, {:6.3f}, {:6.3f}, Center: {:6.3f} {:6.3f} {:6.3f}'
                        .format(i, obj.width, obj.height, obj.depth, obj.rgb[0], obj.rgb[1], obj.rgb[2], obj.center.x, obj.center.y, obj.center.z))
                rgb_obj = np.array(obj.rgb)
                dist = self.calculate_distance(rgb_obj, rgb)
                
                #np.dot(rgb_obj, rgb)/(np.linalg.norm(rgb_obj)*np.linalg.norm(rgb))
                # rospy.loginfo('Color difference: ', dist, 'threshold', threshold)
                # debug_dist_history.append(dist)
                # select_obj_ids.append(i)
                # select_objs.append(obj)
                obj_rgbmetric.append(dist)
                obj_distmetric.append((obj.center.x**2 + obj.center.y**2)**0.5)
                obj_max_dimension.append(max(obj.width, obj.height, obj.depth))
                print('Max dimension: ', obj_max_dimension[-1])

            obj_ids = [j if obj_max_dimension[j]<max_dimension for i in range(N)]
            obj_rgbmetric = np.array(obj_rgbmetric)[obj_ids]
            top1 = obj_rgbmetric.argsort()[0]
            select_obj_id = obj_ids[top1]

            # top2 = obj_rgbmetric.argsort()[:2].astype(int)
            # obj_distmetric = np.array(obj_distmetric)[top2]
            # top1 = np.argmin(obj_distmetric).astype(int) #closest y coordinate to the robot
            # select_obj_id = int(top2[top1])

            # print(top2)
            # print(obj_distmetric[top2])        
            # select_i = np.argmin(obj_distmetric[top2]) #closest y coordinate to the robot
            # select_i = int(top2[select_i])
            # select_obj_ids = [select_obj_ids[select_i]]
        # elif len(obj_rgbmetric) == 0:
        #     return None

        self.select_object = deepcopy(object_list.objects[select_obj_id])

        # select_obj_id = select_obj_ids[0]
        # self.select_object = deepcopy(select_objs[0])
        rospy.loginfo('Final obj- Dimensions in wxhxd: {:6.3f} x {:6.3f} x {:6.3f}'
                .format(self.select_object.width, self.select_object.height, self.select_object.depth))

        # rospy.loginfo('Centroid wrt table {:6.3f} {:6.3f} {:6.3f}'.format(self.select_object.centroid.x, self.select_object.centroid.y, self.select_object.centroid.z + 0.047178))
        rospy.loginfo('Center wrt table {:6.3f} {:6.3f} {:6.3f}'.format(self.select_object.center.x, self.select_object.center.y, self.select_object.center.z + 0.047178))
        euler_rotation = euler_from_quaternion([self.select_object.orientation.x, self.select_object.orientation.y, self.select_object.orientation.z, self.select_object.orientation.w])
        quat_rotation = quaternion_from_euler(0, 0, euler_rotation[-1])

        return np.array([self.select_object.center.x, self.select_object.center.y, self.select_object.center.z]), np.array(quat_rotation), select_obj_id

    def get_grasp_pose(self, pos, quat):
#        pos[0] -= 0.017
#        pos[1] += 0.017
        pos[-1] = 0.1#+= self.select_object.depth
        return pos, quat


class GoToObjServer:

    THRESHOLD = 0.05
    _result = GoToObjResult()

    def __init__(self):
        self.joint_traj_client = actionlib.SimpleActionClient("jaco_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        self.moveit_controller = JacoPosControl()
        self.gripper_controller = Jaco2fingerGripper()

        self.table_segmentation = TableSegmentation()
        self.server = actionlib.SimpleActionServer('gotoobj', GoToObjAction, self.execute, False)
        self.server.start()
        rospy.loginfo('GoToObj service ready!')

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
        self.joint_traj_client.send_goal(follow_goal)
        self.joint_traj_client.wait_for_result()

    def goto_pose(self, pos, quat, is_close, speed=0.5):
        self.moveit_controller.goto_pose(pos, quat)
        if is_close is not None:
            gripper_flag = int(is_close)
            self.gripper_controller.close(gripper_flag, speed=speed)

    def return_result(self, done, object_id=-1):
        self._result.done = int(done)
        self._result.object_id = object_id
        self.server.set_succeeded(self._result)       

    def execute(self, goal):
        object_list = deepcopy(goal.object_list)
        obj_config = json.loads(goal.json_config)

        if len(object_list.objects) == 0:
            self.return_result(False)

        else:
            standby_state = np.array([-pi/4, pi + pi/6, 2*pi, 2*pi/6+pi/4, pi, pi-pi/6-pi/4, pi/4])
            #np.array([-1.44197603, 4.05524706, 5.84119964, 2.05820314, 3.51034539, 1.92444854, -0.28556714])

            msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
            current_position = np.array(msg.position[:7])

            max_dimension = np.inf
            if 'max_dimension' in obj_config:
                max_dimension = obj_config['max_dimension']

            result = self.table_segmentation.get_centroid_orientation(obj_config['color'], 
                                                                      object_list, max_dimension)

            if result is None:
                rospy.loginfo('----- FAILED TO FIND MATCHING OBJECT -----')
                self.return_result(False)

            else:
                target_pos, target_quat, select_obj_id = result
                target_pos[-1] = 0.1
                if 'center_offset' in obj_config:
                    target_pos[0] += obj_config['center_offset'][0]
                    target_pos[1] += obj_config['center_offset'][1]
                if 'orientation_xyz' in obj_config:
                    target_quat = quaternion_from_euler(*obj_config['orientation_xyz'])
                self.goto_pose(target_pos, target_quat, False)
                
                if goal.is_close:
                    grasp_pos = target_pos
                    grasp_pos[-1] = obj_config['center_height'] 
                    self.goto_pose(grasp_pos, target_quat, True)

                    grasp_pos[-1] = 0.15 
                    self.goto_pose(grasp_pos, target_quat, None)

                    # Check if grip is a success or not:
                    rospy.sleep(0.05)
                    gripper_pos = self.gripper_controller.get_state()
                    threshold = float(obj_config['min_thickness'])
                    if abs(gripper_pos) < threshold:
                        rospy.loginfo('----- FAILED GRIP- {:6.3f} -----'.format(gripper_pos))
                        self.return_result(False)

                    else:
                        # Kill gripper speed
                        self.gripper_controller.close(True, speed=0.0, T=0.01)

                        target_pos = [0.05, -0.6, 0.15] #[-0.05, -0.5, 0.15]
                        self.goto_pose(target_pos, target_quat, None)

                        target_pos[-1] = obj_config['center_height']+ 0.0025
                        self.goto_pose(target_pos, target_quat, False, speed=0.1)

                        target_pos[-1] += 0.1
                        self.goto_pose(target_pos, target_quat, None)

                        self.return_result(True, object_id=select_obj_id)
        msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
        current_position = np.array(msg.position[:7])
        print(current_position)

        if np.linalg.norm(current_position - standby_state) > 1e-3:
            self.goto_joint(standby_state, duration=2.0)

        # self.goto_joint(standby_state, duration=3.0)
        self.gripper_controller.close(False)

if __name__ == '__main__':
    rospy.init_node('GoToObj_server')
    server = GoToObjServer()
    server.gripper_controller.close(False)
    rospy.spin()
