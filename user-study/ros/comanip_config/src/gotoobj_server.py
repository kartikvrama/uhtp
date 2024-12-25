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
        # sqd = (c1-c2)**2
        # rbar = 0.5*(c1[0] + c2[0])
        # dist = 255*((2+rbar)*sqd[0] + 4*sqd[1] + (2+255/256-rbar)*sqd[2])**0.5
        dist = np.sum((c1-c2)**2)
        return dist

    def get_centroid_orientation(self, rgb, object_list, max_dimension):
        rgb = np.array(rgb)
        
        N = len(object_list.objects)
        rospy.loginfo('Number of segmented_objects: {}'.format(N))
        d = np.inf

        # obj_distmetric = []
        obj_rgbmetric = []
        obj_max_dimension = []

        select_obj_id = 0
        if N > 1:

            for i, obj in enumerate(object_list.objects):
                rospy.loginfo('Obj {:3d}, Dimensions in wxhxd: {:6.3f} x {:6.3f} x {:6.3f}, RGB: {:6.3f}, {:6.3f}, {:6.3f}, Center: {:6.3f} {:6.3f} {:6.3f}'
                        .format(i, obj.width, obj.height, obj.depth, obj.rgb[0], obj.rgb[1], obj.rgb[2], obj.center.x, obj.center.y, obj.center.z))
                rgb_obj = np.array(obj.rgb)
                dist = self.calculate_distance(rgb_obj, rgb)
                print('Dist: {:7.2f}'.format(dist))                
                obj_rgbmetric.append(dist)
                # obj_distmetric.append((obj.center.x**2 + obj.center.y**2)**0.5)
                obj_max_dimension.append(max(obj.width, obj.height, obj.depth))
                print('Max dimension: ', obj_max_dimension[-1])

            obj_ids = []
            for j in range(N):
                if obj_max_dimension[j]<max_dimension:
                    obj_ids.append(j)
            obj_rgbmetric = np.array(obj_rgbmetric)[obj_ids]
            top1 = obj_rgbmetric.argsort()[0]
            select_obj_id = obj_ids[top1]

        else:
            obj = object_list.objects[0] 
            rospy.loginfo('Obj {:3d}, Dimensions in wxhxd: {:6.3f} x {:6.3f} x {:6.3f}, RGB: {:6.3f}, {:6.3f}, {:6.3f}, Center: {:6.3f} {:6.3f} {:6.3f}'
                    .format(0, obj.width, obj.height, obj.depth, obj.rgb[0], obj.rgb[1], obj.rgb[2], obj.center.x, obj.center.y, obj.center.z))


        self.select_object = deepcopy(object_list.objects[select_obj_id])

        rospy.loginfo('Final obj- Dimensions in wxhxd: {:6.3f} x {:6.3f} x {:6.3f}'
                .format(self.select_object.width, self.select_object.height, self.select_object.depth))

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

    # THRESHOLD = 0.05
    HEIGHT = 0.2
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
        self.moveit_controller.goto_pose(pos, quat, max_vel_scaling=self.obj_max_vel_scaling)
        if is_close is not None:
            gripper_flag = int(is_close)
            self.gripper_controller.close(gripper_flag, speed=speed)

    def return_result(self, done, object_id=-99):
        self._result.done = int(done)
        self._result.object_id = object_id
        rospy.loginfo('Sending Result: Done {}, ObjectID {}\n\n'.format(self._result.done, self._result.object_id))
        self.server.set_succeeded(self._result)       

    def execute(self, goal):
        rospy.loginfo('\n EXECUTING GOAL NOW \n')
        object_list = deepcopy(goal.object_list)
        obj_config = json.loads(goal.json_config)

        if len(object_list.objects) == 0:
            rospy.loginfo('----- FAILURE: NO SEGMENTED OBJECTS -----')
            self.return_result(False)

        else:
            is_fail = False
            is_gripper_fail = False

            standby_state = np.array([-pi/4, pi + pi/6, 2*pi, 2*pi/6+pi/4, pi, pi-pi/6-pi/4, pi/4])
            #np.array([-1.44197603, 4.05524706, 5.84119964, 2.05820314, 3.51034539, 1.92444854, -0.28556714])

            msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
            current_position = np.array(msg.position[:7])

            max_dimension = np.inf
            if 'max_dimension' in obj_config:
                max_dimension = obj_config['max_dimension']

            self.obj_max_vel_scaling = 0.6
            if 'max_vel_scaling' in obj_config:
                self.obj_max_vel_scaling = obj_config['max_vel_scaling']

            result = self.table_segmentation.get_centroid_orientation(obj_config['color'], 
                                                                      object_list, max_dimension)

            if result is None:
                rospy.loginfo('----- FAILED TO FIND MATCHING OBJECT -----')
                is_fail = True

            else:
                target_pos, target_quat, select_obj_id = result

                # If particular object needs to be picked from fixed position
                if 'fixed_position' in obj_config:
                    xy = obj_config['fixed_position']
                    target_pos[:2] = xy

                # If object detected using rgb add center offset
                elif 'center_offset' in obj_config:
                    target_pos[0] += obj_config['center_offset'][0]
                    target_pos[1] += obj_config['center_offset'][1]
                target_pos[-1] = 0.1

                if 'orientation_xyz' in obj_config:
                    target_quat = quaternion_from_euler(*obj_config['orientation_xyz'])

                self.goto_pose(target_pos, target_quat, False)

                # #DEBUG
                # print('\n\n')
                # print(target_pos)
                # print('cartesian')
                # print('{:7.3f} {:7.3f} {:7.3f} '.format(*target_pos))
                # print('joint space')
                # msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
                # current_position = np.array(msg.position[:7])
                # print('{:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} '.format(*current_position))                
                # print('\n\n')
                
                if goal.is_close:
                    grasp_pos = target_pos
                    grasp_pos[-1] = obj_config['center_height'] 
                    self.goto_pose(grasp_pos, target_quat, True)

                    grasp_pos[-1] = self.HEIGHT
                    self.goto_pose(grasp_pos, target_quat, None)

                    # Check if grip is a success or not:
                    rospy.sleep(0.05)
                    gripper_pos = self.gripper_controller.get_state()
                    threshold = float(obj_config['min_thickness'])

                    if abs(gripper_pos) < threshold:
                        is_gripper_fail = True
                        self.gripper_controller.close(False)
                        rospy.loginfo('----- FAILED GRIP: {:6.3f}, Threshold: {:6.3f}-----'.format(gripper_pos, threshold))

                    else:
                        rospy.loginfo('----- Grip succesful!: {:6.3f}, Threshold: {:6.3f}-----'.format(gripper_pos, threshold))
                        # Kill gripper speed
                        self.gripper_controller.close(True, speed=0.0, T=0.01)

                        target_pos = [-0.05, -0.6, self.HEIGHT] #[-0.05, -0.5, 0.15]
                        self.goto_pose(target_pos, target_quat, None)

                        target_pos[-1] = obj_config['center_height']+ 0.0025
                        self.goto_pose(target_pos, target_quat, False, speed=0.1)

                        target_pos[-1] += 0.1
                        self.goto_pose(target_pos, target_quat, None)

                        self.return_result(True, object_id=select_obj_id)

                else:
                    self.return_result(True, object_id=select_obj_id) 

            msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
            current_position = np.array(msg.position[:7])

            if is_gripper_fail:
                self.return_result(False, object_id=select_obj_id)

            else:
                if np.linalg.norm(current_position - standby_state) > 1e-3 and goal.is_close:
                    self.goto_joint(standby_state, duration=2.0)

                # self.goto_joint(standby_state, duration=3.0)
                self.gripper_controller.close(False)

            if is_fail is True:
                self.return_result(False)


if __name__ == '__main__':
    rospy.init_node('GoToObj_server')
    server = GoToObjServer()
    server.gripper_controller.close(False)
    rospy.spin()
