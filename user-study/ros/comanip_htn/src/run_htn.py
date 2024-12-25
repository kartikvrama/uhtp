#!/usr/bin/env python

import os
import sys
import time
import yaml
import json
import pickle
import numpy as np
from math import pi
from copy import deepcopy

import actionlib
import rospy, rospkg
from std_srvs.srv import Empty

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from visualization_msgs.msg import MarkerArray
from comanip_htn.msg import GoToObjAction, GoToObjGoal
from rail_manipulation_msgs.msg import SegmentedObject, SegmentedObjectList

sys.path.insert(0, '/home/comanip/adapcomanip_ws/src/comanip_htn/src')

from htn.htn_utils import *
from htn.agent_utils import Agent
from action_classifier.classifier import ActionPred

def is_close_object(query, target):
    query_center = np.array([query.center.x, query.center.y, query.center.z])
    target_center = np.array([target.center.x, target.center.y, target.center.z])
    return np.linalg.norm(query_center - target_center) < 5e-2

class CallServRobot:
    standby_state = np.array([0., pi, 
                             6.283185307179586, pi/6+pi/4, 
                             pi, 3*pi/4-pi/4, pi/2])

    def __init__(self):
        self.joint_traj_client = actionlib.SimpleActionClient("jaco_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        self.joint_traj_client.wait_for_server()
        msg = rospy.wait_for_message('/j2s7s300_driver/out/joint_state', JointState)
        current_position = np.array(msg.position[:7])

        if np.linalg.norm(current_position - self.standby_state) > 1e-3:
            self.goto_joint(self.standby_state, duration=3.0)

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

        self.object_list = deepcopy(rospy.wait_for_message('rail_segmentation/segmented_objects', SegmentedObjectList))
        rospy.loginfo('Table segmented!')

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
        config = self.config[object]
        if len(self.object_list.objects) == 0:
            rospy.loginfo('No more valid objects, resegmenting...')
            self.segmentcall()
            self.object_list = deepcopy(rospy.wait_for_message('rail_segmentation/segmented_objects', SegmentedObjectList))
        goal = GoToObjGoal()
        goal.json_config = json.dumps(config)
        goal.object_list = deepcopy(self.object_list)
        goal.is_close = True
        self.client.send_goal(goal)
        if wait:
            self.client.wait_for_result(rospy.Duration.from_sec(0))
            rospy.sleep(0.1)
            return self.robot_success()

    def robot_success(self):
        result = self.client.get_result()
        # DEBUG
        # print('Result: ', result)
        if result is None:
            return None
        else:
            if result.object_id >= 0:
                remid = result.object_id

                #DEBUG
                '''
                print(remid)
                l1 = len(self.object_list.objects) 
                for obj in self.object_list.objects:
                    print(obj.rgb)
                print('-')
                '''
                # Removing duplicate segments from the same object
                removed_obj = self.object_list.objects.pop(remid)
                for i, obj in enumerate(self.object_list.objects):
                    if is_close_object(removed_obj, obj):
                        _ = self.object_list.objects.pop(i)
                #DEBUG
                '''
                l2 = len(self.object_list.objects) 
                print(l1, l2)
                for obj in self.object_list.objects:
                    print(obj.rgb)
                print('')
                '''
            return bool(result.done)


class ROSHTN:
    def __init__(self, yaml_path, model_path, visualize=False):
        self.htn = construct_htn_from_yaml(yaml_path)
        self.htn.calculate_costs()
        if visualize:
            print(self.htn.text_output())
            visualize_from_node(self.htn)
            return 0
        stream = open(yaml_path, 'r')
        data = yaml.safe_load(stream)
        self.robot = CallServRobot()
        self.human_model = ActionPred(model_path)
        self.fixed_sequence = data['fixed_sequence']

        self.t = 0

        with open('./classification_output.txt', 'w+') as tf:
            tf.write('----*----')

    def get_human_action(self):
        skeleton_msg = rospy.wait_for_message('/front/body_tracking_data', MarkerArray)
        skeleton_data = np.array([np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]) for marker in skeleton_msg.markers  if marker.pose.position.x < 1])
        if len(skeleton_data) == 32:
            return self.human_model.predict_action(skeleton_data)
        else:
            return 'hold'

    def get_valid_human_actions(self):
        update_active_paths(self.htn)

        self.htn_valid_actions, probs = get_actions_for_agent(self.htn, Agent.HUMAN)
        valid_actions = [action[:-2] for action in self.htn_valid_actions]

        return valid_actions, probs

    def find_legal_human_action(self, pred):
        for axn in self.htn_valid_actions:
            if pred == axn[:-2]:
                return axn
        raise ValueError

    def run(self, robot_behavior):
        """                     
        Test autonomous adaptive behavior with simulated human decisions

        robot_behavior:
          adaptive: cumulative accumulation function reasoning algorithm
          fixed: fixed action sequence
        """

        self.htn = construct_htn_from_yaml(yaml_path)
        self.htn.calculate_costs()

        action_timer = {}      

        human_action_count = 0

        executing_human_action = None
        robot_action = None

        # Get valid human actions
        valid_human_actions, _ = self.get_valid_human_actions()

        # Initialize costs
        self.htn.calculate_costs()

        textfile = open('./classification_output.txt', 'a+')

        # debugstart = time.time()

        if robot_behavior == 'adaptive':
            rospy.loginfo('Running Adaptive behavior')
            # Run Adaptive task planning
            while len(self.htn.children) > 0:

                # #DEBUG
                # now = time.time()
                # print(now-debugstart)
                # debugstart=now


                # Detect human action
                pred_human_action = self.get_human_action()

                #DEBUG
                string = 'Step: {:4d}, Prediction: {:16s}, Currently executing: {:16s}, Valid human actions: {}\n'.format(self.t, pred_human_action, str(executing_human_action), self.htn_valid_actions)
                textfile.write(string)

                # Taking new human action
                if executing_human_action is None: # If it is the first human action, or an action has recently been completed
                    if pred_human_action != 'hold' and pred_human_action in valid_human_actions:

                        #DEBUG
                        print('t: {}'.format(self.t))

                        # Start timing
                        human_action_timer = time.time()

                        # First human action
                        executing_htn_action = self.find_legal_human_action(pred_human_action)
                        begin_action(self.htn, executing_htn_action, Agent.HUMAN)
                        executing_human_action = pred_human_action

                        # Reset action counter
                        human_action_count = 0
                        human_lookup_cost = lookup_cost(executing_htn_action, Agent.HUMAN)

                        # Calculate cost
                        self.htn.calculate_costs()    

                # Finishing prev human action
                elif pred_human_action != executing_human_action:
                    # If human has performed atleast 75% of action (lookup cost is in secs with 5 fps)
                    if human_action_count > 10:#0.75*human_lookup_cost*5: 

                        #DEBUG
                        print('t: {}, count is: {}, lookup_cost (num frames): {}'.format(self.t, human_action_count, human_lookup_cost*5))

                        # Finish previous action
                        finish_action(self.htn, executing_htn_action, Agent.HUMAN)

                        # Record time taken to complete action
                        end = time.time()
                        action_timer.update({'{}_{}'.format(self.t, executing_htn_action):(end-human_action_timer)})

                        # Update valid human actions
                        valid_human_actions, _ = self.get_valid_human_actions()

                        # Reset current human action
                        executing_human_action = None

                # Increment action couunter
                human_action_count += 1

                # Taking new robot action
                if robot_action is None:

                    # Update paths and check for best robot action based on cost
                    _, robot_actions = update_active_paths(self.htn)
                    robot_action, robot_params = find_best_robot_action(self.htn, robot_actions)
                    #DEBUG
                    # print('Valid robot actions: ', robot_actions)
                    # print('Best action: ', robot_action)

                    if robot_action != 'wait':

                        # Begin new action
                        self.robot.call_robot(robot_params['object'])
                        robot_action_timer = time.time()
                        begin_action(self.htn, robot_action, Agent.ROBOT)

                        # Calculate cost
                        self.htn.calculate_costs()

                # Finishing prev robot action
                if robot_action is not None:
                    if robot_action != 'wait':
                        
                        status = self.robot.robot_success()

                        if status is not None:
                            if status == True:

                                # Finish previous action
                                finish_action(self.htn, robot_action, Agent.ROBOT)

                                # Record time taken to complete action
                                end = time.time()
                                action_timer.update({'{}_{}'.format(self.t, robot_action):(end-robot_action_timer)})
                                robot_action_timer = end

                                # Calculate cost
                                self.htn.calculate_costs()

                                # Update valid human actions
                                valid_human_actions, _ = self.get_valid_human_actions()

                                # Reset current robot action to None
                                robot_action = None

                            else:
                                #TODO: Implement for gripper failure

                                raise NotImplementedError

                    else:
                        robot_action = None

                if len(self.htn.children) == 0:
                    break

                if rospy.is_shutdown():
                    break

                self.t += 1


        elif robot_behavior == 'fixed':
            rospy.loginfo('Running Fixed behavior')
            for item in self.fixed_sequence:

                start = time.time()
                rospy.loginfo('Executing: {}'.format(item[0]))

                self.robot.call_robot(item[1], wait=True)
                rospy.loginfo('Time taken in seconds: {:7.2f}'.format(time.time()-start))

                # if len(item) == 3:
                #     sleeptime = float(item[3])
                #     rospy.sleep(sleeptime)

        else:
            raise NotImplementedError

        textfile.close()
        return action_timer

if __name__ == "__main__":

    ROBOT_BHV = 'fixed'

    rospy.init_node('Adaptive_HTNtask_planner', anonymous=False)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('comanip_htn')
    yaml_path = os.path.join(package_path, 'config/htn_drill_assm_2drills.yaml')    
    model_path = os.path.join(package_path, 'src/action_classifier/ckpt')    

    ROSHTN = ROSHTN(yaml_path, model_path, visualize=False)
    print('Initializing... \n')
    rospy.sleep(3)
    print('Start!\n')

    input('wait')
    start = time.time()
    action_timer = ROSHTN.run(robot_behavior=ROBOT_BHV)
    input('Press to start next run')
    end1 = time.time()


    action_timer_2 = ROSHTN.run(robot_behavior=ROBOT_BHV)
    input('Enter when done:')
    end = time.time()

    print('RUN1: Time taken in seconds: {:7.2f}'.format(end1-start))

    print('RUN2: Time taken in seconds: {:7.2f}'.format(end-end1))

    print('Total time taken in seconds: {:7.2f}'.format(end-start))

    action_timer.update(action_timer_2)

    if ROBOT_BHV == 'adaptive':
        for key in action_timer.keys():
            print('{}:- {:7.2f}'.format(key, action_timer[key]))

        with open('adaptive_action_timer.p', 'wb') as pickf:
            pickle.dump(action_timer, pickf) #, protocol=pickle.HIGHEST_PROTOCOL)

    print('--- Ending ---')