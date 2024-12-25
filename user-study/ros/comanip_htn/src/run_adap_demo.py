#!/usr/bin/env python

import os
import sys
import time
import yaml
import json
import pickle
import signal
import argparse

import numpy as np

import rospy, rospkg
from rospy.core import rospyinfo
from std_msgs.msg import String

from visualization_msgs.msg import MarkerArray

sys.path.insert(0, '/home/comanip/adapcomanip_ws/src/adaptive-comanip/comanip_htn/src')

from htn.htn_utils import *
from htn.agent_utils import Agent
from action_classifier.classifier import ActionPred
from robot_caller import CallServRobot

COLOR_SCHEME = {'color_right': 'B', 'color_left': 'Y'}
MOVAVG_LEN = 5

class ROSHTN:
    def __init__(self, yaml_path, model_path):

        self.yaml_path = yaml_path

        self.robot = CallServRobot()
        self.human_model = ActionPred(model_path)

        self.t = 0
        self.initial_marker_dict = None

        # Check that color marker is detecting
        rospy.loginfo('Waiting for color marker...')
        rospy.wait_for_message('color_marker_detector', String)
        rospy.loginfo('Color marker detector active!')

    def load_htn_from_yaml(self):
        self.htn = construct_htn_from_yaml(self.yaml_path)
        self.htn.calculate_costs()

        stream = open(self.yaml_path, 'r')
        data = yaml.safe_load(stream)
        self.fixed_sequence = data['fixed_sequence']

        #DEBUG
        # print(self.htn.text_output(include_costs=True))
        # print(self.fixed_sequence)

    def handler(self, num, _):
        print('Got: {}'.format(num))
        self.early_exit = True
        return 1

    def get_human_action(self):
        # Collect skeleton data
        skeleton_msg = rospy.wait_for_message('/front/body_tracking_data', MarkerArray)
        skeleton_data = np.array([np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]) for marker in skeleton_msg.markers  if marker.pose.position.x < 1])

        # Human is visible
        if len(skeleton_data) == 32:

            # ----** OLD VERSION **----
            # return self.human_model.predict_action(skeleton_data)

            # ---- NEW VERSION ----
            pred = self.human_model.predict_action(skeleton_data)

            # Append current prediction to history
            self.prediction_history.append(pred)

            if len(self.prediction_history) > 0:

                # Remove old preds from history
                if len(self.prediction_history) > MOVAVG_LEN:
                    self.prediction_history.pop(0)

                # Return latest pred with max frequency
                pred = max(self.prediction_history[::-1], key=self.prediction_history.count)

                #DEBUG
                # print('MA: {}, orig: {}'.format(pred, self.prediction_history[-1]))

            return pred
 
        # Cannot see human, return None
        else:
            return None

    def get_color_marker(self):
        marker_msg = rospy.wait_for_message('color_marker_detector', String)
        dict = json.loads(marker_msg.data)
        
        # if True: #self.human_action_count < 2500:

        if dict['color_right'] == '0':
            self.attach_shell_flag = False
            return COLOR_SCHEME['color_right']

        elif dict['color_left'] == '0':
            self.attach_shell_flag = False
            return COLOR_SCHEME['color_left']

        else:
            return None

        # else:
        #     # If unable to detect color marker
        #     self.attach_shell_flag = False

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

    def find_multiple_legal_actions(self, pred):
        count = 0
        for axn in self.htn_valid_actions:
            if pred == axn[:-2]:
                count += 1
        
        return count > 1


    def run(self, robot_behavior, debugfile):
        """                     
        Test autonomous adaptive behavior with simulated human decisions

        robot_behavior:
          'adaptive': cumulative accumulation function reasoning algorithm
          'fixed': fixed action sequence
        """
        self.load_htn_from_yaml()
        self.htn.calculate_costs()

        #DEBUG
        print(self.htn.text_output())

        action_timer = {}
        robot_actions = []
        self.prediction_history = []

        # To log if gripper has failed      
        logmsgs = ''

        # Adaptive behavior
        if robot_behavior == 'adaptive':
            rospy.loginfo('Running Adaptive behavior')

            self.human_action_count = 0

            # Setting all current actions as None
            robot_action = None
            executing_human_action = None

            # Get valid human actions
            valid_human_actions, _ = self.get_valid_human_actions()

            # Initialize costs
            self.htn.calculate_costs()

            # Call pick_shell action by default and set attach_shell_flag as True
            self.robot.call_robot('Shell')
            self.attach_shell_flag = True

            robot_action_history = []

            # Run Adaptive task planning
            while len(self.htn.children) > 0:

                # Detect human action
                pred_human_action = self.get_human_action()

                string = 'Step: {:4d}, Prediction: {:16s}, Currently executing: {:16s}, Valid human actions: {}\n' \
                        .format(self.t, pred_human_action, str(executing_human_action), self.htn_valid_actions)
                debugfile.write(string)

                if pred_human_action is not None:
                    # Taking the first attach_shell action
                    if self.attach_shell_flag:

                        marker = self.get_color_marker()

                        if marker is None:
                            continue
    
                        else:
                            rospy.loginfo('Choosing marker: {}'.format(marker))

                            # Robot: Updating pick shell action in HTN 
                            robot_action = 'pick_shell_' + marker
                            robot_action_timer = time.time()
                            begin_action(self.htn, robot_action, Agent.ROBOT)

                            '''
                            # Start timing for human
                            human_action_timer = time.time()

                            # Human: Updating attach_shell action in HTN
                            executing_htn_action = 'attach_shell_' + marker
                            begin_action(self.htn, executing_htn_action, Agent.HUMAN)
                            executing_human_action = 'attach_shell'
                            '''

                            self.htn.calculate_costs()

                            # Update valid human actions
                            valid_human_actions, _ = self.get_valid_human_actions()

                    # Taking any other new human action
                    elif executing_human_action is None: # If an action has recently been completed
                        if pred_human_action != 'hold' and pred_human_action != 'grab_parts' \
                            and pred_human_action in valid_human_actions:

                            #DEBUG
                            print('Step t: {}'.format(self.t))

                            # Start timing
                            human_action_timer = time.time()

                            # First human action
                            executing_htn_action = self.find_legal_human_action(pred_human_action)
                            begin_action(self.htn, executing_htn_action, Agent.HUMAN)
                            executing_human_action = pred_human_action

                            # Calculate cost
                            self.htn.calculate_costs()    

                            # Reset action counter
                            self.human_action_count = 0
                            human_lookup_cost = lookup_cost(executing_htn_action, Agent.HUMAN)

                    # Finishing prev human action
                    elif pred_human_action != executing_human_action and pred_human_action != 'grab_parts':
                        # If human has performed atleast 2 secs of action (lookup cost is in secs with 5 fps)
                        if self.human_action_count > 10: #Tried using 0.75*human_lookup_cost*5, did not work 

                            #DEBUG
                            # print('t: {}, count is: {}, lookup_cost (num frames): {}'.format(self.t, self.human_action_count, human_lookup_cost*5))

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
                self.human_action_count += 1

                # Taking new robot action
                if robot_action is None and self.attach_shell_flag is False: # if attach shell has already been detected

                    # Update paths and check for best robot action based on cost
                    _, robot_actions = update_active_paths(self.htn)
                    robot_action, robot_params = find_best_robot_action(self.htn, robot_actions)

                    #DEBUG
                    print('Valid robot actions: {}'.format(robot_actions))
                    print('Choosing: {}'.format(robot_action))

                    if robot_action != 'wait':

                        #DEBUG
                        print('Step  t: {}'.format(self.t))

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
                                robot_action_history.append(robot_action)
            
                                # Record time taken to complete action
                                end = time.time()
                                action_timer.update({'{}_{}'.format(self.t, robot_action):(end-robot_action_timer)})
                                robot_action_timer = end

                                # Calculate cost
                                self.htn.calculate_costs()

                                # Update valid human actions
                                valid_human_actions, _ = self.get_valid_human_actions()

                                # Reset current robot action to None
                                prev_robot_action = robot_action
                                robot_action = None

                            else:
                                #TODO: Better ways to handle robot errors?
                                rospy.loginfo('Robot action unsuccesful, exiting')
                                logmsgs = 'segmentation'
                                break

                                # Restart action if not succesful
                                # rospy.loginfo('Robot action unsuccesful, trying again')
                                # self.robot.call_robot(robot_params['object'])

                        else:
                            # If gripper has failed, log the error
                            if len(self.robot.standby_obj_list) > 0:
                                logmsgs = 'gripper'

                    else:
                        robot_action = None

                # If no more robot actions left and all 6 actions of robot are executed
                if len(robot_action_history) == 6:
                    print('Robot action history')
                    print(robot_action_history)
                    rospy.loginfo('No more robot actions left')
                    break

                # If HTN is completed
                if len(self.htn.children) == 0:
                    break

                # If code is terminated
                if rospy.is_shutdown():
                    break

                self.t += 1

        # Fixed behavior
        elif robot_behavior == 'fixed':
            rospy.loginfo('Running Fixed behavior')
            for item in self.fixed_sequence:
                if not rospy.is_shutdown():
                    start = time.time()
                    rospy.loginfo('Executing: {}'.format(item[0]))

                    self.robot.call_robot(item[1], wait=True)
                    rospy.loginfo('Time taken in seconds: {:7.2f}'.format(time.time()-start))

                else:
                    break

        # For any other behavior not modeled
        else:
            raise NotImplementedError

        return action_timer, logmsgs

if __name__ == "__main__":

    # # User arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-user_id')
    # parser.add_argument('-mode')
    # args = parser.parse_args(sys.argv[1:5])

    # tag = 'User-{}_mode-{}'.format(args.user_id, args.mode)

    # HARDCODING USER ID FOR DEMO
    user_id = 'visitor'
    tag = 'User-{}_mode-{}'.format(user_id, 'adaptive')

    # Setting up ROS Node
    rospy.init_node('Adaptive_HTNtask_planner', anonymous=False)
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('comanip_htn')

    # Select Robot behavior
    ROBOT_BHV = 'adaptive'
    if ROBOT_BHV not in ['adaptive', 'fixed']:
        print('Input valid behavior: "adaptive" or "fixed"')
        raise NotImplementedError
    else:
        print('Running {} behavior'.format(ROBOT_BHV))

    # Create folder for user
    logdir_path = os.path.join(package_path, 'expt_logs/user_{}'.format(user_id))
    if not os.path.exists(logdir_path):
        os.mkdir(logdir_path)
        print('Created directory for User {}'.format(user_id))

    # Config files
    yaml_path = os.path.join(package_path, 'config/htn_drill_assm_2drills.yaml')    

    # yaml_path_run1 = os.path.join(package_path, 'config/htn_drill_assm_2drills_run1.yaml')    
    # yaml_path_run2 = os.path.join(package_path, 'config/htn_drill_assm_2drills_run2.yaml')    
    model_path = os.path.join(package_path, 'src/action_classifier/ckpt')    


    # Create object
    print('Initializing... \n')
    ROSHTN = ROSHTN(yaml_path, model_path)

    # Load HTN from appropriate yaml file
    ROSHTN.load_htn_from_yaml()   

    # Text file to debug classification output
    debugfile = open(os.path.join(logdir_path, '{}_debug.txt'.format(tag)), 'w')

    # Text file for execution output
    exfile = open(os.path.join(logdir_path, '{}_exop.txt'.format(tag)), 'w') 


    # Run 1 commands 
    raw_input('Start?')
    if not rospy.is_shutdown():
        print('Starting Now!')


        # ------------------------ RUN 1 ------------------------ 
        rospy.sleep(1)
        start1 = time.time()
        action_timer, logmsgs = ROSHTN.run(robot_behavior=ROBOT_BHV, debugfile=debugfile)

    if not rospy.is_shutdown():
        raw_input('Press enter to end current run')
        end1 = time.time()

        # Log robot failures
        if logmsgs != '':
            exfile.write('Run 1 failures: {}\n'.format(logmsgs))
            if logmsgs != 'gripper':
                raise RuntimeError

        # Run 1 summary
        exfile.write('RUN1: Time taken in seconds: {:7.2f}\n'.format(end1-start1))

        # Log current action timer values
        if ROBOT_BHV == 'adaptive':
            exfile.write('\nRun 1 action timer\n')
            for key in action_timer.keys():
                exfile.write('{}:- {:7.2f}\n'.format(key, action_timer[key]))
            exfile.write('\n')

        # Go to standby state
        ROSHTN.robot.goto_standby()

        print('--- Ending ---')
    
    else:
        print('Rospy Shutdown, killing process')
    