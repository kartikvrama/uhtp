#!/usr/bin/env python

from htn_utils import finish_action, update_active_paths
from htn_utils import begin_action
from typing import NewType
from htn_utils import *
from agent_utils import Agent
import random
import math
import yaml

class ROSHTN:

    def __init__(self, yaml_path, visualize=False):
        self.htn = construct_htn_from_yaml(yaml_path)
        if visualize:
            visualize_from_node(self.htn)
        self.htn.calculate_costs()

        stream = open(yaml_path, 'r')
        data = yaml.safe_load(stream)

        self.fixed_sequence = data['config']['fixed_sequence']

    def test_find_next_action(self):
        from copy import deepcopy

        all_nodes = [self.htn.children[0].children[0].children[0],
                     self.htn.children[0].children[0].children[1].children[1],
                     self.htn.children[0].children[-1],
                     self.htn.children[1].children[0].children[0],
                     self.htn.children[1].children[-1]
                    ]

        for anode in all_nodes:
            newhtn = deepcopy(self.htn)
            node = deepcopy(anode)
            _, actions = update_active_paths(newhtn, Agent.HUMAN) 
            print(node.name)
            begin_action(newhtn, node.action, Agent.HUMAN)
            _, actions = update_active_paths(newhtn, Agent.HUMAN) 
            print(actions)            
            finish_action(newhtn, node.action, Agent.HUMAN)
            _, actions = update_active_paths(newhtn, Agent.HUMAN) 
            print(actions)
            print('')

    def run(self, robot_behavior=0):
        """
        Test autonomous adaptive behavior with simulated human decisions

        robot_behavior:
          0: cumulative accumulation function reasoning algorithm
          1: fixed action sequence
        """

        t = 0
        # robot_action_timer = 0
        # human_action_timer = 0
        robot_action = None
        human_action = 'wait'

        while len(self.htn.children) > 0:
            # resolve last robot action
            if robot_action is not None:
                if robot_action != 'wait':
                    if robot_suceeded():
                        finish_action(self.htn, robot_action, Agent.ROBOT)
                        self.htn.calculate_costs()
                        robot_action = None
                    else:
                        #TODO: Implement for gripper failure
                        raise NotImplementedError
                else:
                    robot_action = None

            if len(self.htn.children) == 0:
                break

            # take new robot action
            if robot_action is None:
                active, robot_actions = update_active_paths(self.htn)

                if robot_behavior == 0:
                    robot_action = find_best_robot_action(self.htn, robot_actions)

                elif robot_behavior == 1:
                    if len(self.fixed_sequence) > 0 and self.fixed_sequence[0] in robot_actions:
                        robot_action = self.fixed_sequence[0]
                        self.fixed_sequence.pop(0)
                    else:
                        robot_action = 'wait'
                else:
                    #TODO: In case additional robot behaviors exist
                    raise NotImplementedError


                if robot_action != 'wait':
                    print('Taking robot action: ' + robot_action)
                    begin_action(self.htn, robot_action, Agent.ROBOT)
                    robot_action_timer = lookup_cost(robot_action, Agent.ROBOT)
                    self.htn.calculate_costs()
                else:
                    robot_action_timer = 1

            # Detect human action
            current_human_action = get_human_action()

            if human_action != current_human_action:
                if human_action != 'wait':
                    finish_action(self.htn, human_action, Agent.HUMAN)
                if current_human_action != 'wait':
                    begin_action(self.htn, current_human_action, Agent.HUMAN)

                human_action = current_human_action

                # if human_action == 'wait':
                #     human_action_timer = 1
                # else:
                #     human_action_timer = lookup_cost(human_action, Agent.HUMAN)
                #     self.htn.calculate_costs()    

                self.htn.calculate_costs()    

        t += 1

    def run_sim(self, robot_behavior=0, human_behavior=0, task=1, param1=0.5):
        """
        Test autonomous adaptive behavior with simulated human decisions

        robot_behavior:
          0: cumulative accumulation function reasoning algorithm
          1: random action
          2: shortest (local) action
          3: fixed action sequence

        human_behavior:
          0: random
        """

        t = 0
        robot_action_timer = 0
        human_action_timer = 0
        robot_action = None
        human_action = None
        waiting_on_human = False
        while len(self.htn.children) > 0:
            # resolve last robot action
            if robot_action is not None and robot_action_timer == 0:
                if robot_action != 'wait':
                    finish_action(self.htn, robot_action, Agent.ROBOT)
                    self.htn.calculate_costs()
                    robot_action = None
                else:
                    if waiting_on_human:
                        robot_action_timer = 1
                    else:
                        robot_action = None

            # resolve last human action
            if human_action is not None:
                if human_action_timer == 0:
                    if human_action != 'wait':
                        finish_action(self.htn, human_action, Agent.HUMAN)
                        self.htn.calculate_costs()
                    human_action = None
                    if waiting_on_human:
                        waiting_on_human = False
                        robot_action = None
                        robot_action_timer = 0
                else:
                    update_remaining_time(self.htn, human_action, Agent.HUMAN, human_action_timer)
                    self.htn.calculate_costs()

            if len(self.htn.children) == 0:
                break

            # take new robot action
            if robot_action is None:
                # find and execute robot action according to robot policy
                # TODO: collapse HTN here first?
                active, robot_actions = update_active_paths(self.htn)
                # print('\n----------------------------\nCurrent HTN state:')
                # print(htn.text_output(include_costs=True, show_active_paths=True))
                # print('t=' + str(t) + ': Determining best robot action...')

                if robot_behavior == 0:
                    robot_action = find_best_robot_action(self.htn, robot_actions)
                elif robot_behavior == 1:
                    robot_action = random.choice(robot_actions)
                    if robot_action == 'wait':
                        waiting_on_human = True
                elif robot_behavior == 2:
                    unique_actions = list(set(robot_actions))
                    # print('---')
                    # print(unique_actions)

                    if task == 1 and 'wait' in unique_actions and len(unique_actions) == 2:
                        # for a in unique_actions:
                        #     if a != 'wait':
                        #         best_action = a
                        # if human_action is None and lookup_cost(best_action, Agent.HUMAN) >= 0 and lookup_cost(best_action, Agent.HUMAN) < lookup_cost(best_action, Agent.ROBOT):
                        #     robot_action = 'wait'
                        # else:
                        #     robot_action = best_action
                        robot_action = 'wait'
                    else:
                        lowest_cost = math.inf
                        best_actions = ['wait']
                        for a in unique_actions:
                            if a == 'wait':
                                continue
                            a_cost = lookup_cost(a)
                            if a_cost < lowest_cost:
                                best_actions = [a]
                                lowest_cost = a_cost
                            elif a_cost == lowest_cost:
                                best_actions.append(a)

                        robot_action = random.choice(best_actions)
                        if robot_action == 'wait':
                            waiting_on_human = True
                    # print(robot_action)
                elif robot_behavior == 3:
                    if len(self.fixed_sequence) > 0 and self.fixed_sequence[0] in robot_actions:
                        robot_action = self.fixed_sequence[0]
                        self.fixed_sequence.pop(0)
                    else:
                        robot_action = 'wait'

                if robot_action != 'wait':
                    begin_action(self.htn, robot_action, Agent.ROBOT)
                    robot_action_timer = lookup_cost(robot_action, Agent.ROBOT)
                    self.htn.calculate_costs()
                else:
                    robot_action_timer = 1

                # if robot_action != 'wait':
                #     print('Taking robot action: ' + robot_action)

            # take new human action
            if human_action is None:
                # TODO: collapse HTN here first?
                update_active_paths(self.htn)
                # print('\n----------------------------\nCurrent HTN state:')
                # print(htn.text_output(include_costs=True, show_active_paths=True))
                human_actions, probs = get_actions_for_agent(self.htn, Agent.HUMAN)
                # human_actions.append('wait')
                # human_actions = list(set(human_actions))

                if len(human_actions) > 0:
                    # normalize probabilities
                    p_total = sum(probs)
                    for i in range(len(probs)):
                        probs[i] /= p_total

                    # select action
                    r_prob = random.random()
                    counter = 0
                    human_action = human_actions[-1]
                    for i in range(len(probs)):
                        counter += probs[i]
                        if counter >= r_prob:
                            human_action = human_actions[i]
                            break
                    # human_action = random.choice(human_actions)
                else:
                    human_action = 'wait'

                # # prompt for human action
                # # print('t=' + str(t) + ': Available human actions: ' + str(human_actions))
                # human_input = str(raw_input('Enter your action from the list: '))
                # while human_input not in human_actions:
                #     human_input = str(raw_input('Enter your action from the list: '))
                # human_action = human_input

                if human_action != 'wait':
                    begin_action(self.htn, human_action, Agent.HUMAN)
                    human_action_timer = lookup_cost(human_action, Agent.HUMAN)
                    self.htn.calculate_costs()
                else:
                    human_action_timer = 1

                # if human_action != 'wait':
                #     print('Taking human action: ' + human_action)

            t += 1
            if robot_action_timer > 0:
                robot_action_timer -= 1
            if human_action_timer > 0:
                human_action_timer -= 1


if __name__ == "__main__":
    ROSHTN = ROSHTN('human_assm.yaml', False)
    ROSHTN.test_find_next_action()
    
    # print(return_valid_actions(Agent.HUMAN, ROSHTN.htn))
    # ROSHTN.run()
