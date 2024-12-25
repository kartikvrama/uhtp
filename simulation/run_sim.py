from __future__ import division
import argparse
import random
import pickle as pkl
import numpy as np

from htn_utils import *
from agent_utils import Agent, lookup_cost
from build_htn_sim import construct_chair_htn, single_drill_htn, two_part_assembly_htn, two_part_assembly_expanded_htn

from recurse_htn_utils import visualize_from_node

parser=argparse.ArgumentParser()
parser.add_argument("--task", help="Specify task as ikea or drill")
parser.add_argument("--verbose", help="Specify if HTNs should be visualized or not", default="False")
ARGS=parser.parse_args()

def draw_htns():
    """Visualize HTNs for chair construction and drill assembly tasks."""

    ikea_htn = construct_chair_htn()
    visualize_from_node(ikea_htn, filename='./plots/ikea_htn')

    drill_htn = two_part_assembly_htn(0.5)
    visualize_from_node(drill_htn, filename='./plots/drill_htn_two_part')

    drill_htn_one = single_drill_htn(0.5)
    visualize_from_node(drill_htn_one, filename='./plots/drill_htn_single')


def run(robot_behavior=0, human_behavior=0, task=1, param1=0.5):
    """Run the simulation for a given robot behavior, human behavior, and task.
    
    Args:
        robot_behavior (int): Robot behavior ID. Defaults to UHTP (0).
        human_behavior (int): Human behavior ID. Defaults to random (0).
        task (int): Task to be performed. Defaults to chair construction task.
        param1 (float): Probability of failure, only applicable to the drill
            assembly task. Defaults to 0.5.

    Description of the arguments:
        robot_behavior:
          0: cumulative accumulation function reasoning algorithm
          1: random action
          2: shortest (local) action
          3: fixed action sequence

        human_behavior:
          0: random

        task:
          0: construct chair
          1: two part assembly linear/parallel task
    
    Returns:
        int: Total time taken to complete the task.
    """
    # build unassigned HTN for chair construction example
    # print('Building initial HTN...')
    base_htn = None
    fixed_sequence = []
    if task == 0:
        base_htn = construct_chair_htn()
        fixed_sequence = ['attach_bl_leg', 'attach_br_leg', 'attach_l_back']
    elif task == 1:
        base_htn = two_part_assembly_expanded_htn(qc_fail_chance=param1)
        fixed_sequence = ['get_parts1', 'get_parts2', 'remove_assembly1', 'remove_assembly2']
    collapsed_htn = collapse_partial_sequences(base_htn)
    htn = add_agent_assignemnts(collapsed_htn)
    # print('HTN built.')

    # print('\n\nCalculating initial costs...')
    htn.calculate_costs()
    # print(htn.text_output(include_costs=True))

    t = 0
    robot_action_timer = 0
    human_action_timer = 0
    robot_action = None
    human_action = None
    waiting_on_human = False
    while len(htn.children) > 0:
        # resolve last robot action
        if robot_action is not None and robot_action_timer == 0:
            if robot_action != 'wait':
                finish_action(htn, robot_action, Agent.ROBOT)
                htn.calculate_costs()
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
                    finish_action(htn, human_action, Agent.HUMAN)
                    htn.calculate_costs()
                human_action = None
                if waiting_on_human:
                    waiting_on_human = False
                    robot_action = None
                    robot_action_timer = 0
            else:
                update_remaining_time(htn, human_action, Agent.HUMAN, human_action_timer)
                htn.calculate_costs()

        if len(htn.children) == 0:
            break

        # take new robot action
        if robot_action is None:
            # find and execute robot action according to robot policy
            # TODO: collapse HTN here first?
            active, robot_actions = update_active_paths(htn)
            # print('\n----------------------------\nCurrent HTN state:')
            # print(htn.text_output(include_costs=True, show_active_paths=True))
            # print('t=' + str(t) + ': Determining best robot action...')

            if robot_behavior == 0:
                robot_action = find_best_robot_action(htn, robot_actions)
            elif robot_behavior == 1:
                robot_action = random.choice(robot_actions)
                if robot_action == 'wait':
                    waiting_on_human = True
            elif robot_behavior == 2:
                unique_actions = list(set(robot_actions))
                # print('---')
                # print(unique_actions)

                if task==1 and 'wait' in unique_actions and len(unique_actions) == 2:
                    # for a in unique_actions:
                    #     if a != 'wait':
                    #         best_action = a
                    # if human_action is None and lookup_cost(best_action, Agent.HUMAN) >= 0 and lookup_cost(best_action, Agent.HUMAN) < lookup_cost(best_action, Agent.ROBOT):
                    #     robot_action = 'wait'
                    # else:
                    #     robot_action = best_action
                    robot_action = 'wait'
                else:
                    lowest_cost = 999999999
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
                if len(fixed_sequence) > 0 and fixed_sequence[0] in robot_actions:
                    robot_action = fixed_sequence[0]
                    fixed_sequence.pop(0)
                else:
                    robot_action = 'wait'

            if robot_action != 'wait':
                begin_action(htn, robot_action, Agent.ROBOT)
                robot_action_timer = lookup_cost(robot_action, Agent.ROBOT)
                htn.calculate_costs()
            else:
                robot_action_timer = 1

            # if robot_action != 'wait':
            #     print('Taking robot action: ' + robot_action)

        # take new human action
        if human_action is None:
            # TODO: collapse HTN here first?
            update_active_paths(htn)
            # print('\n----------------------------\nCurrent HTN state:')
            # print(htn.text_output(include_costs=True, show_active_paths=True))
            human_actions, probs = get_actions_for_agent(htn, Agent.HUMAN)
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
                begin_action(htn, human_action, Agent.HUMAN)
                human_action_timer = lookup_cost(human_action, Agent.HUMAN)
                htn.calculate_costs()
            else:
                human_action_timer = 1

            # if human_action != 'wait':
            #     print('Taking human action: ' + human_action)

        t += 1
        if robot_action_timer > 0:
            robot_action_timer -= 1
        if human_action_timer > 0:
            human_action_timer -= 1

    # print('\n\nFinished.')
    return t


def evaluate_ikea_chair():
    """Evaluate IKEA chair assembly task"""

    num_trials = 1000
    task = 0 # chair:0, drill:1
    param1 = 1.0

    print('Starting trials for IKEA chair assembly...')
    alg_results = {}

    for rbid, robot_behav in enumerate(['uhtp', 'random', 'greedy', 'fixed']):

        result = []

        for _ in range(num_trials):
            result.append(run(rbid, task=task, param1=param1))

        print(robot_behav)
        print('\tMin: ' + str(min(result)))
        print('\tMax: ' + str(max(result)))
        print('\tAverage: ' + str(sum(result)/len(result)))
        print('\tStd: ' + str(np.std(np.array(result))))
        print('')

        alg_results[rbid] = result

    with open('./results/ikea_chair_results.pickle', 'wb') as handle:
        pkl.dump(alg_results, handle, protocol=pkl.HIGHEST_PROTOCOL)


def evaluate_drill_assm():

    num_trials = 1000
    task = 1 # chair:0, drill:1

    print('Starting trials for power drill assembly...')

    pfail_range = list(np.arange(0, 1.01, 0.1))

    for rbid, robot_behav in enumerate(['uhtp', 'random', 'greedy', 'fixed']):

        locals()['result_{}'.format(robot_behav)] = {} # result_{robot_behav} = {}

        # Run assembly for each pfail
        for pfail in pfail_range:
            result = []
            for _ in range(num_trials):
                result.append(run(rbid, task=task, param1=pfail))

            # Print stats
            print('Algo {}, pfail={}'.format(robot_behav, pfail))
            print('\tMin: ' + str(min(result)))
            print('\tMax: ' + str(max(result)))
            print('\tAverage: ' + str(sum(result)/len(result)))
            print('\tStd: ' + str(np.std(np.array(result))))
            print('')

            # Save results to a dictionary of pfail:results
            locals()['result_{}'.format(robot_behav)][str(pfail)] = copy(result)

        alg_results = copy(locals()['result_{}'.format(robot_behav)])

        # Save results of individual algorithms to different pickle files
        with open('./results/drill_assm_algo_{}_results.pickle'.format(robot_behav), 'wb') as handle:
            pkl.dump(alg_results, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)

    if not ARGS.task in ["ikea", "drill"]:
        raise ValueError(f"Expected task to be either ikea or drill, instead received {ARGS.task}")

    if ARGS.verbose == "True":
        draw_htns()
    if ARGS.task == "ikea":
        evaluate_ikea_chair()
    elif ARGS.task == "drill":
        evaluate_drill_assm()
