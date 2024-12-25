from agent_utils import Agent
from htn_node import HTNNode
from copy import deepcopy
from copy import copy

def collapse_partial_sequences(base_htn):
    htn = deepcopy(base_htn)

    frontier = htn.children
    while len(frontier) > 0:
        children = []
        for node in frontier:
            if node.node_type == HTNNode.PARTIALLY_ORDERED and node.parent.node_type == HTNNode.PARTIALLY_ORDERED:
                node.parent.remove_child(node)
                node.parent.add_children(node.children)
                for c in node.children:
                    c.parent = node.parent
            children.extend(node.children)
        frontier = children

    return htn

def add_agent_assignemnts(base_htn):
    htn = deepcopy(base_htn)

    # traverse tree, at each primitive leaf that isn't assigned, generate assignment decision nodes
    frontier = [htn]
    while len(frontier) > 0:
        children = []
        for node in frontier:
            if len(node.children) > 0:
                children.extend(node.children)
            else:
                if node.agent == Agent.UNASSIGNED:
                    node.convert_to_decision()
        frontier = children

    return htn

def update_active_paths(node):
    '''Update active state of nodes in HTN and return the current actions the robot can choose from'''
    if node.state == HTNNode.INCOMPLETE or node.state == HTNNode.EXECUTING:
        if node.node_type == HTNNode.PRIMITIVE:  # return primitive action
            node.active = True
            if node.agent == Agent.ROBOT:
                return node.active, [node.action]
            else:
                return node.active, ['wait']

        elif node.node_type == HTNNode.FULLY_ORDERED:  # consider only next un-executed child in sequence
            next_node = None
            for c in node.children:
                if c.state == HTNNode.INCOMPLETE or c.state == HTNNode.EXECUTING:
                    next_node = c
                    break
            if next_node is None:
                node.active = False
                return False, []
            else:
                result = update_active_paths(next_node)
                node.active = result[0]
                return node.active, result[1]

        else:  # partially ordered or decision, consider all children
            result_actions = []
            for c in node.children:
                if c.state == HTNNode.INCOMPLETE or c.state == HTNNode.EXECUTING:
                    result = update_active_paths(c)
                    node.active = node.active or result[0]
                    result_actions.extend(result[1])
            return node.active, result_actions
    else:
        node.active = False
        return node.active, []

def get_actions_for_agent(node, agent=Agent.HUMAN, p=1.0):
    '''Determine what actions a given agent can take at the current state of execution (update_active_paths first)'''
    if node.active:
        if node.node_type == HTNNode.PRIMITIVE:
            if node.agent == agent:
                return [node.action], [p]
            else:
                return [], []
        else:
            result_actions = []
            result_probs = []
            for i in range(len(node.children)):
                if node.node_type == HTNNode.DECISION:
                    p_mod = node.probabilities[i]
                else:
                    p_mod = 1.0
                child_actions, child_probs = get_actions_for_agent(node.children[i], agent, p_mod*p)
                result_actions.extend(child_actions)
                result_probs.extend(child_probs)
            return result_actions, result_probs
    else:
        return [], []

def begin_action(node, action, agent=Agent.ROBOT):
    if node.active:
        if node.node_type == HTNNode.PRIMITIVE:
            # check for matching node to mark as executing
            if node.agent == agent and node.action == action and node.state == HTNNode.INCOMPLETE:
                node.state = HTNNode.EXECUTING
        elif node.node_type != HTNNode.DECISION:
            # traverse the rest of the tree
            for c in node.children:
                begin_action(c, action, agent)
        else:
            # determine whether each branch contains the action
            branches_with_action = []
            other_branches = []
            for c in node.children:
                if c.contains_primitive(action, agent, active=True):
                    branches_with_action.append(c)
                else:
                    other_branches.append(c)

            if len(branches_with_action) > 0:
                if len(other_branches) > 0:
                    # prune other branches
                    for b in other_branches:
                        node.remove_child(b)

                    if len(node.children) == 1:
                        # convert single branch decision node
                        node.parent.replace_child(node, node.children[0])
                        node.children[0].parent = node.parent
                    else:
                        # normalize decision node probabilities
                        node.normalize_probabilities()

                # prune remaining children
                for n in branches_with_action:
                    begin_action(n, action, agent)

def update_remaining_time(node, action, agent=Agent.HUMAN, remaining_time=0):
    if node.active:
        if node.node_type == HTNNode.PRIMITIVE:
            # check for matching node to remove
            if node.agent == agent and node.action == action and node.state == HTNNode.EXECUTING:
                node.remaining_cost = remaining_time
        else:
            # traverse children
            for c in node.children:
                update_remaining_time(c, action, agent, remaining_time)

def finish_action(node, action, agent=Agent.ROBOT):
    if node.active:
        if node.node_type == HTNNode.PRIMITIVE:
            # check for matching node to remove
            if node.agent == agent and node.action == action and node.state == HTNNode.EXECUTING:
                if node.parent != None:
                    p = node.parent
                    p.remove_child(node)
                    # print('p: ' + str(p) + ', node: ' + str(node))
                    while len(p.children) == 0 and p.parent != None:
                        gp = p.parent
                        # print('gp: ' + str(gp) + ', p: ' + str(p))
                        gp.remove_child(p)
                        p = gp
        else:
            # traverse children
            for c in node.children:
                finish_action(c, action, agent)

def prune_branches_by_action(node, action, agent=Agent.ROBOT):
    '''reduce active decision nodes that include the specified action to only branches that contain that action'''
    pruned_nodes = []
    if node.active:
        if node.node_type != HTNNode.DECISION:
            # traverse the rest of the tree
            for c in node.children:
                pruned_nodes.extend(prune_branches_by_action(c, action, agent))
            return pruned_nodes
        else:
            # determine whether each branch contains the action
            branches_with_action = []
            other_branches = []
            if action == 'wait':
                # special case where we need to prune branches containing our agent only
                other_agent = Agent.HUMAN
                if agent == Agent.HUMAN:
                    other_agent = Agent.ROBOT
                for c in node.children:
                    if c.contains_agent(other_agent, active=True):
                        branches_with_action.append(c)
                    else:
                        other_branches.append(c)
            else:
                for c in node.children:
                    if c.contains_primitive(action, agent, active=True):
                        branches_with_action.append(c)
                    else:
                        other_branches.append(c)

            if len(branches_with_action) > 0:
                if len(other_branches) > 0:
                    # prune other branches
                    for b in other_branches:
                        node.remove_child(b)

                    if len(node.children) == 1:
                        # convert single branch decision node
                        node.parent.replace_child(node, node.children[0])

                        # add this node's parent to pruned list, as it's child has changed
                        if node.parent not in pruned_nodes:
                            pruned_nodes.append(node.parent)
                    else:
                        # normalize decision node probabilities
                        node.normalize_probabilities()

                        # add node to pruned list
                        pruned_nodes.append(node)

                # prune remaining children
                for n in branches_with_action:
                    pruned_nodes.extend(prune_branches_by_action(n, action, agent))
                return pruned_nodes
            else:
                return []
    else:
        return []

def find_best_robot_action(htn, actions):
    pruned_htns = []
    robot_actions = list(set(actions))
    action_costs = []
    for a in robot_actions:
        pruned_htn = deepcopy(htn)
        pruned_htn.action_taken = a
        # if a == 'wait':
        #     print('\n\nHandle wait action by not changing the HTN...')
        #     pruned_htns.append(pruned_htn)
        #     continue

        pruned_nodes = prune_branches_by_action(pruned_htn, a)
        pruned_htn.calculate_costs()
        action_costs.append(pruned_htn.total_cost)
        pruned_htns.append(pruned_htn)

        # if a == 'wait':
        # print('---------------------------')
        # print('HTN for ' + pruned_htn.action_taken + ':')
        # print(pruned_htn.text_output(include_costs=True, show_active_paths=True))
        # print('---------------------------')

    if len(pruned_htns) > 0:
        # sort from low cost to high cost
        pruned_htns = sorted(pruned_htns, key=lambda htn: htn.total_cost)
        return pruned_htns[0].action_taken
    else:
        return 'wait'
