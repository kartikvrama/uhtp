from htn.agent_utils import Agent, lookup_cost
from htn.htn_node import HTNNode
from copy import deepcopy
import networkx, pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import yaml

NODE_TYPES = {
    'FULLY_ORDERED': HTNNode.FULLY_ORDERED,
    'PARTIALLY_ORDERED': HTNNode.PARTIALLY_ORDERED,
    'PRIMITIVE': HTNNode.PRIMITIVE,
    'DECISION': HTNNode.DECISION
}

AGENT_TYPES = {
    'HUMAN': Agent.HUMAN,
    'ROBOT': Agent.ROBOT
}

def construct_htn_from_yaml(path):
    """ Returns the root HTN Node constructed from a YAML. """
    stream = open(path, 'r')
    data = yaml.safe_load(stream)
    root = None

    if 'htn' in data and 'name' in data['htn'] and 'type' in data['htn']:
        root = HTNNode(name=data['htn']['name'], node_type=NODE_TYPES[data['htn']['type']])
    else:
        print("Malformed YAML.")
        return None

    if 'children' in data['htn']:
        __construct_htn_from_yaml_recursive(root, data['htn']['children'])

    return __add_agent_assignments(root)


def __construct_htn_from_yaml_recursive(node, children):
    """ Recursive helper method for constructing HTN from YAML. """
    for child in children:
        child_node = HTNNode(name=child['name'], node_type=NODE_TYPES[child['type']], parent=node)

        if child_node.node_type == NODE_TYPES['PRIMITIVE']:

            if 'agent' in child:
                child_node.set_agent(AGENT_TYPES[child['agent']])

            if 'action' in child:
                child_node.set_action(child['action'])

                if 'parameters' in child['action']:
                    child_node.set_action_params(child['action']['parameters'])

        node.add_child(child_node)

        if 'children' in child:
            __construct_htn_from_yaml_recursive(child_node, child['children'])


def __recurse_children(edges, colors, node):
    """ Recursive method for visualizing nodes. """
    for child in node.children:
        if child.node_type == NODE_TYPES['FULLY_ORDERED']:
            colors.append('C3')
        elif child.node_type == NODE_TYPES['PARTIALLY_ORDERED']:
            colors.append('C0')
        elif child.node_type == NODE_TYPES['DECISION']:
            colors.append('C1')
        else:
            colors.append('C7')
        edges.append((node.name, child.name))
        __recurse_children(edges, colors, child)


def visualize_from_node(node):
    """ Outputs a visualization of the HTN from a node. """
    edges = []
    colors = []
    if node.node_type == NODE_TYPES['FULLY_ORDERED']:
        colors.append('C3')
    elif node.node_type == NODE_TYPES['PARTIALLY_ORDERED']:
        colors.append('C0')
    elif node.node_type == NODE_TYPES['DECISION']:
        colors.append('C1')
    else:
        colors.append('C7')
    __recurse_children(edges, colors, node)

    ColorLegend = {'FULLY_ORDERED': 'C3','PARTIALLY_ORDERED': 'C0','DECISION': 'C1','PRIMITIVE': 'C7'}
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for label in ColorLegend:
        ax.plot([0], [0], color=ColorLegend[label], label=label)

    G = networkx.DiGraph()
    G.add_edges_from(edges)
    pos = graphviz_layout(G, prog='dot')
    networkx.draw(G, pos, node_color=colors, with_labels=True, ax=ax, font_size=6)
    plt.axis('off')

    fig.tight_layout()
    plt.legend(loc='upper right', prop={'size': 6})

    plt.savefig('./htn_image.png', dpi=96*15)
    # plt.show()
    return edges


def __add_agent_assignments(base_htn):
    """ Converts unassigned primitive actions into decision nodes, deciding between which agent. """
    htn = deepcopy(base_htn)

    # traverse tree, at each primitive leaf that isn't assigned, generate assignment decision nodes
    frontier = [htn]
    while len(frontier) > 0:
        children = []
        for node in frontier:
            if len(node.children) > 0:
                children.extend(node.children)
            else:
                if node.agent == Agent.UNASSIGNED and node.node_type == HTNNode.PRIMITIVE :
                    __convert_to_decision(node)
        frontier = children

    return htn


def __convert_to_decision(node):
    """ Helper method for _add_agent_assignments. """
    if lookup_cost(node.action, Agent.ROBOT) >= 0 and lookup_cost(node.action, Agent.HUMAN) >= 0:
        n1 = HTNNode(name=node.name + '-r', node_type=HTNNode.PRIMITIVE, parent=node, action=node.action,
                     agent=Agent.ROBOT)
        n2 = HTNNode(name=node.name + '-h', node_type=HTNNode.PRIMITIVE, parent=node, action=node.action,
                     agent=Agent.HUMAN)
        node.set_children([n1, n2])
        node.probabilities = [.5, .5]
        node.node_type = HTNNode.DECISION
        node.action = None
    elif lookup_cost(node.action, Agent.ROBOT) >= 0:
        node.agent = Agent.ROBOT
    elif lookup_cost(node.action, Agent.HUMAN) >= 0:
        node.agent = Agent.HUMAN

def find_next_action(node, agent, return_result=[]):
    child_actions = []
    parent = node.parent

    # if parent.node_type == HTNNode.FULLY_ORDERED:
    flag = False
    for c in parent.children:
        if parent.node_type == HTNNode.FULLY_ORDERED:
            if flag:
                child_actions = [c]
                break
            if c == node:
                flag = True
        else:
            if c != node and c.state == HTNNode.INCOMPLETE:
                child_actions.append(c)
    # print('child actions', child_actions)

    if child_actions == []:
        if parent.parent is None:
            return []
        else:
            return_result += find_next_action(node.parent, agent, return_result)
    else:
        for next_node in child_actions:
            print(next_node, next_node.children)
            if next_node.node_type == HTNNode.PRIMITIVE:
                if next_node.agent == agent:
                    return_result += [next_node.name]#.action]
            elif next_node.node_type == HTNNode.FULLY_ORDERED:
                return_result += return_valid_actions(agent, next_node.children[0])#find_next_action(next_node.children[0], agent, return_result)
            else:
                for c in next_node.children:
                    return_result += return_valid_actions(agent, c)

    return return_result

    #         if c.state == HTNNode.INCOMPLETE or c.state == HTNNode.EXECUTING:
    #             flag = True
    #     if flag == False:
    #         return []
    # elif parent.node_type == HTNNode.DECISION or parent.node_type == HTNNode.PARTIALLY_ORDERED:
    #     actions = []
    #     for c in node.children:
    #         if c.state == HTNNode.INCOMPLETE or c.state == HTNNode.EXECUTING:
    #             continue
    #         else:
    #             actions.append()

def update_active_paths(node, agent=Agent.ROBOT):
    """ Update active state of nodes in HTN and return the current actions the robot can choose from. """
    if node.state == HTNNode.INCOMPLETE or node.state == HTNNode.EXECUTING:
        if node.node_type == HTNNode.PRIMITIVE:  # return primitive action
            node.active = True
            if node.agent == agent:#Agent.ROBOT:
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
                return_result = update_active_paths(next_node, agent)
                node.active = return_result[0]
                return node.active, return_result[1]

        else:  # partially ordered or decision, consider all children
            result_actions = []
            for c in node.children:
                if c.state == HTNNode.INCOMPLETE or c.state == HTNNode.EXECUTING:
                    return_result = update_active_paths(c, agent)
                    node.active = node.active or return_result[0]
                    result_actions.extend(return_result[1])
            return node.active, result_actions
    else:
        node.active = False
        return node.active, []

def return_valid_actions(agent, node, valid_actions=[]):
    nodetype = node.node_type

    # Primitive
    if nodetype == HTNNode.PRIMITIVE:
        if node.agent == agent:
            valid_actions.append(node)
            # if node.action in valid_actions.keys():
            #     valid_actions[node.action].append(node.name)
            # else:
            #     valid_actions.update({node.action: [node.name]})

            # if node.action not in valid_primitives:
            #     valid_primitives.append(node.action)

        # parent = node.parent
        # if parent.node_type == HTNNode.FULLY_ORDERED:
        #     parent.children = []
        # else:
        #     parent.remove_child(node)

        # valid_actions = return_valid_actions(agent, parent, valid_actions)

    else:
        children = node.children  

        # Fully ordered
        if nodetype == HTNNode.FULLY_ORDERED:
            if children:
                valid_actions = return_valid_actions(agent, children[0], valid_actions)
        
        # Partially ordered or decision
        elif nodetype == HTNNode.PARTIALLY_ORDERED or nodetype == HTNNode.DECISION:
            for child in children:
                valid_actions = return_valid_actions(agent, child, valid_actions)

    return valid_actions

def get_actions_for_agent(node, agent=Agent.HUMAN, p=1.0):
    '''Determine what actions a given agent can take at the current state of execution (update_active_paths first)'''
    if node.active:
        if node.node_type == HTNNode.PRIMITIVE:
            if node.agent == agent:
                # print(node.name)
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
                child_actions, child_probs = get_actions_for_agent(node.children[i], agent, p_mod * p)
                result_actions.extend(child_actions)
                result_probs.extend(child_probs)
            return result_actions, result_probs
    else:
        return [], []


def begin_action(node, action, agent=Agent.ROBOT, return_result=False):
    if node.active:
        if node.node_type == HTNNode.PRIMITIVE:
            # check for matching node to mark as executing
            if node.agent == agent and node.action == action and node.state == HTNNode.INCOMPLETE:
                node.state = HTNNode.EXECUTING
                return_result = True
                # print('done')
        elif node.node_type != HTNNode.DECISION:
            # traverse the rest of the tree
            for c in node.children:
                if return_result is True:
                    break
                return_result = begin_action(c, action, agent, return_result)
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
                    if return_result is True:
                        break
                    return_result = begin_action(n, action, agent, return_result)
    return return_result

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
                    # print('Finishing action for ', node.name)
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
