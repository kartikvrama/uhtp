from htn.agent_utils import Agent, lookup_cost
from htn.htn_node import HTNNode
from copy import deepcopy
import networkx
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

                if 'parameters' in child:
                    child_node.set_action_params(child['parameters'])

        node.add_child(child_node)

        if 'children' in child:
            __construct_htn_from_yaml_recursive(child_node, child['children'])

    if node.node_type == NODE_TYPES['DECISION']:

        num_children = len(children)

        node.probabilities = [1]*num_children
        node.normalize_probabilities()


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
    networkx.draw(G, pos, node_color=colors, with_labels=True, ax=ax, font_size=8)
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


def update_active_paths(node):
    """ Update active state of nodes in HTN and return the current actions the robot can choose from. """
    if node.state == HTNNode.INCOMPLETE or node.state == HTNNode.EXECUTING:
        if node.node_type == HTNNode.PRIMITIVE:  # return primitive action
            node.active = True
            if node.agent == Agent.ROBOT:
                return node.active, [[node.action, node.action_params]]
            else:
                return node.active, [['wait', None]]

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
                return_result = update_active_paths(next_node)
                node.active = return_result[0]
                return node.active, return_result[1]

        else:  # partially ordered or decision, consider all children
            result_actions = []
            for c in node.children:
                if c.state == HTNNode.INCOMPLETE or c.state == HTNNode.EXECUTING:
                    return_result = update_active_paths(c)
                    node.active = node.active or return_result[0]
                    result_actions.extend(return_result[1])
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
                child_actions, child_probs = get_actions_for_agent(node.children[i], agent, p_mod * p)
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
                #DEBUG
                print('\nStarting {} action for {} \n'.format(node.name, list(AGENT_TYPES.keys())[list(AGENT_TYPES.values()).index(agent)]))

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
                    #DEBUG
                    print('\nFinishing {} action for {} \n'.format(node.name, list(AGENT_TYPES.keys())[list(AGENT_TYPES.values()).index(agent)]))

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
                        # #DEBUG
                        # print(node.name)
                        # for child in node.children:
                        #     print(child.name)

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
    actions_names = [i[0] for i in actions]
    actions_params = {i[0]: i[1] for i in actions}
    pruned_htns = []
    robot_actions = list(set(actions_names))
    action_costs = []
    for a in robot_actions:
        pruned_htn = deepcopy(htn)
        pruned_htn.action_taken = a
        pruned_htn.set_action_params(actions_params[a])
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
        full_pruned_htns = pruned_htns
        pruned_htns = sorted(pruned_htns, key=lambda htn: htn.total_cost)

        idx = 0
        if len(pruned_htns) > 1:
            if pruned_htns[0].action_taken == 'wait' and pruned_htns[0].total_cost == pruned_htns[1].total_cost:
                idx = 1
                print('Choosing action {} instead of wait'.format(pruned_htns[idx].action_taken))

        #DEBUG
        # for i in range(len(full_pruned_htns)):
        #     print(full_pruned_htns[i].action_taken)
        #     print(full_pruned_htns[i].text_output(include_costs=True))
        #     print('----')

        # input('wait')

        return pruned_htns[idx].action_taken, pruned_htns[idx].action_params
    else:
        return 'wait', None
