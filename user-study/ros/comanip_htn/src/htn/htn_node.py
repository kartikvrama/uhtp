from __future__ import division
from htn.agent_utils import Agent, lookup_cost


class HTNNode:
    FULLY_ORDERED = 0
    PARTIALLY_ORDERED = 1
    DECISION = 2
    PRIMITIVE = 3

    INCOMPLETE = 0
    EXECUTING = 1
    COMPLETE = 2

    def __init__(self, name='', node_type=PRIMITIVE, action=None, action_params=None, agent=Agent.UNASSIGNED, parent=None, state=INCOMPLETE):
        self.name = name  # name of the node
        self.node_type = node_type  # how execution of children is ordered
        self.action = action  # primitive action to execute (for primitives)
        self.state = state  # execution state
        self.action_params = action_params

        self.parent = parent  # parent of the HTN node, if any; used for traversing up network
        self.children = []  # list of child HTN nodes, if any; used for traversing down to reach primitive actions

        self.agent = agent  # agent executing the action (for primitives); -1 is unassigned
        self.agent_costs = [0, 0]  # accumulated agent costs [robot, human]  # TODO: support more than 2 agents
        self.total_cost = 0  # combined cost for the node, determined by characteristic accumulation function
        self.remaining_cost = 0  # used for tracking execution state once an action is started

        self.probabilities = []  # probabilities for executing each child node (for decisions)

        self.active = False  # whether a node is on the current active path

        self.take_action = None  # used for tracking what action created the current HTN during pruning

    def set_agent(self, agent):
        self.agent = agent

    def set_agent_costs(self, costs):
        self.agent_costs = costs

    def set_action(self, action):
        self.action = action

    def set_action_params(self, action_params):
        self.action_params = action_params

    def set_agent_cost(self, agent_id, cost):
        self.agent_costs[agent_id] = cost

    def set_children(self, nodes):
        self.children = nodes

    def add_child(self, node):
        self.children.append(node)

    def add_children(self, node_list):
        self.children.extend(node_list)

    def remove_child(self, node):
        if self.node_type == HTNNode.DECISION:
            self.probabilities.pop(self.children.index(node))
        self.children.remove(node)

    def replace_child(self, old_child, new_child):
        i = self.children.index(old_child)
        self.remove_child(old_child)
        self.children.insert(i, new_child)

    def calculate_costs_node(self):
        for i in range(len(self.agent_costs)):
            self.agent_costs[i] = 0

        if self.node_type == HTNNode.PRIMITIVE:
            if self.state == HTNNode.EXECUTING and self.remaining_cost > 0:
                if self.agent == Agent.ROBOT:
                    self.agent_costs[0] = self.remaining_cost
                elif self.agent == Agent.HUMAN:
                    self.agent_costs[1] = self.remaining_cost
            else:
                if self.agent == Agent.ROBOT:
                    self.agent_costs[0] = lookup_cost(self.action, self.agent)
                elif self.agent == Agent.HUMAN:
                    self.agent_costs[1] = lookup_cost(self.action, self.agent)
            self.total_cost = sum(self.agent_costs)

        elif self.node_type == HTNNode.FULLY_ORDERED:
            self.total_cost = 0
            for child in self.children:
                for i in range(len(self.agent_costs)):
                    self.agent_costs[i] += child.agent_costs[i]
                self.total_cost += child.total_cost
            # self.total_cost = sum(self.agent_costs)

        elif self.node_type == HTNNode.PARTIALLY_ORDERED:
            lower_bound = 0
            upper_bound = 0
            for child in self.children:
                for i in range(len(self.agent_costs)):
                    self.agent_costs[i] += child.agent_costs[i]
                lower_bound = max(lower_bound, child.total_cost)
                upper_bound += child.total_cost
            lower_bound = max(lower_bound, max(self.agent_costs))
            upper_bound = max(upper_bound, sum(self.agent_costs))

            # TODO: special cases for fully-divided costs
            if len(self.children) == 2 and (
                    (self.children[0].agent_costs[0] == 0 and self.children[1].agent_costs[1] == 0)
                    or (self.children[0].agent_costs[1] == 0 and self.children[1].agent_costs[0] == 0)):
                self.total_cost = max(self.agent_costs)
            else:
                self.total_cost = 0.5*lower_bound + 0.5*upper_bound

            # lower_bound = max(self.agent_costs)
            # upper_bound = sum(self.agent_costs)
            # self.total_cost = 0.5*lower_bound + 0.5*upper_bound

        elif self.node_type == HTNNode.DECISION:
            self.total_cost = 0

            for i in range(len(self.children)):
                for j in range(len(self.agent_costs)):
                    self.agent_costs[j] += self.probabilities[i]*self.children[i].agent_costs[j]
                self.total_cost += self.probabilities[i]*self.children[i].total_cost

    def calculate_costs(self):
        if self.node_type != HTNNode.PRIMITIVE:
            for c in self.children:
                c.calculate_costs()
            self.calculate_costs_node()
        else:
            self.calculate_costs_node()



    def contains_primitive(self, action, agent, active=False):
        if active:
            if not self.active:
                return False

        if self.node_type == HTNNode.PRIMITIVE:
            return self.action == action and self.agent == agent

        for c in self.children:
            if c.contains_primitive(action, agent, active):
                return True

        return False

    def contains_agent(self, agent, active=False):
        if self.node_type == HTNNode.PRIMITIVE:
            return self.agent == agent

        for c in self.children:
            if c.contains_agent(agent, active):
                return True

        return False

    def normalize_probabilities(self):
        total = sum(self.probabilities)
        for i in range(len(self.probabilities)):
            self.probabilities[i] /= total

    def text_output(self, level=0, parent_type=None, include_costs=False, show_active_paths=False):
        htn_str = ''
        for i in range(level):
            htn_str += '  '
        if parent_type is not None:
            if parent_type == HTNNode.FULLY_ORDERED:
                htn_str += '=> '
            elif parent_type == HTNNode.PARTIALLY_ORDERED:
                htn_str += '-> '
            elif parent_type == HTNNode.DECISION:
                htn_str += '<: '
        if show_active_paths and not self.active:
            htn_str += '(inactive) '
        htn_str += str(self)
        if include_costs:
            htn_str += ' {' + str(self.total_cost) + ' (' + str(self.agent_costs) + ')}'

        for c in self.children:
            htn_str += '\n' + c.text_output(level + 1, self.node_type, include_costs, show_active_paths)
        return htn_str

    @staticmethod
    def type_to_string(node_type):
        if node_type == HTNNode.PRIMITIVE:
            return 'primitive'
        elif node_type == HTNNode.PARTIALLY_ORDERED:
            return 'partially-ordered'
        elif node_type == HTNNode.FULLY_ORDERED:
            return 'fully-ordered'
        elif node_type == HTNNode.DECISION:
            return 'decision'

    def __str__(self):
        return self.name + ' (' + HTNNode.type_to_string(self.node_type) + ')'

    def __repr__(self):
        return str(self)
