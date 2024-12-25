import networkx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from htn_node import HTNNode
from agent_utils import Agent

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

def visualize_from_node(node, filename):
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

    plt.savefig('{}.png'.format(filename), dpi=96*15)
    # plt.show()
    return edges

