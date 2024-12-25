#!/usr/bin/env python

import os
import sys
import yaml
import numpy as np
from math import pi
from copy import deepcopy

sys.path.insert(0, '/home/comanip/adapcomanip_ws/src/comanip_htn/src')

from htn.htn_utils import *
from htn.htn_node import HTNNode
from htn.agent_utils import Agent

def chair_htn_example():
    root = HTNNode(name='build_chair', node_type=HTNNode.FULLY_ORDERED)

    # layer 2
    n1 = HTNNode(name='build_components', node_type=HTNNode.PARTIALLY_ORDERED, parent=root)
    n2 = HTNNode(name='attach_back_to_seat', node_type=HTNNode.PRIMITIVE, parent=root, action='attach_back_to_seat')
    n_qc = HTNNode(name='quality_control', node_type=HTNNode.DECISION, parent=root)
    root.set_children([n1, n2, n_qc])

    # layer 3
    n3 = HTNNode(name='build_seat', node_type=HTNNode.FULLY_ORDERED, parent=n1)
    n4 = HTNNode(name='build_back', node_type=HTNNode.PARTIALLY_ORDERED, parent=n1)
    n1.set_children([n3, n4])

    n_qc1 = HTNNode(name='qc_inspect_pass', node_type=HTNNode.PRIMITIVE, parent=n_qc, action='inspect_pass')
    n_qc2 = HTNNode(name='qc_inspect_fail_repair', node_type=HTNNode.PRIMITIVE, parent=n_qc, action='inspect_fail_repair')
    n_qc.set_children([n_qc1, n_qc2])

    # layer 4
    n5 = HTNNode(name='attach_legs', node_type=HTNNode.DECISION, parent=n3)
    n6 = HTNNode(name='flip_seat' , node_type=HTNNode.PRIMITIVE, parent=n3, action='flip_seat')
    n3.set_children([n5, n6])

    n7 = HTNNode(name='attach_l_back' , node_type=HTNNode.PRIMITIVE, parent=n4, action='attach_l_back')
    n8 = HTNNode(name='attach_r_back', node_type=HTNNode.PRIMITIVE, parent=n4, action='attach_r_back')
    n4.set_children([n7, n8])

    # layer 5
    n5_human = HTNNode(name='attach_legs_H', node_type=HTNNode.PRIMITIVE, parent=n5, action='attach_legs')
    n5_robot = HTNNode(name='attach_legs_R', node_type=HTNNode.PRIMITIVE, parent=n5, action='attach_legs')
    n5.set_children([n5_human, n5_robot])

    return root

class VisualizeYamlHTN:
    def __init__(self, yaml_path):
        self.htn = construct_htn_from_yaml(yaml_path)
        print(self.htn.text_output())
        visualize_from_node(self.htn)

if __name__ == "__main__":

    package_path = '/home/comanip/adapcomanip_ws/src/comanip_htn'
    yaml_path = os.path.join(package_path, 'config/htn_drill_assm_2drills.yaml')    

    VisualizeYamlHTN = VisualizeYamlHTN(yaml_path)

    # htn = chair_htn_example()
    # print(htn.text_output())
    # visualize_from_node(htn)
