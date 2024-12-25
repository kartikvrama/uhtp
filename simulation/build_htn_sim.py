from htn_node import HTNNode
from htn_utils import *
from agent_utils import lookup_cost


def construct_chair_htn():
    root = HTNNode(name='build_chair', node_type=HTNNode.FULLY_ORDERED)

    # layer 2
    n1 = HTNNode(name='build_components', node_type=HTNNode.PARTIALLY_ORDERED, parent=root)
    n2 = HTNNode(name='attach_back_to_seat', node_type=HTNNode.PRIMITIVE, parent=root, action='attach_back_to_seat')
    root.set_children([n1, n2])

    # layer 3
    n3 = HTNNode(name='build_seat', node_type=HTNNode.FULLY_ORDERED, parent=n1)
    n4 = HTNNode(name='build_back', node_type=HTNNode.PARTIALLY_ORDERED, parent=n1)
    n1.set_children([n3, n4])

    # layer 4
    n5 = HTNNode(name='attach_legs', node_type=HTNNode.PARTIALLY_ORDERED, parent=n3)
    n6 = HTNNode(name='flip_seat' , node_type=HTNNode.PRIMITIVE, parent=n3, action='flip_seat')
    n3.set_children([n5, n6])

    n7 = HTNNode(name='attach_l_back' , node_type=HTNNode.PRIMITIVE, parent=n4, action='attach_l_back')
    n8 = HTNNode(name='attach_r_back', node_type=HTNNode.PRIMITIVE, parent=n4, action='attach_r_back')
    n4.set_children([n7, n8])

    # layer 5
    n9 = HTNNode(name='left_legs', node_type=HTNNode.PARTIALLY_ORDERED, parent=n5)
    n10 = HTNNode(name='right_legs', node_type=HTNNode.PARTIALLY_ORDERED, parent=n5)
    n5.set_children([n9, n10])

    # layer 6
    n11 = HTNNode(name='attach_bl_leg' , node_type=HTNNode.PRIMITIVE, parent=n9, action='attach_bl_leg')
    n12 = HTNNode(name='attach_fl_leg', node_type=HTNNode.PRIMITIVE, parent=n9, action='attach_fl_leg')
    n9.set_children([n11, n12])

    n13 = HTNNode(name='attach_br_leg' , node_type=HTNNode.PRIMITIVE, parent=n10, action='attach_br_leg')
    n14 = HTNNode(name='attach_fr_leg', node_type=HTNNode.PRIMITIVE, parent=n10, action='attach_fr_leg')
    n10.set_children([n13, n14])

    return root


def chair_back_htn():
    root = HTNNode(name='build_chair_back', node_type=HTNNode.PARTIALLY_ORDERED)

    # layer 2
    n1 = HTNNode(name='attach_l_back' , node_type=HTNNode.PRIMITIVE, parent=root, action='attach_l_back')
    n2 = HTNNode(name='attach_r_back', node_type=HTNNode.PRIMITIVE, parent=root, action='attach_r_back')
    root.set_children([n1, n2])

    return root


def abstract_test_htn():
    root = HTNNode(name='abstract_test', node_type=HTNNode.PARTIALLY_ORDERED)

    # layer 2
    n1 = HTNNode(name='first_action', node_type=HTNNode.DECISION, parent=root)
    n2 = HTNNode(name='second_action', node_type=HTNNode.DECISION, parent=root)
    # n2 = HTNNode(name='a3-h', node_type=HTNNode.PRIMITIVE, parent=root, action='a3', agent=Agent.HUMAN)
    root.set_children([n1, n2])

    # layer 3
    n3 = HTNNode(name='a1-r', node_type=HTNNode.PRIMITIVE, parent=n1, action='a1', agent=Agent.ROBOT)
    n4 = HTNNode(name='a1-h', node_type=HTNNode.PRIMITIVE, parent=n1, action='a1', agent=Agent.HUMAN)
    n5 = HTNNode(name='a2-h', node_type=HTNNode.PRIMITIVE, parent=n1, action='a2', agent=Agent.HUMAN)
    n1.set_children([n3, n4, n5])
    n1.probabilities = [1, 1, 5]
    n1.normalize_probabilities()

    n6 = HTNNode(name='a3-r', node_type=HTNNode.PRIMITIVE, parent=n2, action='a3', agent=Agent.ROBOT)
    n7 = HTNNode(name='a3-h', node_type=HTNNode.PRIMITIVE, parent=n2, action='a3', agent=Agent.HUMAN)
    n2.set_children([n6, n7])
    n2.probabilities = [1, 1]
    n2.normalize_probabilities()

    return root


def single_drill_htn(qc_fail_chance = 0.25):
    root = HTNNode(name='single_drill_assembly', node_type=HTNNode.FULLY_ORDERED)
    n1 = HTNNode(name='get_parts1', node_type=HTNNode.PRIMITIVE, parent=root, action='get_parts1')
    n3 = HTNNode(name='linear_assisted', node_type=HTNNode.FULLY_ORDERED, parent=root)
    root.set_children([n1, n3])

    n6 = HTNNode(name='l_hold_part1', node_type=HTNNode.PRIMITIVE, parent=n3, action='hold_part1')
    n7 = HTNNode(name='l_assisted_assemble_parts1', node_type=HTNNode.PRIMITIVE, parent=n3, action='assisted_assemble_parts1')
    n8 = HTNNode(name='l_quality_control1', node_type=HTNNode.DECISION, parent=n3)
    n9 = HTNNode(name='l_remove_assembly1', node_type=HTNNode.PRIMITIVE, parent=n3, action='remove_assembly1')

    n3.set_children([n6, n7, n8, n9])

    n20 = HTNNode(name='l_qc_fail1', node_type=HTNNode.FULLY_ORDERED, parent=n8)
    n21 = HTNNode(name='l_inspect_pass1', node_type=HTNNode.PRIMITIVE, parent=n8, action='inspect_pass1')
    n8.set_children([n20, n21])
    n8.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n8.normalize_probabilities()

    n30 = HTNNode(name='l_inspect_fail1', node_type=HTNNode.PRIMITIVE, parent=n20, action='inspect_fail1')
    n31 = HTNNode(name='l_assisted_rewire1', node_type=HTNNode.PRIMITIVE, parent=n20, action='assisted_rewire1')
    n20.set_children([n30, n31])

    return root

def two_part_assembly_htn(qc_fail_chance = 0.25):
    root = HTNNode(name='two_part_assembly', node_type=HTNNode.FULLY_ORDERED)

    # layer 2
    n1 = HTNNode(name='get_parts1', node_type=HTNNode.PRIMITIVE, parent=root, action='get_parts1')
    n2 = HTNNode(name='decide_approach', node_type=HTNNode.DECISION, parent=root)
    root.set_children([n1, n2])

    # layer 3
    n3 = HTNNode(name='linear_assisted', node_type=HTNNode.FULLY_ORDERED, parent=n2)
    n4 = HTNNode(name='parallel_unassisted', node_type=HTNNode.FULLY_ORDERED, parent=n2)
    n5 = HTNNode(name='mixed_assisted', node_type=HTNNode.FULLY_ORDERED, parent=n2)
    n2.set_children([n3, n4, n5])
    n2.probabilities = [1, 1, 1]
    n2.normalize_probabilities()

    # layer 4
    n6 = HTNNode(name='l_hold_part1', node_type=HTNNode.PRIMITIVE, parent=n3, action='hold_part1')
    n7 = HTNNode(name='l_assisted_assemble_parts1', node_type=HTNNode.PRIMITIVE, parent=n3, action='assisted_assemble_parts1')
    n8 = HTNNode(name='l_quality_control1', node_type=HTNNode.DECISION, parent=n3)
    n9 = HTNNode(name='l_remove_assembly1', node_type=HTNNode.PRIMITIVE, parent=n3, action='remove_assembly1')
    n10 = HTNNode(name='l_get_parts2', node_type=HTNNode.PRIMITIVE, parent=n3, action='get_parts2')
    n11 = HTNNode(name='l_hold_part2', node_type=HTNNode.PRIMITIVE, parent=n3, action='hold_part2')
    n12 = HTNNode(name='l_assisted_assemble_parts2', node_type=HTNNode.PRIMITIVE, parent=n3, action='assisted_assemble_parts2')
    n13 = HTNNode(name='l_quality_control2', node_type=HTNNode.DECISION, parent=n3)
    n14 = HTNNode(name='l_remove_assembly2', node_type=HTNNode.PRIMITIVE, parent=n3, action='remove_assembly2')
    n3.set_children([n6, n7, n8, n9, n10, n11, n12, n13, n14])

    n15 = HTNNode(name='p_step1', node_type=HTNNode.PARTIALLY_ORDERED, parent=n4)
    n16 = HTNNode(name='p_step2', node_type=HTNNode.PARTIALLY_ORDERED, parent=n4)
    n17 = HTNNode(name='p_remove_assembly2', node_type=HTNNode.PRIMITIVE, parent=n4, action='remove_assembly2')
    n4.set_children([n15, n16, n17])

    n18 = HTNNode(name='m_step1', node_type=HTNNode.PARTIALLY_ORDERED, parent=n5)
    n19 = HTNNode(name='m_remove_assembly1', node_type=HTNNode.PRIMITIVE, parent=n5, action='remove_assembly1')
    n5.set_children([n18, n19])

    # layer 5
    n20 = HTNNode(name='l_qc_fail1', node_type=HTNNode.FULLY_ORDERED, parent=n8)
    n21 = HTNNode(name='l_inspect_pass1', node_type=HTNNode.PRIMITIVE, parent=n8, action='inspect_pass1')
    n8.set_children([n20, n21])
    n8.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n8.normalize_probabilities()

    n22 = HTNNode(name='l_qc_fail2', node_type=HTNNode.FULLY_ORDERED, parent=n13)
    n23 = HTNNode(name='l_inspect_pass2', node_type=HTNNode.PRIMITIVE, parent=n13, action='inspect_pass2')
    n13.set_children([n22, n23])
    n13.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n13.normalize_probabilities()

    n24 = HTNNode(name='p_get_parts2', node_type=HTNNode.PRIMITIVE, parent=n15, action='get_parts2')
    n25 = HTNNode(name='p_assemble1', node_type=HTNNode.FULLY_ORDERED, parent=n15)
    n15.set_children([n24, n25])

    n26 = HTNNode(name='p_remove_assembly1', node_type=HTNNode.PRIMITIVE, parent=n16, action='remove_assembly1')
    n27 = HTNNode(name='p_assemble2', node_type=HTNNode.FULLY_ORDERED, parent=n16)
    n16.set_children([n26, n27])

    n28 = HTNNode(name='m_assist_assembly2', node_type=HTNNode.FULLY_ORDERED, parent=n18)
    n29 = HTNNode(name='m_complete_assembly1', node_type=HTNNode.FULLY_ORDERED, parent=n18)
    n18.set_children([n28, n29])

    # layer 6
    n30 = HTNNode(name='l_inspect_fail1', node_type=HTNNode.PRIMITIVE, parent=n20, action='inspect_fail1')
    n31 = HTNNode(name='l_assisted_rewire1', node_type=HTNNode.PRIMITIVE, parent=n20, action='assisted_rewire1')
    n20.set_children([n30, n31])

    n32 = HTNNode(name='l_inspect_fail2', node_type=HTNNode.PRIMITIVE, parent=n22, action='inspect_fail2')
    n33 = HTNNode(name='l_assisted_rewire2', node_type=HTNNode.PRIMITIVE, parent=n22, action='assisted_rewire2')
    n22.set_children([n32, n33])

    n34 = HTNNode(name='p_assemble_parts1', node_type=HTNNode.PRIMITIVE, parent=n25, action='assemble_parts1')
    n35 = HTNNode(name='p_quality_control1', node_type=HTNNode.DECISION, parent=n25)
    n25.set_children([n34, n35])

    n36 = HTNNode(name='p_assemble_parts2', node_type=HTNNode.PRIMITIVE, parent=n27, action='assemble_parts2')
    n37 = HTNNode(name='p_quality_control2', node_type=HTNNode.DECISION, parent=n27)
    n27.set_children([n36, n37])

    n38 = HTNNode(name='m_get_parts2', node_type=HTNNode.PRIMITIVE, parent=n28, action='get_parts2')
    n39 = HTNNode(name='m_hold_part2', node_type=HTNNode.PRIMITIVE, parent=n28, action='hold_part2')
    n40 = HTNNode(name='m_assisted_assemble_parts2', node_type=HTNNode.PRIMITIVE, parent=n28, action='assisted_assemble_parts2')
    n41 = HTNNode(name='m_quality_control2', node_type=HTNNode.DECISION, parent=n28)
    n42 = HTNNode(name='m_remove_assembly2', node_type=HTNNode.PRIMITIVE, parent=n28, action='remove_assembly2')
    n28.set_children([n38, n39, n40, n41, n42])

    n43 = HTNNode(name='m_assemble_parts1', node_type=HTNNode.PRIMITIVE, parent=n29, action='assemble_parts1')
    n44 = HTNNode(name='m_quality_control1', node_type=HTNNode.DECISION, parent=n29)
    n29.set_children([n43, n44])

    # layer 7
    n45 = HTNNode(name='p_qc_fail1', node_type=HTNNode.FULLY_ORDERED, parent=n35)
    n46 = HTNNode(name='p_inspect_pass1', node_type=HTNNode.PRIMITIVE, parent=n35, action='inspect_pass1')
    n35.set_children([n45, n46])
    n35.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n35.normalize_probabilities()

    n47 = HTNNode(name='p_qc_fail2', node_type=HTNNode.FULLY_ORDERED, parent=n37)
    n48 = HTNNode(name='p_inspect_pass2', node_type=HTNNode.PRIMITIVE, parent=n37, action='inspect_pass2')
    n37.set_children([n47, n48])
    n37.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n37.normalize_probabilities()

    n49 = HTNNode(name='m_qc_fail2', node_type=HTNNode.FULLY_ORDERED, parent=n41)
    n50 = HTNNode(name='m_inspect_pass2', node_type=HTNNode.PRIMITIVE, parent=n41, action='inspect_pass2')
    n41.set_children([n49, n50])
    n41.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n41.normalize_probabilities()

    n51 = HTNNode(name='m_qc_fail1', node_type=HTNNode.FULLY_ORDERED, parent=n44)
    n52 = HTNNode(name='m_inspect_pass1', node_type=HTNNode.PRIMITIVE, parent=n44, action='inspect_pass1')
    n44.set_children([n51, n52])
    n44.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n44.normalize_probabilities()

    # layer 8
    n53 = HTNNode(name='p_inspect_fail1', node_type=HTNNode.PRIMITIVE, parent=n45, action='inspect_fail1')
    n54 = HTNNode(name='p_rewire1', node_type=HTNNode.PRIMITIVE, parent=n45, action='rewire1')
    n45.set_children([n53, n54])

    n55 = HTNNode(name='p_inspect_fail2', node_type=HTNNode.PRIMITIVE, parent=n47, action='inspect_fail2')
    n56 = HTNNode(name='p_rewire2', node_type=HTNNode.PRIMITIVE, parent=n47, action='rewire2')
    n47.set_children([n55, n56])

    n57 = HTNNode(name='m_inspect_fail2', node_type=HTNNode.PRIMITIVE, parent=n49, action='inspect_fail2')
    n58 = HTNNode(name='m_assisted_rewire2', node_type=HTNNode.PRIMITIVE, parent=n49, action='assisted_rewire2')
    n49.set_children([n57, n58])

    n59 = HTNNode(name='m_inspect_fail1', node_type=HTNNode.PRIMITIVE, parent=n51, action='inspect_fail1')
    n60 = HTNNode(name='m_rewire1', node_type=HTNNode.PRIMITIVE, parent=n51, action='rewire1')
    n51.set_children([n59, n60])

    return root


def two_part_assembly_expanded_htn(qc_fail_chance = 0.25):
    root = HTNNode(name='two_part_assembly', node_type=HTNNode.FULLY_ORDERED)

    # layer 2
    n1 = HTNNode(name='get_parts1', node_type=HTNNode.PRIMITIVE, parent=root, action='get_parts1')
    n2 = HTNNode(name='decide_approach', node_type=HTNNode.DECISION, parent=root)
    root.set_children([n1, n2])

    # layer 3
    n3 = HTNNode(name='linear_assisted', node_type=HTNNode.FULLY_ORDERED, parent=n2)
    n4 = HTNNode(name='parallel_unassisted', node_type=HTNNode.FULLY_ORDERED, parent=n2)
    n5 = HTNNode(name='mixed_assisted', node_type=HTNNode.FULLY_ORDERED, parent=n2)
    n2.set_children([n3, n4, n5])
    n2.probabilities = [1, 1, 1]
    n2.normalize_probabilities()

    # layer 4
    n6 = HTNNode(name='l_hold_part1', node_type=HTNNode.PRIMITIVE, parent=n3, action='hold_part1')
    n7 = HTNNode(name='l_assisted_assemble_parts1', node_type=HTNNode.PRIMITIVE, parent=n3, action='assisted_assemble_parts1')
    n8 = HTNNode(name='l_quality_control1', node_type=HTNNode.DECISION, parent=n3)
    n9 = HTNNode(name='l_remove_assembly1', node_type=HTNNode.PRIMITIVE, parent=n3, action='remove_assembly1')
    n10 = HTNNode(name='l_get_parts2', node_type=HTNNode.PRIMITIVE, parent=n3, action='get_parts2')
    n10_1 = HTNNode(name='l_assist_decision', node_type=HTNNode.DECISION, parent=n3)
    n14 = HTNNode(name='l_remove_assembly2', node_type=HTNNode.PRIMITIVE, parent=n3, action='remove_assembly2')
    n3.set_children([n6, n7, n8, n9, n10, n10_1, n14])

    # layer 4 linear assist decision subtree
    n10_2 = HTNNode(name='l_assist', node_type=HTNNode.FULLY_ORDERED, parent=n10_1)
    n10_3 = HTNNode(name='l_no_assist', node_type=HTNNode.FULLY_ORDERED, parent=n10_1)
    n10_1.set_children([n10_2, n10_3])
    n10_1.probabilities = [1, 1]
    n10_1.normalize_probabilities()

    # layer 4 linear assist subtree assist
    n11 = HTNNode(name='l_hold_part2', node_type=HTNNode.PRIMITIVE, parent=n10_2, action='hold_part2')
    n12 = HTNNode(name='l_assisted_assemble_parts2', node_type=HTNNode.PRIMITIVE, parent=n10_2, action='assisted_assemble_parts2')
    n13 = HTNNode(name='l_quality_control2', node_type=HTNNode.DECISION, parent=n10_2)
    n10_2.set_children([n11, n12, n13])

    # layer 4 linear no assist subtree assist
    n12b = HTNNode(name='l_no_assist_assemble_parts2', node_type=HTNNode.PRIMITIVE, parent=n10_3, action='assemble_parts2')
    n13b = HTNNode(name='l_no_assist_quality_control2', node_type=HTNNode.DECISION, parent=n10_3)
    n10_3.set_children([n12b, n13b])

    # layer 4 linear qc fail no assist subtree
    n22b = HTNNode(name='l_no_assist_qc_fail2', node_type=HTNNode.FULLY_ORDERED, parent=n13b)
    n23b = HTNNode(name='l_no_assist_inspect_pass2', node_type=HTNNode.PRIMITIVE, parent=n13b, action='inspect_pass2')
    n13b.set_children([n22b, n23b])
    n13b.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n13b.normalize_probabilities()

    # layer 4 linear qc fail no assist rewire subtree
    n32b = HTNNode(name='l_no_assist_inspect_fail2', node_type=HTNNode.PRIMITIVE, parent=n22b, action='inspect_fail2')
    n33b = HTNNode(name='l_no_assist_assisted_rewire2', node_type=HTNNode.PRIMITIVE, parent=n22b, action='rewire2')
    n22b.set_children([n32b, n33b])

    n15 = HTNNode(name='p_step1', node_type=HTNNode.PARTIALLY_ORDERED, parent=n4)
    n16 = HTNNode(name='p_step2', node_type=HTNNode.PARTIALLY_ORDERED, parent=n4)
    n17 = HTNNode(name='p_remove_assembly2', node_type=HTNNode.PRIMITIVE, parent=n4, action='remove_assembly2')
    n4.set_children([n15, n16, n17])

    n18 = HTNNode(name='m_step1', node_type=HTNNode.PARTIALLY_ORDERED, parent=n5)
    n19 = HTNNode(name='m_remove_assembly1', node_type=HTNNode.PRIMITIVE, parent=n5, action='remove_assembly1')
    n5.set_children([n18, n19])

    # layer 5
    n20 = HTNNode(name='l_qc_fail1', node_type=HTNNode.FULLY_ORDERED, parent=n8)
    n21 = HTNNode(name='l_inspect_pass1', node_type=HTNNode.PRIMITIVE, parent=n8, action='inspect_pass1')
    n8.set_children([n20, n21])
    n8.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n8.normalize_probabilities()

    n22 = HTNNode(name='l_qc_fail2', node_type=HTNNode.FULLY_ORDERED, parent=n13)
    n23 = HTNNode(name='l_inspect_pass2', node_type=HTNNode.PRIMITIVE, parent=n13, action='inspect_pass2')
    n13.set_children([n22, n23])
    n13.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n13.normalize_probabilities()

    n24 = HTNNode(name='p_get_parts2', node_type=HTNNode.PRIMITIVE, parent=n15, action='get_parts2')
    n25 = HTNNode(name='p_assemble1', node_type=HTNNode.FULLY_ORDERED, parent=n15)
    n15.set_children([n24, n25])

    n26 = HTNNode(name='p_remove_assembly1', node_type=HTNNode.PRIMITIVE, parent=n16, action='remove_assembly1')
    n27 = HTNNode(name='p_assemble2', node_type=HTNNode.FULLY_ORDERED, parent=n16)
    n16.set_children([n26, n27])

    n28 = HTNNode(name='m_assist_assembly2', node_type=HTNNode.FULLY_ORDERED, parent=n18)
    n29 = HTNNode(name='m_complete_assembly1', node_type=HTNNode.FULLY_ORDERED, parent=n18)
    n18.set_children([n28, n29])

    # layer 6
    n30 = HTNNode(name='l_inspect_fail1', node_type=HTNNode.PRIMITIVE, parent=n20, action='inspect_fail1')
    n31 = HTNNode(name='l_assisted_rewire1', node_type=HTNNode.PRIMITIVE, parent=n20, action='assisted_rewire1')
    n20.set_children([n30, n31])

    n32 = HTNNode(name='l_inspect_fail2', node_type=HTNNode.PRIMITIVE, parent=n22, action='inspect_fail2')
    n33 = HTNNode(name='l_assisted_rewire2', node_type=HTNNode.PRIMITIVE, parent=n22, action='assisted_rewire2')
    n22.set_children([n32, n33])

    n34 = HTNNode(name='p_assemble_parts1', node_type=HTNNode.PRIMITIVE, parent=n25, action='assemble_parts1')
    n35 = HTNNode(name='p_quality_control1', node_type=HTNNode.DECISION, parent=n25)
    n25.set_children([n34, n35])

    n36 = HTNNode(name='p_assemble_parts2', node_type=HTNNode.PRIMITIVE, parent=n27, action='assemble_parts2')
    n37 = HTNNode(name='p_quality_control2', node_type=HTNNode.DECISION, parent=n27)
    n27.set_children([n36, n37])

    n38 = HTNNode(name='m_get_parts2', node_type=HTNNode.PRIMITIVE, parent=n28, action='get_parts2')
    n39 = HTNNode(name='m_hold_part2', node_type=HTNNode.PRIMITIVE, parent=n28, action='hold_part2')
    n40 = HTNNode(name='m_assisted_assemble_parts2', node_type=HTNNode.PRIMITIVE, parent=n28, action='assisted_assemble_parts2')
    n41 = HTNNode(name='m_quality_control2', node_type=HTNNode.DECISION, parent=n28)
    n42 = HTNNode(name='m_remove_assembly2', node_type=HTNNode.PRIMITIVE, parent=n28, action='remove_assembly2')
    n28.set_children([n38, n39, n40, n41, n42])

    n43 = HTNNode(name='m_assemble_parts1', node_type=HTNNode.PRIMITIVE, parent=n29, action='assemble_parts1')
    n44 = HTNNode(name='m_quality_control1', node_type=HTNNode.DECISION, parent=n29)
    n29.set_children([n43, n44])

    # layer 7
    n45 = HTNNode(name='p_qc_fail1', node_type=HTNNode.FULLY_ORDERED, parent=n35)
    n46 = HTNNode(name='p_inspect_pass1', node_type=HTNNode.PRIMITIVE, parent=n35, action='inspect_pass1')
    n35.set_children([n45, n46])
    n35.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n35.normalize_probabilities()

    n47 = HTNNode(name='p_qc_fail2', node_type=HTNNode.FULLY_ORDERED, parent=n37)
    n48 = HTNNode(name='p_inspect_pass2', node_type=HTNNode.PRIMITIVE, parent=n37, action='inspect_pass2')
    n37.set_children([n47, n48])
    n37.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n37.normalize_probabilities()

    n49 = HTNNode(name='m_qc_fail2', node_type=HTNNode.FULLY_ORDERED, parent=n41)
    n50 = HTNNode(name='m_inspect_pass2', node_type=HTNNode.PRIMITIVE, parent=n41, action='inspect_pass2')
    n41.set_children([n49, n50])
    n41.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n41.normalize_probabilities()

    n51 = HTNNode(name='m_qc_fail1', node_type=HTNNode.FULLY_ORDERED, parent=n44)
    n52 = HTNNode(name='m_inspect_pass1', node_type=HTNNode.PRIMITIVE, parent=n44, action='inspect_pass1')
    n44.set_children([n51, n52])
    n44.probabilities = [qc_fail_chance, 1 - qc_fail_chance]
    n44.normalize_probabilities()

    # layer 8
    n53 = HTNNode(name='p_inspect_fail1', node_type=HTNNode.PRIMITIVE, parent=n45, action='inspect_fail1')
    n54 = HTNNode(name='p_rewire1', node_type=HTNNode.PRIMITIVE, parent=n45, action='rewire1')
    n45.set_children([n53, n54])

    n55 = HTNNode(name='p_inspect_fail2', node_type=HTNNode.PRIMITIVE, parent=n47, action='inspect_fail2')
    n56 = HTNNode(name='p_rewire2', node_type=HTNNode.PRIMITIVE, parent=n47, action='rewire2')
    n47.set_children([n55, n56])

    n57 = HTNNode(name='m_inspect_fail2', node_type=HTNNode.PRIMITIVE, parent=n49, action='inspect_fail2')
    n58 = HTNNode(name='m_assisted_rewire2', node_type=HTNNode.PRIMITIVE, parent=n49, action='assisted_rewire2')
    n49.set_children([n57, n58])

    n59 = HTNNode(name='m_inspect_fail1', node_type=HTNNode.PRIMITIVE, parent=n51, action='inspect_fail1')
    n60 = HTNNode(name='m_rewire1', node_type=HTNNode.PRIMITIVE, parent=n51, action='rewire1')
    n51.set_children([n59, n60])

    return root