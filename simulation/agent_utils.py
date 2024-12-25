class Agent:
    UNASSIGNED = 0
    ROBOT = 1
    HUMAN = 2

def lookup_cost(action, agent=Agent.ROBOT):
    if agent == Agent.ROBOT:
        if action == 'attach_bl_leg':
            return 10
        elif action == 'attach_br_leg':
            return 10
        elif action == 'attach_fl_leg':
            return 15
        elif action == 'attach_fr_leg':
            return 15
        elif action == 'flip_seat':
            return -1  # invalid, i.e. robot can not do this action
        elif action == 'attach_l_back':
            return 8
        elif action == 'attach_r_back':
            return 11
        elif action == 'attach_back_to_seat':
            return 35

        # abstract test task
        elif action == 'a1':
            return 10
        elif action == 'a2':
            return 10
        elif action == 'a3':
            return 10

        # industrial assembly task
        elif action == 'get_parts1' or action == 'get_parts2':
            return 30
        elif action == 'hold_part1' or action == 'hold_part2':
            return 1
        elif action == 'assemble_parts1' or action == 'assemble_parts2':
            return -1
        elif action == 'assisted_assemble_parts1' or action == 'assisted_assemble_parts2':
            return -1
        elif action == 'inspect_pass1' or action == 'inspect_pass2':
            return -1
        elif action == 'inspect_fail1' or action == 'inspect_fail2':
            return -1
        elif action == 'rewire1' or action == 'rewire2':
            return -1
        elif action == 'assisted_rewire1' or action == 'assisted_rewire2':
            return -1
        elif action == 'remove_assembly1' or action == 'remove_assembly2':
            return 20
    elif agent == Agent.HUMAN:
        if action == 'attach_bl_leg':
            return 15
        elif action == 'attach_br_leg':
            return 15
        elif action == 'attach_fl_leg':
            return 8
        elif action == 'attach_fr_leg':
            return 8
        elif action == 'flip_seat':
            return 5
        elif action == 'attach_l_back':
            return 10
        elif action == 'attach_r_back':
            return 5
        elif action == 'attach_back_to_seat':
            return 10

        elif action == 'a1':
            return 5
        elif action == 'a2':
            return 21
        elif action == 'a3':
            return 4

        # industrial assembly task
        elif action == 'get_parts1' or action == 'get_parts2':
            return -1
        elif action == 'hold_part1' or action == 'hold_part2':
            return -1
        elif action == 'assemble_parts1' or action == 'assemble_parts2':
            return 30
        elif action == 'assisted_assemble_parts1' or action == 'assisted_assemble_parts2':
            return 15
        elif action == 'inspect_pass1' or action == 'inspect_pass2':
            return 1
        elif action == 'inspect_fail1' or action == 'inspect_fail2':
            return 1
        elif action == 'rewire1' or action == 'rewire2':
            return 30
        elif action == 'assisted_rewire1' or action == 'assisted_rewire2':
            return 5
        elif action == 'remove_assembly1' or action == 'remove_assembly2':
            return -1

    return -1