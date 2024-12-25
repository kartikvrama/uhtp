class Agent:
    UNASSIGNED = 0
    ROBOT = 1
    HUMAN = 2


def lookup_cost(action, agent=Agent.ROBOT):
    # print('action looking up- ', action)
    #TODO: Automate this somehow
    action = action[:-2]

    if agent == Agent.ROBOT:
        if action == 'pick_screws':
            return 28.54 #new: 
        elif action == 'pick_battery':
            return 25.37 #new: 
        elif action == 'pick_shell':
            return 26.35 #new: 
        else:
            print('Action {} not found'.format(action))
        #     raise NotImplementedError

    elif agent == Agent.HUMAN:
        if action == 'attach_shell':
            return 25 #10.5
        elif action == 'screw':
            return 35 #55
        elif action == 'attach_battery':
            return 8 #1.5
        elif action == 'place_drill':
            return 8 #3
        else:
            print('Action {} not found'.format(action))
        #     raise NotImplementedError

    return -1
