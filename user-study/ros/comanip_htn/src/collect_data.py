#!/usr/bin/env python

import os
import yaml
import json
import time
from copy import deepcopy

import actionlib
import rospy, rospkg

from math import pi
from comanip_config.msg import GoToObjAction, GoToObjGoal

if __name__ == '__main__':
    rospy.init_node('GoToObj_client', anonymous=True)
    client = actionlib.SimpleActionClient('gotoobj', GoToObjAction)
    client.wait_for_server()

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('comanip_config')
    
    with open(os.path.join(package_path, 'config/objects.yaml'), 'r') as cfile:
        master_config = yaml.load(cfile)

    user_ip = input('1 for screws; 2 for battery; 0 to stop')
    print('')
    while user_ip:
        if user_ip == 1:
            config = deepcopy(master_config['ScrewBox'])
        elif user_ip == 2:
            config = deepcopy(master_config['Battery'])
        else:
            raise NotImplementedError

        goal = GoToObjGoal()
        goal.json_config = json.dumps(config)
        goal.is_close = True

        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(0))
        user_ip = input('1 for screws; 2 for battery; 0 to stop')
        print('')
