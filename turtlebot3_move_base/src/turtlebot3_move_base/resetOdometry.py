#!/usr/bin/env python2

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import rospy
from environment_stage_1 import Env

def main():
    rospy.init_node("reset_all", anonymous=False)

    rospy.set_param('/stage_number',1)

    env = Env(4)
    print ("Trying to reset sim")
    env.reset()


if __name__ == '__main__':
    main()