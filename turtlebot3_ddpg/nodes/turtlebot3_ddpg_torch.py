#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.turtlebot3_ddpg.environment import Env
from agent_ddpg import *



if __name__ == '__main__':
	
	
	args = rospy.myargv(argv=sys.argv)
	stage = args[1]
	print(type(stage))	
	method = args[2]
	mode = args[3]
	
	if stage == "1":
		state_size = 26
		action_size = 1

	else:
		state_size = 28
		action_size = 1

	# current_time = '2020_12_07-13:39:36' #S1
	# current_time = '2020_12_08-07:54:57' #S2
	# current_time = '2020_12_05-18:43:24' #S3
	# current_time = '2020_12_06-03:21:16' #S4
	current_time = '2020_12_09-03:58:29' #S5
	# current_time = '2020_12_08-07:54:57'
	env = Env(action_size)
	if mode == "test":
		agent = ReinforceAgent(env, state_size, action_size, stage , "ddpg", "test", current_time)
		agent.test_model(10)
		
	elif mode == "cont":
		agent = ReinforceAgent(env, state_size, action_size, stage, "ddpg", "cont", current_time)
		agent.train_model()
	else:
		agent = ReinforceAgent(env, state_size, action_size, stage, "ddpg", "train")
		agent.train_model()
	
   
