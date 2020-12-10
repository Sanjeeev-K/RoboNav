#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
import numpy as np
import time
import argparse
import sys
import os
from datetime import datetime


"""
data_logger.py
"""

class DataLogger():

    def __init__(self):
        # define objects to track
        self.stage = rospy.get_param("stage")
        self.method = rospy.get_param("method")

        if self.stage == '1' or self.stage == '2':
            self.object_id = ['turtlebot3_burger']
        if self.stage == '3':
            self.object_id = ['turtlebot3_burger',
                              'obstacle']
        if self.stage == '4':
            self.object_id = ['turtlebot3_burger',
                              'obstacle_1',
                              'obstacle_2']
        # variables
        self.x = [[],[],[]]
        self.y = [[],[],[]]
        self.theta = [[],[],[]]
        self.v = [[],[],[]]  
        self.omega = [[],[],[]] 
        self.min_dist = []

        self.current_min_scan = 10
        self.done = 'success'
        # define path
        self.directory = os.path.dirname(os.path.abspath(__file__))+'/logs/'+'stage'+self.stage+'/'

        rospy.set_param("end_flag", False)
        rospy.set_param("goal_reached_flag", False)

        print("Hello World")
        self.looper()

    def store_data(self):
        # get one instance of message
        data = None
        scan_data = None
        while data is None:
            try:
                
                data = rospy.wait_for_message('/gazebo/model_states', 
                                ModelStates, timeout=3)
                scan_data = rospy.wait_for_message('/scan', LaserScan, timeout=3)
                
                
            except:
                pass

        
        for i in range(len(self.object_id)):
            # Find the index of this object_id in the name list:
            idx = data.name.index(self.object_id[i])

            # Retrieve states from data
            self.x[i].append(data.pose[idx].position.x)
            self.y[i].append(data.pose[idx].position.y)
            self.v[i].append(data.twist[idx].linear.x)
            self.omega[i].append(data.twist[idx].angular.z)

        # obtain min obstacle distance
        scan_range = []
        for i in range(len(scan_data.ranges)):
            if scan_data.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan_data.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan_data.ranges[i])

        self.current_min_scan = round(min(scan_range), 2)
        
        self.min_dist.append(round(min(scan_range), 2))

    def clear_data(self):
        self.x = [[],[],[]]
        self.y = [[],[],[]]
        self.theta = [[],[],[]]
        self.v = [[],[],[]] 
        self.omega = [[],[],[]] 
        self.min_dist = [] 
    
    def save_data(self, trial, done='success'):
        # save state data
        for i in range(len(self.object_id)):
            if self.object_id[i] == 'turtlebot3_burger':
                data = np.array([done, self.x[i], self.y[i], self.v[i], self.omega[i], self.min_dist])
            else:
                data = np.array([self.x[i], self.y[i], self.v[i], self.omega[i]])

            # filename = '/logs/'+self.object_id[i]+'_stage_'+self.stage+'_'+self.method+'_'+str(trial)
            filename = self.object_id[i]+'_stage_'+self.stage+'_'+self.method+'_'+str(trial)

            if os.path.isdir(self.directory):
                np.save(self.directory+filename+".npy", data)
            else:
                os.makedirs(self.directory)
                np.save(self.directory+filename+".npy", data)
        self.clear_data()

    def looper(self):
        end_flag = rospy.get_param("end_flag")
        # self.done = 'success'

        goal_status = []

        collision_in_episode = []
        while  (not end_flag) and (not rospy.is_shutdown()):

            store_flag =  rospy.get_param("store_flag")
            save_flag = rospy.get_param("save_flag")
            goal_reached_flag = rospy.get_param("goal_reached_flag")
            
            
            if (store_flag) :       # done at every step

                print("Storing data")
                self.store_data()

                if (self.current_min_scan < 0.13):
                    collision_in_episode.append(True)
                    rospy.set_param("collision", True)
                else:
                    collision_in_episode.append(False)


                rospy.set_param("store_flag", False)


            if goal_reached_flag :

                if any(collision_in_episode):
                    goal_status.append(False)
                else:
                    goal_status.append(True)   

                collision_in_episode = []   
                rospy.set_param("goal_reached_flag", False) 
            
            self.trial = rospy.get_param("trial")
            if save_flag : 
                # self.done = 'success'

                print("Save DATA : Results : ")
                print(goal_status)
                
                print("Saving data, trial = ", self.trial)
                
                
                done_stat = all(goal_status)  # checks if all 5 entries in the goal_status list are true & returns true 
                print ("All True? : ", done_stat)

                done = ''

                if(done_stat):
                    done = 'success'
                else:
                    done = 'failure'


                self.save_data(self.trial,done_stat)
                rospy.set_param("save_flag", False)
                goal_status = []
                
                
            if (rospy.get_param("all_finished_flag")):
                rospy.set_param("end_flag", True)
                end_flag = True

        if(rospy.get_param("all_finished_flag") and rospy.get_param("end_flag")):
            done_stat = all(goal_status)
            done = ''

            if(done_stat):
                done = 'success'
            else:
                done = 'failure'
            self.save_data(self.trial,done_stat)
            rospy.set_param("save_flag", False)

        
        print("End of Trials")

if __name__ == '__main__':
    rospy.init_node('data_logger')
    dataLog = DataLogger()