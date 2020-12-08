#!/usr/bin/env python2
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import math
from math import pi
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_logger import DataLogger

from environment_stage_1 import Env

class Move():
    def __init__(self,last=False):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        
        print('Hello World!')

        self.env = Env(4)

        # goal type
        self.goal = MoveBaseGoal()
        self.goalReached = False

        #self.goal_list = np.array([[0.6, 0.0], [0.9, 1.2]])


        self.goal_list = np.array([[[0.6, 0.0], [0.9, 1.2], [-0.9, 0.0], [-0.1, -0.1], [0.5, 0.1]],
                                   [[0.9, 1.2], [-1.1, 0.9], [0.9, 1.2], [1.1, 0.3], [0.1, 1.0]],
                                   [[-0.6, -1.2], [1.2, 0.5], [1.2, -0.6], [-0.1, -1.2], [1.2, 1.2]],
                                   [[-0.5, -0.1], [-1.2, -1.1], [-0.1, -0.8], [0.8, 1.1], [-0.3, 1.2]],
                                   [[1.2, 1.0], [-0.6, 0.0], [1.2, -1.0], [-0.2, -1.2], [0.1, 0.7]],
                                   [[-1.1, -0.7], [-0.9, 1.1], [-0.8, 1.2], [-1.2, -0.3], [1.1, -0.1]],
                                   [[1.2, 0.7], [-1.2, -0.5], [1.2, -0.8], [-1.2, 0.8], [1.1, 0.7]],
                                   [[-1.1, 0.9], [0.0, -1.0], [-1.2, 0.1], [-0.5, -1.2], [0.6, 1.1]],
                                   [[-1.1, 0.5], [1.1, 1.0], [1.0, 0.0], [-0.9, 1.2], [0.0, -0.7]],
                                   [[-0.1, 1.1], [1.1, 0.3], [-0.8, 1.2], [-1.2, -0.3], [0.0, 0.9]]])




        self.goal_list_RowIndex = 0                   # 0-10


        self.goal_x_list = self.goal_list[self.goal_list_RowIndex , : , 0]
        self.goal_y_list = self.goal_list[self.goal_list_RowIndex , : , 1]
        
        
        
        self.finished = False

        self.noOfTests  = self.goal_list.shape[1]     # 5 tests in a trial
        self.noOfTrials = self.goal_list.shape[0]     # 10 trials for evaluation
        self.trial_completed = 0
        self.count = 0


        if self.noOfTrials == 1 :
            self.last = True
        else:
            self.last = False

        self.goalIndex = 0                           # 0-5

        self.goalX = 0
        self.goalY = 0

        self.currentX = 0
        self.currentY = 0

        self.stage = rospy.get_param("stage")
        self.method = str("move_base")
        rospy.set_param("method" , self.method)
        
        rospy.set_param("store_flag", False) 
        rospy.set_param("save_flag", False) 
        rospy.set_param("end_flag", False) 
        rospy.set_param("all_finished_flag", False)
        # self.logger = DataLogger(self.stage, self.logger)
        rospy.set_param("trial",self.trial_completed+1)



        print(self.noOfTests)
        print(self.noOfTrials)

        goal = self.setGoal()
        self.execute(goal)



    def distanceToGoal(self):
        return round(math.hypot(self.goalX - self.currentX , self.goalY - self.currentY),2)




    def setGoal(self):
        goal = MoveBaseGoal()
        self.goalX = self.goal_x_list[self.goalIndex]
        self.goalY = self.goal_y_list[self.goalIndex]

        print('Set New Goal to ', self.goalX, self.goalY)
        self.reachedGoal = False
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = self.goalX
        goal.target_pose.pose.position.y = self.goalY
        goal.target_pose.pose.orientation.w = 1.0

        self.goalIndex += 1
        
        if self.goalIndex <= self.noOfTests:
            self.finished = False

        return goal


    # send goal to move_base to execute goal
    def execute(self,goal):

        print("Executing goal")
        self.client.send_goal(goal,self.done, self.active_cb, self.feedback_cb)


        while not  (self.last and self.finished):
            rospy.spin()


    # runs this call-back function when move_base done executing goal
    def done(self,status, result):

        if status == 1 : 
            rospy.loginfo("Status 1")

        elif status == 2 : 
            rospy.loginfo("Status 2")

        elif status == 3 : 
            rospy.loginfo("Reached Goal")
            rospy.set_param("trial",self.trial_completed+1)
            if (self.goalIndex < self.noOfTests):
                
                self.run()

            
            else:
                rospy.loginfo(' ######### End of Trial ######### ')
                rospy.set_param("save_flag", True)
                self.trial_completed += 1
                self.goal_list_RowIndex += 1
                self.finished = True
                self.goalIndex = 0
                
                

                if self.last:
                    print('All Trials Completed')
                    rospy.set_param("all_finished_flag", True)
                    self.shutdown()
                else :  
                    self.goal_x_list = self.goal_list[self.goal_list_RowIndex , : , 0]
                    self.goal_y_list = self.goal_list[self.goal_list_RowIndex , : , 1]

                    if self.trial_completed == self.noOfTrials-1:
                        self.last = True

                    # reset env
                    self.env.reset()

                    self.run()
                    
                

        elif status == 4:
            rospy.loginfo("Goal pose "+str(self.count)+" was aborted by the Action Server")
            self.shutdown()

        elif status == 5 : 
            rospy.loginfo("Status 5")

        elif status == 6 : 
            rospy.loginfo("Status 6")

        elif status == 7 : 
            rospy.loginfo("Status 7")

        elif status == 8 : 
            rospy.loginfo("Status 8")
        else: 
            rospy.logerr("Something else happened ")

    def active_cb(self):
        # print("Active callback")
        rospy.set_param("goal_reached_flag", True)
    
    def feedback_cb(self, feedback):
        # print("Feedback callback")
        rospy.set_param("store_flag", True) 

    def run(self):
        goal = self.setGoal()
        self.client.send_goal(goal,self.done, self.active_cb, self.feedback_cb)


    def shutdown(self):
        rospy.signal_shutdown("Goal pose "+str(self.count)+" aborted, shutting down!")

if __name__ == '__main__':
    env = Env(4)
    print ("Trying to reset sim")
    rospy.init_node('movebase_action_client')
    env.reset()
    try:
        
        result = Move()

    except:
        rospy.loginfo("Error encountered in move_base initialization")

    print("RosPy got shut down, Bye Bye!")