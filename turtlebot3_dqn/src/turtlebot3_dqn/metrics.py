#!usr/bin/env python
import numpy as np
import math
import pickle
import csv
import statistics
import os


def calculate_dist(x2,y2,x1,y1):
    return math.sqrt ((x2-x1)**2 + (y2-y1)**2)

def calculate_path_length(x_list, y_list):
    path_length = 0.0
    if len(x_list) != len(y_list):
        print("Different Sizes of arrays")
        return
    for i in range (1,len(x_list)):
        path_length += calculate_dist(x_list[i],y_list[i],x_list[i-1],y_list[i-1])
    return path_length

def calculate_average_clearance(dist_list):
    return np.mean(dist_list)

def calculate_computation_time(x_list): # figure out convertion from size to seconds
    return len(x_list)

def calculate_smoothing(x_list, y_list):
    slope_list = []
    for i in range (1,len(x_list)):
        if x_list[i] - x_list[i-1] == 0.0:
            slope_list.append(math.inf)
        else:
            slope_list.append((y_list[i] - y_list[i-1])/(x_list[i] - x_list[i-1]))
    return statistics.stdev(slope_list)


# # Debugging
# x_data = np.array([1, 2, 3, 4, 5])
# y_data = np.array([1, 2, 3, 4, 5])
# clearance_data = np.array([1, 2, 3, 4, 5])
# print(calculate_path_length(x_data,y_data))
# print(calculate_average_clearance(clearance_data))
# print(calculate_computation_time(x_data), "time units") # figure out convertion from size to seconds

# with actual Numpy files
# a = np.load('a1.npy', allow_pickle=True)
# print("Size of a", a.size)
# print("Size of a", type(a))
# print(a)
# x_data = a[1]
# y_data = a[2]
# clearance_data = a[5]
# print("Path Length (in meters)", calculate_path_length(x_data,y_data))
# print("Average Clearance (in meters) ", calculate_average_clearance(clearance_data))
# print(calculate_computation_time(x_data), "time units")
# path_length = calculate_path_length(x_data,y_data)
# avg_clearance = calculate_average_clearance(clearance_data)
# computation_time = calculate_computation_time(x_data)
# smoothness = calculate_smoothing(x_data,y_data)

# # define path
# directory = os.path.dirname(os.path.abspath(__file__))+'/logs/'+'stage'+stage+'/'
log_directory = os.path.dirname(os.path.abspath(__file__))+'/logs/'


for i in range(1,5): # iterating through stage folders

    with open(log_directory+"stage_"+str(i)+'_summary.csv', mode='a') as output:
        dw = csv.DictWriter(output, delimiter='\t', fieldnames=['Success','Path Length (in meters)', 'Average Clearance (in meters)', 'Computation_time', 'Smoothness Factor'])
        dw.writeheader()

        success_count = 0
        total_count = 0
        path_length_list = []
        clearance_list = []
        time_list = []

        for trial_no in range(1,11): #iterating through trail numbers
            a = np.load(log_directory+"stage"+str(i) + "/turtlebot3_burger" + "_stage_" + str(i) + "_dqn_" + str(trial_no) + ".npy", allow_pickle=True)
            # if a[0] == 'success':
            #     success = 1
            # else:
            #     success = 0

            status = a[0]
            x_data = a[1]
            y_data = a[2]
            clearance_data = a[5]

            path_length = calculate_path_length(x_data,y_data)
            avg_clearance = calculate_average_clearance(clearance_data)
            computation_time = calculate_computation_time(x_data)
            smoothness = calculate_smoothing(x_data,y_data)
            # with open('output.csv', mode='a') as output:

            total_count += 1
            if status == "success":
                success_count += 1
                path_length_list.append(path_length)
                clearance_list.append(avg_clearance)
                time_list.append(computation_time)
            
            csv_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([ status, path_length,avg_clearance,computation_time, smoothness])
        
        csv_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([ '**********', '**********','**********','**********', '**********'])
        csv_writer.writerow([ 'Successful Trials', success_count,])
        csv_writer.writerow([ 'Total Trials',total_count,])
        csv_writer.writerow([ 'Percentage Success' ,100*float(success_count)/float(total_count)])
        csv_writer.writerow([ 'Mean Path Length (meters)' ,statistics.mean(path_length_list), 'Standard Deviation in Path Length' ,statistics.stdev(path_length_list)])
        csv_writer.writerow([ 'Mean Clearance (meters)' ,statistics.mean(clearance_list), 'Standard Deviation in Clearance' ,statistics.stdev(clearance_list)])
        csv_writer.writerow([ 'Mean Computation Time' ,statistics.mean(time_list), 'Standard Deviation in Computation Time' ,statistics.stdev(time_list)])
        # statistics.stdev(slope_list)
        csv_writer.writerow([ '**********', '**********','**********','**********', '**********'])