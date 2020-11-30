import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util
import matplotlib.pyplot as plt

# summary_dir = 'tmp/summaries'
# summary_writer = tf.summary.create_file_writer('tmp/summaries')

# with summary_writer.as_default():
#   tf.summary.scalar('loss', 0.1, step=42)
#   tf.summary.scalar('loss', 0.2, step=43)
#   tf.summary.scalar('loss', 0.3, step=44)
#   tf.summary.scalar('loss', 0.4, step=45)


from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

# summary_dir = '/home/kenembanisi/.ros/logs/20201123-215943/' + \
# 'events.out.tfevents.1606186784.kenembanisi-Precision-Tower-3620.18721.0'
summary_dir = '/home/kenembanisi/.ros/logs/20201124-113023/' + \
'events.out.tfevents.1606235424.kenembanisi-Precision-Tower-3620.11398.0'

step_reward, avg_step_reward = [], []

# define averaging function:
def average(data, w_size):
    return[np.mean(data[i - w_size:i]) for i in range(w_size, len(data))]


count = 0
for e in tf.compat.v1.train.summary_iterator(summary_dir):
    for v in e.summary.value:
        if v.tag == 'step reward':
            step_reward.append(v.simple_value)

        if v.tag == 'average step reward (over 30 steps)':
            avg_step_reward.append(v.simple_value)
        
        count +=1
        # print(str(count))
    if count > 316900:
        break

print(str(len(step_reward)))

avg_rewards = average(step_reward, 100)

fig, axes = plt.subplots()


# axes.plot(step_reward, color = (255/255, 120/255, 50/255, 0.3), label = 'Reward')
axes.plot(avg_rewards, color = (255/255, 80/255, 10/255, 0.8), label = 'Average reward over 100 steps')
axes.set_xlabel("Steps")
axes.set_ylabel("Step rewards")
axes.legend()
axes.grid()
plt.show()
# for filename in os.listdir(summary_dir):
#     path = os.path.join(summary_dir, filename)
#     for event in my_summary_iterator(path):
#         for value in event.summary.value:
#             # t = tensor_util.MakeNdarray(value.simple_value)
#             t = value.simple_value
#             # print(value.tag, event.step, t, type(t))

#             if value.tag == 'step reward':
#                 step_reward.append(t)

#             if value.tag == 'average step reward (over 30 steps)':
#                 step_reward.append(t)

# print(str(step_reward))