 #!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2


ROWSIZE = 0
COLSIZE = 0

b_FIRST_INITIALIZATION = False

image = np.zeros([60,60,1], dtype=np.uint8)


# occupancy map callback
def costmapCallBack(scan):
    rospy.loginfo(scan.header)
    
    global ROWSIZE 
    global COLSIZE
    global b_FIRST_INITIALIZATION
    global image

    if not b_FIRST_INITIALIZATION:
        ROWSIZE = scan.info.height
        COLSIZE = scan.info.width
        b_FIRST_INITIALIZATION = True

    image = imageReset()
    
    size = len(scan.data) 

    for k in range(size):
        image[COLSIZE - 1 - k % COLSIZE , ROWSIZE - 1 - k /COLSIZE] = scan.data[k] * 255 / 100

    image = cv2.resize(image, (2*ROWSIZE, 2*COLSIZE))
    dispImage(image)


# return the occupancy map image
def getImage():
    return image


# return black image
def imageReset():
    return np.zeros([ROWSIZE , COLSIZE, 1], dtype = np.uint8)

# display image
def dispImage(image):
    cv2.imshow("Occupancy Image", image)
    cv2.waitKey(1)

# initiates node and subscriber
def costmap2Image():

    sub = rospy.Subscriber ("/move_base/local_costmap/costmap", OccupancyGrid,costmapCallBack)
    rospy.init_node('costmap2image_node', anonymous=True)
    # rate = rospy.Rate(10)

    rospy.spin()


if __name__ == '__main__':
    try:
        costmap2Image()
    
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()