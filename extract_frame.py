#!/usr/bin/python

# Extract images from a bag file.

#PKG = 'beginner_tutorials'
import roslib;   #roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

# Reading bag filename from command line or roslaunch parameter.
#import os
#import sys


class ImageCreator():
    frames = -1  # 将frames设置为类变量
    def __init__(self):
        self.bridge = CvBridge()
        with rosbag.Bag('/media/seu/4000098A0009885C/datasaets/fast-livo/hku1.bag', 'r') as bag:  # 要读取的bag文件；
            for topic, msg, t in bag.read_messages():
                if topic == "/left_camera/image":  # 图像的topic；
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    except CvBridgeError as e:
                        print(e)
                    ImageCreator.frames += 1  # 更新帧计数
                    image_name = "frame" + str(ImageCreator.frames).zfill(6) + ".jpg"  # 使用zfill确保四位数命名
                    cv2.imwrite(image_name, cv_image)  # 保存；
                    print(f"Saved: {image_name}")  # 输出保存的信息

if __name__ == '__main__':

    #rospy.init_node(PKG)

    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass


