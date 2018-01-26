#!/usr/bin/python

import cv2
import numpy as np
import os
import time
import rospkg
import sys
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image, LaserScan, Joy, Imu
from cv_bridge import CvBridge, CvBridgeError

# globals
bridge = CvBridge()
rospack = rospkg.RosPack()

ackermann_msg = None
zed_msg = None
lidar_msg = None
joy_msg = Joy()
imu_msg = Imu()
is_collecting = False

# helper functions
def ros_to_opencv_img(data):
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	return cv_image

# callback functions
def handle_ackermann(data):
	global ackermann_msg
	ackermann_msg = data

def handle_camera(data):
	global zed_msg
	zed_msg = ros_to_opencv_img(data)

def handle_lidar(data):
	global lidar_msg
	lidar_msg = data

def handle_joy(data):
	global is_collecting
	a_button = data.buttons[1] # A
	b_button = data.buttons[2] # B

	if not is_collecting and a_button == 1:
		is_collecting = True

	if is_collecting and b_button == 1:
		is_collecting = False

def handle_imu(data):
	global imu_msg
	imu_msg = data


# logger class
class Logger(object):
	def __init__(self):
		package_path = rospack.get_path('deep_learning')
		path = package_path + '/datasets/'
		if not os.path.exists(path):
			os.makedirs(path)
		i=1
		while True:
			dir_name = path + '%03d' % i
			if os.path.exists(dir_name):
				i += 1
			else:
				os.makedirs(dir_name)
				break

		self.path = dir_name + '/'
		self.dir_index = i
		self.filep = file(self.path + 'data.csv', 'w+')
		self.filep.write('Speed,SteeringAngle,Image,Lidar,Imu\n')
		self.index = 0

	def write(self, speed, steering_angle, image, lidar_data, imu_data):
		image_name = self.path + str(self.dir_index) + '_' +'%05d.jpg' % self.index
		image_name_cv2 = str(self.dir_index) + '_' + '%05d.jpg' % self.index
		
		print(image_name)
		cv2.imwrite(image_name, image)

		# get string of ranges joined with space ' '
		ranges_str = ''
		for r in lidar_data:
			ranges_str += '%.3f ' % r

		# get imu string
		imu_str = ''
		imu_str += str(imu_data.orientation.x)
		imu_str += ' '
		imu_str += str(imu_data.orientation.y)
		imu_str += ' '
		imu_str += str(imu_data.orientation.z)
		imu_str += ' '
		imu_str += str(imu_data.orientation.w)
		imu_str += ' '
		imu_str += str(imu_data.angular_velocity.x)
		imu_str += ' '
		imu_str += str(imu_data.angular_velocity.y)
		imu_str += ' '
		imu_str += str(imu_data.angular_velocity.z)
		imu_str += ' '
		imu_str += str(imu_data.linear_acceleration.x)
		imu_str += ' '
		imu_str += str(imu_data.linear_acceleration.y)
		imu_str += ' '
		imu_str += str(imu_data.linear_acceleration.z)

		# imu_str = '1 2 3 4 5 6 7 8 9 10'


		# get csv line
		line = str(speed) + ',' + str(steering_angle) + ',' + image_name + ',' + ranges_str + ',' + imu_str +'\n'
		self.filep.write(line)
		self.index += 1	

	def close(self):
		self.filep.close()

# main node
def node():

	logger = Logger()
	rospy.init_node('collect_data')
	rate = rospy.Rate(20)
	
	# set subscribers
	zed_sub = rospy.Subscriber("/zed/rgb/image_rect_color", Image, handle_camera, queue_size=1)
 	ackermann_sub = rospy.Subscriber('/ackermann_cmd',AckermannDriveStamped, handle_ackermann, queue_size=1)
 	scan_sub = rospy.Subscriber('/scan', LaserScan, handle_lidar, queue_size=1)
 	joy_sub = rospy.Subscriber('/joy', Joy, handle_joy, queue_size=1)
 	imu_sub = rospy.Subscriber('/imu', Imu, handle_imu, queue_size=1)

	print(lidar_msg)
	while not rospy.is_shutdown():
		rate.sleep()
		time.sleep(0.04)
		if is_collecting and zed_msg is not None and lidar_msg is not None and ackermann_msg is not None and imu_msg is not None and ackermann_msg.drive.speed > 0:
			logger.write(ackermann_msg.drive.speed, ackermann_msg.drive.steering_angle, zed_msg, lidar_msg.ranges, imu_msg)
	
	# rospy.spin()
	logger.close()
		
if __name__ == '__main__':
	node()


# todo: make image 3d

# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# def get_zed_data(img_left, img_right):
#     img_left = img[0:376, 0:672]
#     img_right = img[0:376, 672:1344]
#     disparity = stereo.compute(img_left,img_right)
#     return disparity