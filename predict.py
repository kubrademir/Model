#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
import rospkg
import sys
import json
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from keras.models import model_from_json
from sensor_msgs.msg import Image, LaserScan, Joy, Imu
from cv_bridge import CvBridge, CvBridgeError
from scipy.misc import imresize

# globals
bridge = CvBridge()
rospack = rospkg.RosPack()

is_driving = True
zed_msg = None
lidar_msg = None
joy_msg = Joy()
imu_msg = Imu()

# load model
package_path = rospack.get_path('deep_learning')
model_path = package_path + '/scripts/model/005/model.h5'
json_path = package_path + '/scripts/model/005/model.json'

json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights(model_path)

# helpers
def ros_to_opencv_img(data):
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	return cv_image

def get_imu_data(imu):
	imu_data = []

	imu_data.append(imu.orientation.x)
	imu_data.append(imu.orientation.y)
	imu_data.append(imu.orientation.z)
	imu_data.append(imu.orientation.w)

	imu_data.append(imu.angular_velocity.x)
	imu_data.append(imu.angular_velocity.y)
	imu_data.append(imu.angular_velocity.z)

	imu_data.append(imu.linear_acceleration.x)
	imu_data.append(imu.linear_acceleration.y)
	imu_data.append(imu.linear_acceleration.z)

	imu_data = np.array(imu_data).reshape(1, 10)
	return imu_data;

def get_image_data(zed):
	cv2_image = ros_to_opencv_img(zed)
	cv2_image = imresize(cv2_image, (376, 1344, 3))
	cv2_image = np.array(cv2_image).reshape(1, 376, 1344, 3)
	return cv2_image

def get_lidar_data(lidar):
	# print(np.array(lidar.ranges))
	
	lidar_ranges = np.array(lidar.ranges).reshape(1, 360)
	# print(len(lidar_ranges))
	return lidar_ranges

# callback functions
def handle_camera(data):
	global zed_msg
	zed_msg = data

def handle_lidar(data):
	global lidar_msg
	lidar_msg = data

def handle_joy(data):
	global is_driving
	a_button = data.buttons[1] # A
	b_button = data.buttons[2] # B

	if not is_driving and a_button == 1:
		is_driving = True

	if is_driving and b_button == 1:
		is_driving = False

def handle_imu(data):
	global imu_msg
	imu_msg = data

# predict steering angle and speed to given image and lidar data
def predict(image, lidar, imu):
	steering_angle = 0
	speed = 2

	output = model.predict([image, lidar, imu], batch_size=1)
	print(output)

	steering = output[0][0]
	return {'steering_angle': steering_angle, 'speed': speed}

def node():
	rospy.init_node('self_drive', anonymous=True)
	rate = rospy.Rate(20)

	zed_sub = rospy.Subscriber("/zed/rgb/image_rect_color", Image, handle_camera, queue_size=1)
 	scan_sub = rospy.Subscriber('/scan', LaserScan, handle_lidar, queue_size=1)
 	joy_sub = rospy.Subscriber('/joy', Joy, handle_joy, queue_size=1)
 	imu_sub = rospy.Subscriber('/imu', Imu, handle_imu, queue_size=1)

 	pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
	
	while not rospy.is_shutdown():
		rate.sleep()
		
		#print('')
		if is_driving and zed_msg is not None and lidar_msg is not None:
			prediction = predict(get_image_data(zed_msg), get_lidar_data(lidar_msg), get_imu_data(imu_msg))
			msg = AckermannDriveStamped()
			msg.drive.speed = prediction['speed']
			msg.drive.steering_angle = prediction['steering_angle']
			pub.publish(msg)

if __name__ == '__main__':
	node()
