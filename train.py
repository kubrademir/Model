import pandas as pd
import numpy as np
import cv2
import matplotlib.pylab as plt
import random
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Merge
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import concatenate, Input, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from scipy.misc import imread, imresize





print('reading csv')
csv_file = pd.read_csv('/home/nvidia/racecar-ws/src/examples/deep_learning/datasets/023/data.csv')
k = 0


# helpers
def get_img(path):
	global k
	print(k)
	k += 1
	img = cv2.imread(path)
	img = imresize(img, (376, 1344, 3))

	#img = img[img.shape[0]/2:,:,:]
	return img

# get csv dataset
def get_image_data():
	image_paths = list(csv_file.Image)
	image_data = []
	for i in image_paths:
		print(i)
		image_data.append(get_img(i))
	image_data = np.array(image_data)
	return image_data

def get_lidar_data():
	lidar_data = list(csv_file.Lidar)
	# normalize lidar data
	for i in xrange(len(lidar_data)):
		lidar_data[i] = lidar_data[i].split(' ')
		del lidar_data[i][-1]
		for j in xrange(len(lidar_data[i])):
			if lidar_data[i][j] == 'inf':
				lidar_data[i][j] = 40
			if(float(lidar_data[i][j])):
				lidar_data[i][j] = float(lidar_data[i][j])
	lidar_data = np.array(lidar_data)
	return lidar_data

def get_imu_data():
	imu_data = list(csv_file.Imu)
	# normalize imu data
	for i in xrange(len(imu_data)):
		imu_data[i] = imu_data[i].split(' ')
		for j in xrange(len(imu_data[i])):
			imu_data[i][j] = float(imu_data[i][j])
			imu_data[i][j] = '%.3f' % imu_data[i][j]
			imu_data[i][j] = float(imu_data[i][j])

	imu_data= np.array(imu_data)
	return imu_data

def get_speed_data():
	speed_data = list(csv_file.Speed)
	speed_data = np.array(speed_data)
	return speed_data

def get_angle_data():
	angle_data = list(csv_file.SteeringAngle)
#	image_data = list(csv_file.Image)
#
#	nitem = len(image_data)
#	for i in xrange(nitem):
#	    if angle_data[i] > 0.05:
#		for j in xrange(7):
#		    image_data.append(image_data[i])
#		    angle_data.append(angle_data[i])    
#	    if angle_data[i] < -0.07:
#		for j in xrange(7):
#		    image_data.append(image_data[i])
#		    angle_data.append(angle_data[i])

	angle_data = np.array(angle_data)
	return angle_data

# create model
def get_model(selection):
	# Model 1
	if selection == 1:
		model_img = Sequential(name="img")
		# Cropping
		model_img.add(Cropping2D(cropping=((124,126),(0,0)), input_shape=(376,1344,3)))
		# Normalization
		model_img.add(Lambda(lambda x: (2*x / 255.0) - 1.0))
		model_img.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(64, (3, 3), activation="relu"))
		model_img.add(Conv2D(64, (3, 3), activation="relu"))
		model_img.add(Flatten())
		model_img.add(Dense(100))
		model_img.add(Dense(50))
		model_img.add(Dense(10))

		model_lidar = Sequential(name="lidar")
		model_lidar.add(Dense(32, input_shape=(360,)))
		model_lidar.add(Dropout(0.1))
		model_lidar.add(Dense(10))

		model_imu = Sequential(name='imu')
		model_imu.add(Dense(32, input_shape=(10, )))
		model_imu.add(Dropout(0.1))
		model_imu.add(Dense(10))

		merged = Merge([model_img, model_lidar, model_imu], mode="concat")
		model = Sequential()
		model.add(merged)
		model.add(Dense(16))
		model.add(Dropout(0.2))
		model.add(Dense(1))

	# Model 2
	elif selection == 2:
		model_img = Sequential(name="img")
		# Cropping
		model_img.add(Cropping2D(cropping=((124,126),(0,0)), input_shape=(376,1344,3)))
		# Normalization
		model_img.add(Lambda(lambda x: (2*x / 255.0) - 1.0))
		model_img.add(Conv2D(16, (7, 7), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(32, (7, 7), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(32, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(64, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(64, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(128, (3, 3), activation="relu"))
		model_img.add(Conv2D(128, (3, 3), activation="relu"))
		model_img.add(Flatten())
		model_img.add(Dense(100))
		model_img.add(Dense(50))
		model_img.add(Dense(10))

		model_lidar = Sequential(name="lidar")
		model_lidar.add(Dense(32, input_shape=(360,)))
		model_lidar.add(Dropout(0.1))
		model_lidar.add(Dense(10))

		model_imu = Sequential(name='imu')
		model_imu.add(Dense(32, input_shape=(10, )))
		model_imu.add(Dropout(0.1))
		model_imu.add(Dense(10))

		merged = Merge([model_img, model_lidar, model_imu], mode="concat")
		model = Sequential()
		model.add(merged)
		model.add(Dense(16))
		model.add(Dropout(0.2))
		model.add(Dense(1))

	# Model 3
	elif selection == 3:
		model_img = Sequential(name="img")
		# Cropping
		model_img.add(Cropping2D(cropping=((124,126),(0,0)), input_shape=(376,1344,3)))
		# Normalization
		model_img.add(Lambda(lambda x: (2*x / 255.0) - 1.0))
		model_img.add(Conv2D(16, (7, 7), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(32, (7, 7), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(32, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(64, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(64, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(128, (3, 3), activation="relu"))
		model_img.add(Conv2D(128, (3, 3), activation="relu"))
		model_img.add(Flatten())
		model_img.add(Dense(100))
		model_img.add(Dense(50))
		model_img.add(Dense(10))

		model_lidar = Sequential(name="lidar")
		model_lidar.add(Dense(32, input_shape=(360,)))
		model_lidar.add(Dropout(0.1))
		model_lidar.add(Dense(10))

		model_imu = Sequential(name='imu')
		model_imu.add(Dense(32, input_shape=(10, )))
		model_imu.add(Dropout(0.1))
		model_imu.add(Dense(10))

		merged = Merge([model_img, model_lidar, model_imu], mode="concat")
		model = Sequential()
		model.add(merged)
		model.add(Dense(16))
		model.add(Dropout(0.2))
		model.add(Dense(1))

	# Model 4
	elif selection == 4:
		model_img = Sequential(name="img")
		# Cropping
		model_img.add(Cropping2D(cropping=((124,126),(0,0)), input_shape=(376,1344,3)))
		# Normalization
		model_img.add(Lambda(lambda x: (2*x / 255.0) - 1.0))
		model_img.add(Conv2D(16, (7, 7), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(32, (7, 7), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(32, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(64, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(64, (5, 5), activation="relu", strides=(2, 2)))
		model_img.add(Conv2D(128, (3, 3), activation="relu"))
		model_img.add(Conv2D(128, (3, 3), activation="relu"))
		model_img.add(Flatten())
		model_img.add(Dense(100))
		model_img.add(Dense(50))
		model_img.add(Dense(10))

		model_lidar = Sequential(name="lidar")
		model_lidar.add(Dense(32, input_shape=(360,)))
		model_lidar.add(Dropout(0.1))
		model_lidar.add(Dense(10))

		model_imu = Sequential(name='imu')
		model_imu.add(Dense(32, input_shape=(10, )))
		model_imu.add(Dropout(0.1))
		model_imu.add(Dense(10))

		merged = Merge([model_img, model_lidar, model_imu], mode="concat")
		model = Sequential()
		model.add(merged)
		model.add(Dense(16))
		model.add(Dropout(0.2))
		model.add(Dense(1))



	model.compile(loss='mse', optimizer='adam')
	print(model.summary())
	return model

# augmentate data
def optimize_data():
	# todo: augmentate data, brightness, rotation etc
	print('heyylo')

def save_model(model):
	# create model directory
	path_to_save = ''

	if not os.path.exists('model/'):
		os.makedirs('model/')
	i=1
	while True:
		dir_name = 'model/' + '%03d' % i
		if os.path.exists(dir_name):
			i += 1
		else:
			os.makedirs(dir_name)
			path_to_save = dir_name
			break

	#Save the model
	# serialize model to JSON
	model_json = model.to_json()
	with open(path_to_save + '/' + "model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(path_to_save + '/' + "model.h5")
	print("Saved model to disk")


# train function
def train_model(model, image_data, lidar_data, imu_data, speed_data, angle_data):
	batch_size = 2;
	epochs = 8;

	# checkpoints = []
	# if not os.path.exists('datasets/checkpoints/'):
	# 	os.makedirs('datasets/checkpoints')
	# checkpoints.append(ModelCheckpoint('datasets/checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
	# model.fit([images, lidars], [speeds, angles], batch_size=batch_size, epochs=epochs, validation_data=([images, lidars], [speeds, angles]), shuffle=True, callbacks=checkpoints)
	
	print('training is starting')
	model.fit([image_data, lidar_data, imu_data], angle_data, batch_size=batch_size, epochs=epochs)
	save_model(model)
	

# train model
model = get_model(selection = 1)
print('Model got')

image_data = get_image_data()
print('images got')

lidar_data = get_lidar_data()
print('lidar got')

imu_data = get_imu_data()
print('imu got')

speed_data = get_speed_data()
print('speed')

angle_data = get_angle_data()
print('angle got')

trained_model = train_model(model, image_data, lidar_data, imu_data, speed_data, angle_data)

# evaluate model
score = model.evaluate([image_data, lidar_data, imu_data], angle_data)
print('Test loss:', score)
