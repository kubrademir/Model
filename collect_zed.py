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


# create model
def get_model(selection):
	# Model 1
	
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

	
		
	model_img.add(Dense(16))
	model_img.add(Dropout(0.2))
	model_img.add(Dense(1))

	

	model_img.compile(loss='mse', optimizer='adam')
	print(model_img.summary())
	return model_img

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
def train_model(model_img, image_data):
	batch_size = 2;
	epochs = 8;

	# checkpoints = []
	# if not os.path.exists('datasets/checkpoints/'):
	# 	os.makedirs('datasets/checkpoints')
	# checkpoints.append(ModelCheckpoint('datasets/checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
	# model.fit([images, lidars], [speeds, angles], batch_size=batch_size, epochs=epochs, validation_data=([images, lidars], [speeds, angles]), shuffle=True, callbacks=checkpoints)
	
	print('training is starting')
	model.fit([image_data], batch_size=batch_size, epochs=epochs)
	save_model(model_img)
	

# train model
model = get_model(selection = 1)
print('Model got')

image_data = get_image_data()
print('images got')


trained_model = train_model(model, image_data)

# evaluate model
score = model_img.evaluate([image_data])
print('Test loss:', score)
