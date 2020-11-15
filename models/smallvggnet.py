# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

class SmallVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 3 => POOL layer set
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

class VGG19:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model.add(ZeroPadding2D((1, 1), input_shape=inputShape))
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

	# model = Sequential()
	# model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
	# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#
	# # model.add(Flatten())
	# model.add(Dense(4096, activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(4096, activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(13, activation='softmax'))
	#
	# # softmax classifier
	# # model.add(Dense(classes))
	# # model.add(Activation("softmax"))
	#
	# if weights_path:
	# 	model.load_weights(weights_path)