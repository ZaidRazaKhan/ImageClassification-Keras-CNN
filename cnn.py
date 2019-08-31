import os
import numpy as np 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model

img_width, img_height = 150,150

fixed_size = tuple((img_width, img_height))

train_data_dir = 'dataset/train'

validation_data_dir = 'dataset/validation'

## preprocessing
# used to rescale the pixel values from [0,255] to [0,1]
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

# automatically retrieve images and their classes for train and validation sets

train_generator = datagen.flow_from_directory(
	train_data_dir,
	target_size = fixed_size,
	batch_size = batch_size,
	class_mode = 'binary')

validation_generator = datagen.flow_from_directory(
	validation_data_dir,
	target_size = fixed_size,
	batch_size = batch_size,
	class_mode = 'binary')

# small convlotional network # simple set of 3 convolutional layers with a ReLU activation and followed by max-pooling layers
model = Sequential()
model.add(Convolution2D(32,(3,3), input_shape = (img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
	optimizer = 'rmsprop',
	metrics = ['accuracy'])


#Training

epochs = 100
train_samples = 2048
validation_samples = 384

model.fit_generator(
	train_generator,
	steps_per_epoch = train_samples //batch_size,
	epochs = epochs,
	validation_data = validation_generator,
	validation_steps = validation_samples //batch_size)

# About 60 seconds an epoch when using CPU

model.save_weight('output/basic_cnn_100_epochs.h5')

# Evaluating on validation set for Computing loss and accuracy
model.evaluate_generator(validation_generator,validation_samples)
