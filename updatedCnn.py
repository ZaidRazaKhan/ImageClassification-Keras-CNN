# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (3, 3)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()

# Part 2 - Fitting the CNN to the images
# originally batch size was 32
batch_size = 20
fixed_size = (150,150)
class_mode = 'categorical'
test_set_location = 'dataset/validation/'
train_set_location = 'dataset/train/'
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set  = train_datagen.flow_from_directory(
	train_set_location,
	target_size = fixed_size,
	batch_size = batch_size,
	class_mode = class_mode)
image_array  = []
output_array = []
print(training_set)
for i in range (1, 5000):
	file_name = 'dataset/training/' + str(i) + '.png'
	img = cv2.imread(file_name)
	ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	image_array.append(thresh1)
	# image_array.append(cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
 #            cv2.THRESH_BINARY,11,2))
	# image_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


import csv
with open('./dataset/solution.csv') as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
    	output_array.append(line[1])
    	# print('elemenet{} = {}'.format(i,line[1]))
		# print('line[{}] = {}'.format(i, line))
# csv_file = open('./dataset/solution.csv')
# csv_f = csv.reader(csv_file)
# for row in csv_file:
	# print(row[0])
	# output_array.append(row[1])
output_array.pop(0)

# print(output_array)
test_set = test_datagen.flow_from_directory(
	test_set_location,
	target_size = fixed_size,
	batch_size = batch_size,
	class_mode = class_mode)
# print(test_set.n)
STEP_SIZE_TRAIN=training_set.n//training_set.batch_size
STEP_SIZE_VALID=test_set.n//test_set.batch_size

classifier.fit(
	x = image_array, 
	y = output_array,
	batch_size = 20, 
	epochs = 1, 
	validation_split = 0.3 , 
	validation_data = test_set, 
	steps_per_epoch = STEP_SIZE_TRAIN, 
	validation_steps = STEP_SIZE_VALID)

# classifier.fit_generator(generator = training_set,
# steps_per_epoch = STEP_SIZE_TRAIN,
# epochs = 25,
# validation_data = test_set,
# validation_steps = STEP_SIZE_VALID)


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/train/1/1.png', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(prediction)
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
