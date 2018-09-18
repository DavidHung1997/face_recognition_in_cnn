# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:13:59 2018

Convolutional Neural Network

Installing Theano
pip install --upgrade --no--deps git+git://github.com/Theano/Theano.git

installing Tensorflow
install Tensorflow from the website 

installing Keras 
pip install --upgrade keras 

@author: Lê Văn Hùng


"""

# part 1 : Building the CNN

# importing the Keras libraries and packages
# initialzing a nerual network either as sequence of layers or as a graph
from keras.models import Sequential
# to convolution
from keras.layers import Convolution2D
# to max pooling
from keras.layers import MaxPooling2D
# to Flatten
from keras.layers import Flatten
# to use to add the fully connected layers and a classic ANN
from keras.layers import Dense, Dropout

from keras.models import load_model

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Part 1 : Builiding a CNNs

# Initialising the CNN

# create a new object of the sequential class
classifier = Sequential()

# Step 1 : Convolution
# create 32 filters has 3x3 size as 32 features map
# 2 dimensional convolutional layer with input of 128 x 128 x 3 (height x width x RGB) dimension.
# use activation = 'relu' to remove number negative that has non-linear
classifier.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu'))
                
# step 2 : Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a third convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a four convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step 3 : flatten
classifier.add(Flatten())

# Step 4 : Full connection layer (hidden layer)
# choise number of output_dim not too smaller and not too big because compute hard and number of about 100 is good
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dropout(0.5))
# output layer
# use sigmoid function because we have  binary outcome Cat or dog
# if our outcome more than two categories then we need to use other activation as softMax activation
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling the CNNs
# lost funtion is outcome binary then chosing binary_crossentropy
# if we have more than two categories as Dog and Cat and Duc... then chosing  categorical_crossentropy
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Part 2 : Fitting the CNNS to the images


# prepare the image augmentation 
train_datagen = ImageDataGenerator(
        # we will rescale all our pixel values between 0 and 1
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
       
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')


# acc : accurancy
classifier.fit_generator(
        training_set,
        # we has 8000 images for training set
        samples_per_epoch=8000,
        epochs=50, # number of loop
        validation_data=test_set,
        # we has 2000 images for test set
        nb_val_samples=2000)



# creates a HDF5 file my_model_cats_dogs.h5
classifier.save('learn_cnn.h5')
print("Model saved !!")
# result = {'cats' : 0, 'dogs' : 1}
training_set.class_indices

# accurancy_training : ~93, accurancy_test : ~90





