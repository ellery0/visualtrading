#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:39:23 2020

@author: fuy
"""

import keras
from keras.models import load_model,Model
from keras import layers
from keras import models
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import LSTM, GlobalAveragePooling1D, TimeDistributed 
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from keras.utils import to_categorical
import glob
from keras.applications import vgg16

window=30

train_filelist,test_filelist = glob.glob('train-pic-ma/*.png'),glob.glob('test-pic-ma/*.png')
train_images = np.array([np.array(Image.open(fname).resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB')) for fname in train_filelist])
train_images = train_images.astype('float32') / 255
test_images = np.array([np.array(Image.open(fname).resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB')) for fname in test_filelist])
test_images = test_images.astype('float32') / 255
#print(train_images.shape)
label = pd.read_csv('label-ma.txt',header = None)

train_labels = label[window-1+207:window-1+int(train_images.shape[0])]
test_labels = label[int(train_images.shape[0])+window-1:int(train_images.shape[0])+window-1+400]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images[207:]
test_images = test_images[:400]


#window=30
#train_filelist,test_filelist = glob.glob('train-pic-ma/*.png')[:100],glob.glob('test-pic-ma/*.png')[:100]
#train_images = np.array([np.array(Image.open(fname).resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB')) for fname in train_filelist])
#train_images = train_images.astype('float32') / 255
#test_images = np.array([np.array(Image.open(fname).resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB')) for fname in test_filelist])
#test_images = test_images.astype('float32') / 255
##print(train_images.shape)
#label = pd.read_csv('label-ma.txt',header = None)
#
#train_labels = label[window-1+50:window-1+int(train_images.shape[0])]
#test_labels = label[int(train_images.shape[0])+window-1:int(train_images.shape[0])+window-1+4]
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)
#
#train_images = train_images[50:]
#test_images = test_images[:4]



base = vgg16.VGG16(input_shape=(int(480/4), int(640/4), 3),include_top=False, weights='imagenet',pooling='max')
for layer in base.layers:
    layer.trainable=False
layer_names=[layer.name for layer in base.layers]
print(layer_names)
base_model = Model(inputs=base.input,outputs=base.layers[-2].output)


model = models.Sequential()
model.add(base_model)
model.add(TimeDistributed(layers.GlobalAveragePooling1D()))
model.add(TimeDistributed(layers.Flatten()))
#model.add(LSTM(1280,return_sequences=True))
#model.add(TimeDistributed(layers.Dense(128, activation='relu')))
model.add(LSTM(128))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
#model.build(input_shape=(int(480/4), int(640/4), 3))
model.summary()

#model.add(ConvLSTM2D(filters=40,kernel_size=(3,3),padding='same',return_sequences='true'))
#model.add(BatchNormalization())
#model.add(layers.Flatten())
#model.add(TimeDistributed(layers.Dense(64, activation='softmax')))
#model.add(TimeDistributed(layers.Dense(1, activation='softmax')))
#model.add(layers.Dense(1, activation='softmax'))



opt = optimizers.SGD(decay=0.001)

model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
for i in range(17):
    print('Training ------------')
    model.fit(train_images, train_labels, epochs=3, batch_size=100)
    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(test_images, test_labels)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

model.save('vgglstm.h5')

























