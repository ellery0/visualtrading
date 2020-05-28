import tensorflow as tf
import torch
import pandas as pd
import numpy as np
from PIL import Image
from keras.utils import to_categorical
import glob
import pickle
window=7
train_filelist,test_filelist = glob.glob('train-pic-7/*.png'),glob.glob('test-pic-7/*.png')
train_images = np.array([np.array(Image.open(fname).resize((224,224),Image.ANTIALIAS).convert('RGB')) for fname in train_filelist])
train_images = train_images.astype('float32') / 255
test_images = np.array([np.array(Image.open(fname).resize((224,224),Image.ANTIALIAS).convert('RGB')) for fname in test_filelist])
test_images = test_images.astype('float32') / 255
# print('train_images_shape',train_images.shape)
label = pd.read_csv('label-7.txt',header = None)


train_labels = label[window-1+207:window-1+int(train_images.shape[0])]
test_labels = label[int(train_images.shape[0])+window-1:int(train_images.shape[0])+window-1+400]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images[207:]
test_images = test_images[:400]

from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,GlobalAveragePooling2D

base_model = applications.resnet50.ResNet50(weights=None,include_top=False,input_shape=(224,224,3))
x  = base_model.output
print(x.shape,'x-shape')
#x = GlobalAveragePooling2D(x)
x = Dropout(0.7)(x)
x = Flatten()(x)
predictions = Dense(2,activation='softmax')(x)
model = Model(inputs = base_model.input,outputs=predictions)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training ------------')
model.fit(train_images, train_labels, epochs=10, batch_size=100)
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(test_images, test_labels)


print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('res-model.h5')
model.save('/Users/fuyuting/PycharmProjects/visualtrading/model-res.h5')