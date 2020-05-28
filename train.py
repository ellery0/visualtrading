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
train_images = np.array([np.array(Image.open(fname).resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB')) for fname in train_filelist])
train_images = train_images.astype('float32') / 255
test_images = np.array([np.array(Image.open(fname).resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB')) for fname in test_filelist])
test_images = test_images.astype('float32') / 255
#print(train_images.shape)
label = pd.read_csv('label-7.txt',header = None)


train_labels = label[window-1+207:window-1+int(train_images.shape[0])]
test_labels = label[int(train_images.shape[0])+window-1:int(train_images.shape[0])+window-1+400]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images[207:]
test_images = test_images[:400]


# pic0 = np.array(Image.open('train-pic/traincandle_0.png').convert('RGB'))
# pic0 = np.array(Image.open('train-pic/traincandle_0.png').resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB'))
# pd.DataFrame.to_csv(pd.DataFrame(pic0),'pic0.txt')

from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
model = models.Sequential()
model.add(layers.Conv2D(12, (11, 11), activation='relu', input_shape=(int(480/4), int(640/4), 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(12, (7, 7), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(12, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(12, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(12, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
# adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999, amsgrad=False)

model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training ------------')
model.fit(train_images, train_labels, epochs=20, batch_size=100)
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(test_images, test_labels)


print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('model.h5')
model.save('/Users/fuyuting/PycharmProjects/visualtrading/model.h5')

























