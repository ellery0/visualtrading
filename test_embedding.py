from keras.models import load_model,Model
from keras import layers
from keras import models
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from keras.utils import to_categorical
import glob
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


model = models.Sequential()

embedding_model = load_model('embedding_model.h5')

layer_names = [layer.name for layer in embedding_model.layers]
layer_output = embedding_model.layers[5].output

latent_model = Model(inputs=embedding_model.input,outputs=layer_output)

model.add(latent_model)
#model.add(ConvLSTM2D(filters=40,kernel_size=(3,3),padding='same',return_sequences='true'))
#model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
#model.summary()

model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training ------------')
model.fit(train_images, train_labels, epochs=20, batch_size=200)
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(test_images, test_labels)


print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('embedding.h5')
















