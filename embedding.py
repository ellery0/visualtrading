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



import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop

batch_size = 200
epochs = 15
inChannel = 3
x, y = train_images[0].shape[0],train_images[0].shape[1]
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder_train = autoencoder.fit(train_images, train_images, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images, test_images))

autoencoder.save('embedding_model.h5')