#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:40:24 2020

@author: fuy
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
#from tensorflow import set_random_seed
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Concatenate, Flatten, Dense, Dropout, Activation, Input, LSTM,Reshape, DepthwiseConv2D, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
#from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import plot_model
from scipy.stats import zscore
from sklearn.metrics import classification_report

# set random seeds
np.random.seed(1)

odb = pd.read_csv('/home/fuy/Downloads/lobster/AAPL_2019-07-05_24900000_57900000_orderbook_50.csv',header=None)
msg = pd.read_csv('/home/fuy/Downloads/lobster/AAPL_2019-07-05_24900000_57900000_message_50.csv',header=None)


def process_msg(data):
    data['time'] = data[0].diff(1)
    return data[['time',1]]


def hist_time(data):
    times = data[0].diff(1)
    times = times[2:]
    plt.hist(times)
    plt.show()
    return 0

def process_extreme(data):
    data = data.replace({9999999999:0,-9999999999:0})
    return data


def normalization(data):
    lev = int(data.shape[1]/2)
    #data[[2*i+1 for i in range(lev)]]=data[[2*i+1 for i in range(lev)]].apply(zscore)
    data[[2 * i + 1 for i in range(lev)]] = zscore(data[[2 * i + 1 for i in range(lev)]],axis = None)
    #data[[2 * i for i in range(lev)]] = data[[2 * i for i in range(lev)]].apply(zscore)
    data[[2 * i for i in range(lev)]] = zscore(data[[2 * i for i in range(lev)]], axis=None)
    return data

def get_midprice(data,days):
    data['mid_price'] = (data[0]+data[2])/2
    data['mid_price-10'] = data.mid_price.rolling(days).mean()
    rev = pd.DataFrame(data.mid_price.iloc[::-1].reset_index(drop=True))
    rev_mid = rev.mid_price.rolling(days).mean()
    data['mid_price+10'] = rev_mid.iloc[::-1].reset_index(drop=True)
    return data

def get_label1(data,threshold):
    data['compare'] = data['mid_price+10']
    data.loc[data['mid_price'] - data['compare']< -threshold,'label'] = 0
    data.loc[data['mid_price'] - data['compare'] > -threshold and data['mid_price'] - data['compare'] < threshold, 'label'] = 1
    data.loc[data['mid_price'] - data['compare'] > threshold, 'label'] = 2
    return data

def get_label(data,days,threshold):
    label_data = data.iloc[days:-days,:]
    label_data['compare'] = (label_data['mid_price+10']-label_data['mid_price'])/label_data['mid_price']
    label_data.loc[label_data['compare']< -threshold,'label'] = 0
    label_data.loc[label_data['compare'] > threshold, 'label'] = 2
    label_data['label'] = label_data.label.replace({np.nan: 1})
    label_data['label'].value_counts()
    return label_data  

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY


def prepare_data(trainX_CNN,NF):
    level = int(NF/4)
    trainX_ask_price = np.flip(trainX_CNN[:,:,[4*i for i in range(level)],:],2)
    trainX_ask_volume = np.flip(trainX_CNN[:,:,[4*i+1 for i in range(level)],:],2)
    trainX_bid_price = trainX_CNN[:,:,[4*i+2 for i in range(level)],:]
    trainX_bid_volume = trainX_CNN[:,:,[4*i+3 for i in range(level)],:]

    trainX_price = np.concatenate((trainX_ask_price,trainX_bid_price),axis=2)  
    trainX_volume = np.concatenate((trainX_ask_volume,trainX_bid_volume),axis=2)  
    trainX_combine = np.concatenate((trainX_price,trainX_volume),axis=3)  
    return trainX_combine  



def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))
    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
        
    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    
    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
    
    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)

    # use the MC dropout here
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)    
    
    # build the last LSTM layer
    # conv_lstm = CuDNNLSTM(number_of_lstm)(conv_reshape)
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='sigmoid')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1)
    sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def new_deeplob(T,NF,number_of_lstm):
    nf = int(NF/2)
    input_lmd = Input(shape=(T, nf, 2))
    conv_first1 = DepthwiseConv2D(2, (1, 1), padding='same')(input_lmd)
    #print('shape',conv_first1.shape)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    #print('shape',conv_first1.shape)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (1, int(nf/2)), strides=(1, int(nf/2)))(conv_first1)
    #print('shape',conv_first1.shape)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = Conv2D(32, (1, 2))(conv_first1)
    #print('shape',conv_first1.shape)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    #print('shape',conv_first1.shape)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    #print('shape',conv_first1.shape)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = Reshape((int(conv_first1.shape[1]), int(conv_first1.shape[3])))(conv_first1)
    
    
    # build the last LSTM layer
    # conv_lstm = CuDNNLSTM(number_of_lstm)(conv_reshape)
    conv_lstm = LSTM(number_of_lstm)(conv_first1)

    # build the output layer
    out = Dense(3, activation='sigmoid')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    sgd = keras.optimizers.SGD(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model








data = odb
data = process_extreme(data)
data = normalization(data) 
data = get_midprice(data,10)
data = get_label(data,10,0.001)  
data[['time','type']] = process_msg(msg) 


start = 100
train_num = 80000
test_num = 20000
end = 100
T = 20
NF = 40
level4 = 40

train_lob = np.array(data)[start:start+train_num,:level4]
train_lob_extra = np.array(data)[start:start+train_num,-2:]
train_lob = np.concatenate((train_lob,train_lob_extra),axis=1)

test_lob = np.array(data)[start+train_num:start+train_num+test_num,:level4]
test_lob_extra = np.array(data)[start+train_num:start+train_num+test_num,-2:]
test_lob = np.concatenate((test_lob,test_lob_extra),axis=1)


train_label = data['label'][start:start+train_num]
test_label = data['label'][start+train_num:start+train_num+test_num]

train_x, train_y = data_classification(train_lob, train_label, T) 
train_x = train_x*1000
train_y = np_utils.to_categorical(train_y, 3)

test_x, test_y = data_classification(test_lob, test_label, T) 
test_x = test_x*1000
test_label = test_y
test_y = np_utils.to_categorical(test_y, 3)





def cnn_model(T,NF):
    nf = int(NF/2)
    input_lmd = Input(shape=(T, nf, 2))
    conv_first1 = DepthwiseConv2D(2, (1, 1), padding='same')(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (1, int(nf/2)), strides=(1, int(nf/2)))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = Conv2D(32, (1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = Reshape((int(conv_first1.shape[1]), int(conv_first1.shape[3])))(conv_first1)

    return Model(input_lmd,conv_first1)




def fusion(number_of_lstm,T,NF):
    base_model = cnn_model(T,NF)
    input1 = base_model.input
    output1 = base_model.output
    input2 = Input(shape=(T,2))
    #merge = Concatenate()([output1,[input2]],axis=2)
    merge = keras.layers.concatenate([output1,input2],axis=2)
    merge = LSTM(number_of_lstm)(merge)
    out = Dense(3, activation='sigmoid')(merge)
    model = Model(inputs=[input1,input2], outputs=out)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    sgd = keras.optimizers.SGD(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



train_x_extra = train_x[:,:,-2:,:]
test_x_extra = test_x[:,:,-2:,:]

train_x = train_x[:,:,:-2,:]
test_x = test_x[:,:,:-2,:]

train_x = prepare_data(train_x,NF)    
test_x = prepare_data(test_x,NF)


#deeplob = create_deeplob(T, level4, 64)
#deeplob = new_deeplob(T,level4,64)
deeplob = fusion(64,T,40)

#deeplob=load_model('deeplob_50.h5')
#adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
#deeplob.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
early_stop = keras.callbacks.EarlyStopping('val_loss',min_delta=0,patience=20,verbose=0,mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint('./weights_50.h5',save_best_only=True)

#deeplob.fit(train_x, train_y, epochs=200, batch_size=64, verbose=2, validation_data=(test_x, test_y),\
#            callbacks=[tensorboard_callback,early_stop,checkpoint])

deeplob.fit([train_x,train_x_extra], train_y, epochs=200, batch_size=64, verbose=2, validation_data=([test_x,test_x_extra], test_y),\
            callbacks=[tensorboard_callback,early_stop,checkpoint])


y_pred = deeplob.predict(test_x,batch_size=64,verbose=1)
y_pred_bool=np.argmax(y_pred,axis=1)
print(classification_report(test_label,y_pred_bool))
deeplob.save('lobster_50_2.h5')

























