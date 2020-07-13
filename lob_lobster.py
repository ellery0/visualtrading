import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import set_random_seed
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input,CuDNNLSTM, LSTM,Reshape, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import plot_model
from scipy.stats import zscore



odb = pd.read_csv('/Users/fuyuting/Downloads/AAPL_07Jan--05Jul/AAPL_2019-07-05_24900000_57900000_orderbook_50.csv',header=None)
msg = pd.read_csv('/Users/fuyuting/Downloads/AAPL_07Jan--05Jul/AAPL_2019-07-05_24900000_57900000_message_50.csv',header=None)

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

def get_mid_price(data,days):
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



data = odb
data = process_extreme(data)
data = normalization(data)


def fusion():

    return 0










