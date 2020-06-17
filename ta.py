# draw graphs
# candle stick

#import talib
#from talib import CDL2CROWS
import numpy as np
import pandas as pd
#import matplotlib

from pandas_datareader import data as pdr
import mpl_finance
#from mpl_finance import candlestick_ohlc
#from mpl_finance import candlestick2_ochl

from datetime import datetime, timedelta
#import matplotlib.dates as mdates
#from matplotlib.pyplot import subplots, draw
import matplotlib.pyplot as plt
from matplotlib import ticker

from matplotlib.gridspec import GridSpec

day_num = 4000
window = 30
p=0.8


# get the data on a symbol (gets last 1 year)
symbol = "GOOGL"
data = pdr.get_data_yahoo(symbol, datetime.now() - timedelta(days=4000))
day_num = np.min([day_num,data.shape[0]])

# drop the date index from the dateframe
data.reset_index(inplace = True)
#print(data.columns)

# convert the datetime64 column in the dataframe to 'float days'
# data.Date = mdates.date2num(data.Date)
# data.Date = mdates.date2num(data.Date.dt.to_pydatetime())
from matplotlib.pylab import date2num
data['trade_date2'] = data['Date'].copy()
data['trade_date'] = pd.to_datetime(data['Date']).map(date2num)
data['dates'] = np.arange(0, len(data))


# make an array of tuples in the specific order needed
#dataAr = [tuple(x) for x in data[['Date', 'Open', 'Close', 'High', 'Low']].to_records(index=False)]

# construct and show the plot
label = np.zeros(day_num-1)
#candlestick_ohlc(ax1, dataAr,width=0.6, colorup='g', colordown='r', alpha=1.0)

data['5'] = data.Close.rolling(5).mean()
data['20'] = data.Close.rolling(20).mean()
# data['30'] = data.Close.rolling(30).mean()
# data['60'] = data.Close.rolling(60).mean()
# data['120'] = data.Close.rolling(120).mean()
# data['250'] = data.Close.rolling(250).mean()
colors = ['cyan','blue']
mas=['5','20']


data['up'] = data.apply(lambda row: 1 if row['Close'] >= row['Open'] else 0, axis=1)


date_tickers = data.Date.values

def format_date(x,pos):
    if x<0 or x>len(date_tickers)-1:
        return ''
    return date_tickers[int(x)]





for i in range(day_num-1):
    # fig = plt.figure()
    # ax1 = plt.subplot(1, 1, 1)

    figure = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 1)
    ax1 = plt.subplot(gs[:2, :])
    ax2 = plt.subplot(gs[2, :])

    #fig, ax = plt.subplots(figsize=(10, 5))

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    # candlestick2_ochl(ax,data['Open'][i:i+window],data['Close'][i:i+window],data['High'][i:i+window],data['Low'][i:i+window],colorup='g', colordown='r',width=1)
    label[i] = (np.sign(data['Close'][i+1]-data['Close'][i])+1)/2
    # #print(CDL2CROWS(data['Open'],data['High'],data['Low'],data['Close']))

    mpl_finance.candlestick_ochl(
         ax=ax1,
         quotes=data[['dates', 'Open', 'Close', 'High', 'Low']][i:i+window].values,
         width=0.7,
         colorup='r',
         colordown='g',
         alpha=0.7)
    for j in range(len(colors)):
         # plt.plot(data['dates'][i:i+window], data[ma][i:i+window])
         ax1.plot(data['dates'][i:i+window], data[mas[j]][i:i+window],linewidth=5,color=colors[j])
     #plt.axis('off')
    ax1.axis('off')
    
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax2.bar(data[i:i+window].query('up == 1')['dates'], data[i:i+window].query('up == 1')['Volume'], color='r', alpha=0.7)
    ax2.bar(data[i:i+window].query('up == 0')['dates'], data[i:i+window].query('up == 0')['Volume'], color='g', alpha=0.7)
    ax2.axis('off')
    
    if i<day_num*p:
        plt.savefig('train-pic-ma/traincandle_%s'%i)
    else:
        plt.savefig('test-pic-ma/testcandle_%s'%i)

label = pd.DataFrame(label)
pd.DataFrame.to_csv(label,'label-ma.txt',header=False,index=False)


# moving average

# volume







# data = pd.read_csv('data.csv', parse_dates={'Timestamp': ['Date', 'Time']}, index_col='Timestamp')
# ticks = data.ix[:, ['Price', 'Volume']]
# bars = ticks.Price.resample('1min', how='ohlc')
# barsa = bars.fillna(method='ffill')
# fig = plt.figure()
# fig.subplots_adjust(bottom=0.1)
# ax = fig.add_subplot(111)
# plt.title("Candlestick chart")
# volume = ticks.Volume.resample('1min', how='sum')
# value = ticks.prod(axis=1).resample('1min', how='sum')
# vwap = value / volume
# Date = range(len(barsa))
# #Date = matplotlib.dates.date2num(barsa.index)#
# DOCHLV = zip(Date , barsa.open, barsa.close, barsa.high, barsa.low, volume)
# candlestick_ohlc(ax, DOCHLV, width=0.6, colorup='g', colordown='r', alpha=1.0)
# plt.show()










