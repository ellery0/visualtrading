import numpy as np
import talib
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
day_num = 4000

symbol_list = pd.read_csv('/Users/fuyuting/Downloads/nasdaqsymbol.csv')
all_symbol = symbol_list['Symbol'].values

bearish_reversal = [
'CDL2CROWS',
'CDL3BLACKCROWS',
'CDLADVANCEBLOCK',
'CDLDARKCLOUDCOVER',
'CDLEVENINGDOJISTAR',
'CDLEVENINGSTAR',
'CDLHANGINGMAN',
'CDLIDENTICAL3CROWS',
'CDLUPSIDEGAP2CROWS',
'CDLSHOOTINGSTAR',
'CDLSTALLEDPATTERN'
]


bullish_reversal = [
'CDL3STARSINSOUTH',
'CDLCONCEALBABYSWALL',
'CDLHAMMER',
'CDLHOMINGPIGEON',
'CDLINVERTEDHAMMER',
'CDLMATCHINGLOW',
'CDLMORNINGDOJISTAR',
'CDLMORNINGSTAR',
'CDLPIERCING',
'CDLUNIQUE3RIVER',
'CDL3WHITESOLDIERS',
'CDLSTICKSANDWICH',
'CDLLADDERBOTTOM',
'CDLTAKURI'
]


def test_single_pattern(func):
    signal = func(open, high, low, close)
    dif = np.sign(-close.diff(periods=-1))
    signal_num = np.abs(signal.sum())/100
    is_signal = signal[signal != 0] / 100
    table = pd.concat([dif, is_signal], axis=1, join='inner')
    right = table[table['Close'] == table[0]].shape[0]
    acc = np.nan
    if signal_num!=0:
        acc = right / signal_num
    return right, signal_num, acc

def combine_patterns():
    return 0

accuracy_bearish = {}
accuracy_bullish = {}

for symbol in all_symbol:
    try:
        data = pdr.get_data_yahoo(symbol, datetime.now() - timedelta(days=day_num))
    except:
        continue
    # open = np.array([1.0,2.0])
    open = data['Open']
    # high = np.array([1.0,2.0])
    high = data['High']
    # low = np.array([1.0,2.0])
    low = data['Low']
    # close = np.array([1.0,2.0])
    close = data['Close']

    symbol_acc_bear = []
    for pattern in bearish_reversal:
        func = getattr(talib, pattern)
        right,signal_num,acc = test_single_pattern(func)
        symbol_acc_bear.append([right,signal_num,acc])
    # symbol_acc = pd.DataFrame(data=symbol_acc)
    #symbol_acc = {symbol:symbol_acc}
    #print(symbol_acc)
    accuracy_bearish[symbol]=symbol_acc_bear

    symbol_acc_bull = []
    for pattern in bullish_reversal:
        func = getattr(talib, pattern)
        right,signal_num,acc = test_single_pattern(func)
        symbol_acc_bull.append([right,signal_num,acc])
    accuracy_bullish[symbol]=symbol_acc_bull




def generate_result(pattern_dict,accuracy_result):
    total_right = np.zeros(len(pattern_dict))
    total = np.zeros(len(pattern_dict))

    for key in accuracy_result:
        for i in range(len(total_right)):
            total_right[i]+= accuracy_result[key][i][0]
            total[i] += accuracy_result[key][i][1]
    total_acc = total_right/total

    total_acc = [[total_right[x],total[x],total_acc[x]] for x in range(len(total))]

    accuracy_result['total_acc'] = total_acc
    #print(accuracy)
    df = pd.DataFrame(accuracy_result,columns = accuracy_result.keys())
    #print(df)
    #print(accuracy.values())
    return df



df_bear = generate_result(bearish_reversal,accuracy_bearish)
df_bear.to_csv('test_single_bearish_pattern.csv')

df_bull = generate_result(bullish_reversal,accuracy_bullish)
df_bull.to_csv('test_single_bullish_pattern.csv')













