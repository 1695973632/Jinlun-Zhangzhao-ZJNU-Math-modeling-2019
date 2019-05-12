'''
这个程序作为实现ARIMA的模型来进行对每个时间序列的分析
输入的是小波分析后的每一阶的时间序列

@Model and program author: Jinlun
'''

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
from  statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from dataUtil import DataUtil
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

# dUtil = DataUtil()
# dUtil.getDataWithDate()
# ts_log = pd.Series(np.array(dUtil.Data[dUtil.goods[0]][1]))

# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()


# Dickey-Fuller test:
def teststationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dfoutput


def draw_moving(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    # rol_weighted_mean = pd.ewma(timeSeries, span=size)
    rol_weighted_mean=timeSeries.ewm(halflife=size,min_periods=0,adjust=True,ignore_na=False).mean()

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
# draw_acf_pacf(ts_diff_2,30)

def furtureCast(ts_log, steps):
    ts_log = pd.Series(ts_log)
    draw_moving(ts_log, 12)
    diff_12 = ts_log.diff(12)
    diff_12.dropna(inplace=True)
    diff_12_1 = diff_12.diff(1)
    diff_12_1.dropna(inplace=True)
    teststationarity(diff_12_1)
    rol_mean = ts_log.rolling(window=12).mean()
    rol_mean.dropna(inplace=True)
    ts_diff_1 = rol_mean.diff(1)
    ts_diff_1.dropna(inplace=True)
    teststationarity(ts_diff_1)
    ts_diff_2 = ts_diff_1.diff(1)
    ts_diff_2.dropna(inplace=True)
    teststationarity(ts_diff_2)
    model = ARIMA(ts_diff_1, order=(1,1,1))
    result_arima = model.fit( disp=-1, method='css')

    draw_acf_pacf(ts_diff_2, 30)

    predict_ts = result_arima.predict()
    # 一阶差分还原
    diff_shift_ts = ts_diff_1.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    # 再次一阶差分还原
    rol_shift_ts = rol_mean.shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    # 移动平均还原
    rol_sum = ts_log.rolling(window=11).sum()
    rol_recover = diff_recover*12 - rol_sum.shift(1)

    # rol_recover.plot(color='blue', label='Predict')
    # ts_log.plot(color='red', label='Original')
    # plt.legend(loc='best')
    # plt.title("wavelet fitting")
    # plt.show()

    future = result_arima.forecast(steps=steps)
    fc = future[0]

    return rol_recover.values, fc


