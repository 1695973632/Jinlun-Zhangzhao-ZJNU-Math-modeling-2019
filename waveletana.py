'''
我们正式开始第一题的解题，通过小波变换来对这些数据进行分解
最后组合这些个系数来实现对最后十天的预测，至于问什么不用神经网络
其实还是因为数据量太小，总共就330多个样例，时间片还不可以组合
如果数据量上万了，那么用LSTM来求解其实还是不错的，毕竟那东西理论可以拟合所有曲线
但是在这种数据量的情况下百分之一百是会欠拟合的

@Model and program author: Jinlun
'''

from dataUtil import DataUtil
import pywt
import numpy as np
import matplotlib.pyplot as plt
import ARIMAUtil
import pandas as pd

def plotwavelet(y, ya4, yd4, yd3, yd2, yd1):
    '''
    可视化小波分析的结果
    :param y:
    :param ya4:
    :param yd4:
    :param yd3:
    :param yd2:
    :param yd1:
    :return:
    '''
    plt.figure(figsize=(12, 12))
    plt.subplot(611)
    plt.plot(y)
    plt.title('original signal')
    plt.subplot(612)
    plt.plot(ya4)
    plt.title('approximated component in level 4')
    plt.subplot(613)
    plt.plot(yd4)
    plt.title('detailed component in level 4')
    plt.subplot(614)
    plt.plot(yd3)
    plt.title('detailed component in level 3')
    plt.subplot(615)
    plt.plot(yd2)
    plt.title('detailed component in level 2')
    plt.subplot(616)
    plt.plot(yd1)
    plt.title('detailed component in level 1')
    plt.tight_layout()
    plt.show()

def cutMinus(num):
    return np.where(num<1, 1, num)

def main():
    algorithm = algorithmUtil()
    goods = algorithm.dUtil.goods

    for i in range(5):
        [y, ya4, yd4, yd3, yd2, yd1] = algorithm.waverec(goods[i])
        plotwavelet(y, ya4, yd4, yd3, yd2, yd1)
        ya41, fa4 = algorithm.predictResult(ya4)
        yd41, fd4 = algorithm.predictResult(yd4)
        yd31, fd3 = algorithm.predictResult(yd3)


        ys = ya41 + yd41 + yd31
        fd = 2*(fa4 + fd4 + fd3)+1

        loc = ys.shape[0]
        xx = np.linspace(loc, loc + 15, 15)

        plt.plot(y, color='red', label='original')
        plt.plot(ys, color='blue', label='fit')

        yo = y[15:]
        yp = ys[15:y.shape[0]]
        RMSE = np.sqrt(np.dot(yp-yo, yp-yo)/(y.shape[0]-15))
        SSR = np.dot(yp-yo, yp-yo)/(y.shape[0]-15)
        SST = np.dot(yo-np.mean(yo), yo-np.mean(yo))/(y.shape[0]-15)
        RSquare = SSR/SST

        plt.title('RMSE:%f  R-Square:%f'%(RMSE, RSquare))
        plt.xlabel('time scale')
        plt.ylabel('volume')

        plt.plot(xx, fd, color='black', label='forecast')
        plt.legend(loc='best')

        plt.show()


        # xs = np.linspace(100,500,400)
        # xx = np.linspace(0, y.shape[0], y.shape[0])
        #
        # plt.plot(xx, y, 'r')
        # plt.plot(xs, ys, 'b')
        #
        # plt.show()



class algorithmUtil():
    def __init__(self):
        # 获取数据处理工具
        self.dUtil = DataUtil()
        self.dUtil.getDataWithDate()
    def waverec(self, key):
        # 生成四层小波模型
        y = np.array(self.dUtil.Data[key][1][1:])
        coeffs = pywt.wavedec(y, 'db4', level=4)

        ya4 = pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0]).tolist(), 'db4')
        yd4 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0]).tolist(), 'db4')
        yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0]).tolist(), 'db4')
        yd2 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0]).tolist(), 'db4')
        yd1 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1]).tolist(), 'db4')
        #     返回四层小波
        return [y, ya4, yd4, yd3, yd2, yd1]

    def dataEvaluation(self, timestep):
        '''
        输入时间序列，评估这个序列, 判断出是否存在
        趋势，如果存在趋势，我们就选择ARIMA模型来做
        :param timestep:
        :return:
        '''
        pass

    def predictResult(self, timestep):
        """
        对时间序列来进行预测,首先我们通过可视化数据了解到整体有周期性趋势
        所以，我们计算每个频段的周期，然后将其整合，但如果从开始之后就没有波动
        那么久可以基本判断出周期为无限（也就是直接判断为所有10以下的平均）
        :param timestep:
        :return model:包括周期，基本时间序列:
        """
        return ARIMAUtil.furtureCast(timestep, 15)


main()