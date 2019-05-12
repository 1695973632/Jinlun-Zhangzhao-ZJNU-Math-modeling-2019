'''
这个程序是用来获取数据并按照时间格式来处理数据的程序
同时，这个程序还会对数据来进行一些简单的分析，比如直接打出来看什么的
首先，我们通过直接观察法，可以看到每种货品的开始销售时间都是2018/4/17
这一天的销售量也绝对是这个货品接下来最大的销售量，而且后面的周期和样子，
都会由这个来决定为此，我们设计下面的数据提取并进行最简单的分析先

@Model and program author: Jinlun
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    dataUtil = DataUtil()
    dataUtil.getDataWithDate()

    for i in range(10,16):
        info = dataUtil.Data[dataUtil.goods[i]]
        x = np.array(info[0][1:])
        y = np.array(info[1][1:])

        plt.subplot(2,3,i-9)
        plt.xlabel('time')
        plt.ylabel('volume')
        plt.plot(y)

    plt.show()

class DataUtil:
    def __init__(self):
        self.monthday = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        self.data1 = pd.read_excel('./data/data_1.xlsx').values
        self.data2 = pd.read_excel('./data/data_2.xlsx').values

    def getDataWithDate(self):
        '''
        获取每种货号销量数据，并且按照时间片排好顺序,最后通过一次方程来进行补全
        :param year:
        :return:
        '''
        self.goods = []
        self.Data = {}
        for i in range(self.data1.shape[0]):
            date = self.data1[i, 1].split('/')
            if self.data1[i,0] not in self.Data.keys():
                # 分别用来存储X和Y,X从2018年1月17开始
                self.Data[self.data1[i,0]] = [[],[]]
                yy = int(date[0])
                mm = int(date[1])
                dd = int(date[2])
                day = (yy-2018)*365 + self.monthday[mm-1] + dd
                val = int(self.data1[i, 2])
                self.Data[self.data1[i, 0]][0].append(day)
                self.Data[self.data1[i, 0]][1].append(val)
                self.goods.append(self.data1[i,0])
            else:
                yy = int(date[0])
                mm = int(date[1])
                dd = int(date[2])
                day = (yy - 2018) * 365 + self.monthday[mm - 1] + dd
                val = int(self.data1[i, 2])
                self.Data[self.data1[i, 0]][0].append(day)
                self.Data[self.data1[i, 0]][1].append(val)
        #     数据补全
        for key in self.Data.keys():
            info = self.Data[key]
            xs = np.linspace(info[0][0], info[0][-1], (info[0][-1]-info[0][0]+1)).tolist()
            ys = np.linspace(info[0][0], info[0][-1], (info[0][-1]-info[0][0]+1)).tolist()
            for i in range(len(info[0])-1):
                if info[0][i+1] - info[0][i] > 1:
                    rate = (info[1][i+1] - info[1][i])/(info[0][i+1] - info[0][i])
                    for j in range(info[0][i]+1, info[0][i+1]+1):
                        ys[int(j-xs[0])] = info[1][i] + rate*(j-info[0][i])
                    pass
                else:
                    ys[int(info[0][i+1]-xs[0])] = info[1][i+1]
            self.Data[key][0] = xs
            self.Data[key][1] = ys


# main()