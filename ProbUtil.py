'''
第二题和第三题，计算不同季度的置信区间之类的信息
最终结果配合第一题的模型来实现最后一题的策略
@Model and program author: Jinlun
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def getHist(hist, range):
    '''
    分别计算置信度为0.95和0.90的置信区间
    :param hist:
    :return: tuple
    '''
    n, bins = [], []
    for rect in hist.patches:
        ((x0,y0),(x1,y1)) = rect.get_bbox().get_points()
        if int(x1) > range:
            break
        n.append(y1-y0)
        bins.append(int(x1))
    return np.array(n), np.array(bins)

def calcConfidence(hist, bins):
    prob = hist/np.sum(hist)
    summer = 0
    conf_5 = []
    conf_10 = []
    if prob[0] > 0.025:
        for i in range(len(hist)):
            summer += prob[i]
            if summer > 0.95:
                conf_5 = (0, bins[i])
                break
        # print(summer)
        summer = 0
    if prob[0] > 0.05:
        for i in range(len(hist)):
            summer += prob[i]
            if summer > 0.9:
                conf_10 = (0, bins[i])
                break
        summer = 0
    else:
        l1,l2,r1,r2 = 0,0,bins[-1],bins[-1]
        k1,k2,k3,k4 = 0,0,0,0
        for i in range(len(hist)):
            summer += prob[i]
            if summer > 0.025 and k1 == 0:
                l1 = bins[i]
                k1 = 1
            if summer > 0.05 and k2 == 0:
                l2 = bins[i]
                k2 = 1
            if summer > 0.95 and k3 == 0:
                r2 = bins[i]
                k3 = 1
            if summer > 0.975 and k4 == 0:
                r1 = bins[i]
                k4 = 1
        conf_5 = (l1,r1)
        conf_10 = (l2,r2)
    return conf_5, conf_10

def getInfoFromData(data, season):
    delay_vol = data['延期比']
    new_in_all = data['总下单量']
    new_in_before = data['上新前下单总量']
    delay_vol = 100 * delay_vol.dropna(axis=0, how='all')
    new_in_all.dropna(axis=0, how='all')
    new_in_before.dropna(axis=0, how='all')
    delay_hist = delay_vol.hist(bins=100)
    new_in_all_hist = new_in_all.hist(bins=2000)

    histval, bins = getHist(delay_hist, np.max(delay_vol.values))
    conf_5, conf_10 = calcConfidence(histval, bins)
    print(season + " 延期比置信度为95%的置信区间:", conf_5)
    print(season + " 延期比置信度为90%的置信区间:",conf_10)

    histval, bins = getHist(new_in_all_hist, np.max(new_in_all.values))
    conf_5, conf_10 = calcConfidence(histval, bins)
    print(season + " 上新量置信度为95%的置信区间",conf_5)
    print(season + " 上新量置信度为90%的置信区间",conf_10)
    # 我们通过对总的情况分析之后得出我们可以把这个整体看作卡方分布
    plt.subplot(121)
    plt.title(season + " delay ratio Distribution")
    plt.hist(delay_vol, bins=100, density=True)
    plt.xlabel('delay ratio (%)')
    plt.ylabel('Frequency')
    plt.subplot(122)
    plt.title(season + " New quantity Distribution")
    plt.hist(new_in_before, bins=100, density=True)
    plt.xlabel('New quantity')
    plt.ylabel('Frequency')
    plt.show()

def AnaFourSeason(data):
    dataSpring = data.loc[27:151]
    dataSummer = data.loc[152:299]
    dataFall = data.loc[302:]
    dataWinter = data.loc[0:27]
    getInfoFromData(dataSpring, "Spring")
    getInfoFromData(dataSummer, "Summer")
    getInfoFromData(dataFall, "fall")
    getInfoFromData(dataWinter, "winter")


def AnalogFile(filename):
    data = pd.read_excel(filename)
    getInfoFromData(data,"")
    AnaFourSeason(data)


if __name__ == '__main__':
    AnalogFile('./data/data_2.xlsx')

