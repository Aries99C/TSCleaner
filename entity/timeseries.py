import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from matplotlib import rcParams
import lttb
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from utils import root_path
from algorithm import error


class MTS(object):

    len = None          # 时间戳长度
    dim = None          # 属性维度

    origin = None       # 观测值
    clean = None        # 正确值
    modified = None     # 修复值
    label = None        # 标记值

    isLabel = None      # 标记标签
    isModified = None   # 修复标签

    def __init__(self, dataset=None):
        filepath = None
        # 根据数据集名称获取文件地址，可自定义
        if dataset == 'fan':
            filepath = root_path() + '/data/fan.csv'

        if filepath is None:
            raise FileExistsError('dataset is Wrong!')

        # 读取原始值
        self.origin = pd.read_csv(filepath, sep=',', index_col='timestamp')
        # 配置索引
        self.origin.index = pd.DatetimeIndex(self.origin.index)

        # 对不同的数据集做预处理
        if dataset == 'fan':
            # 过滤模拟信号
            self.origin.drop(
                columns=[
                    'U3_HNV10CT111XH01',
                    'U3_HNV10CT111XH52',
                    'U3_HNV20CF001',
                    'U3_HNV20CF002',
                    'U3_HNV20CL001XH52',
                    'U3_HNV20CL001XH54',
                    'U3_HNV20CP101XH01',
                    'U3_HNV20CP101XH52',
                    'U3_HNV20CP102XH52',
                    'U3_HNV20CP103XH01',
                    'U3_HNV20CT111XH01',
                    'U3_HNV20CT111XH52'],
                inplace=True)
            # 取数据整体状态稳定的段
            self.origin = self.origin[:int(len(self.origin) * 0.36)]
            # 填充缺失值
            self.origin.replace(0, np.nan, inplace=True)
            self.origin.interpolate(method='time', inplace=True)
            # 分析数据
            self.origin = self.origin[66180:66180+1000]
            self.origin = self.origin[['U3_HNV10CT102', 'U3_HNV10CT103', 'U3_HNV10CT104']]
            # # 整合相同物理含义的序列
            # # U3_HNC10CT11
            # self.origin['U3_HNC10CT11'] = (self.origin['U3_HNC10CT111'] + self.origin['U3_HNC10CT112']
            #                                + self.origin['U3_HNC10CT113']) / 3
            # self.origin.drop(columns=['U3_HNC10CT111', 'U3_HNC10CT112', 'U3_HNC10CT113'], inplace=True)
            # # U3_HNC10CT12
            # self.origin['U3_HNC10CT12'] = (self.origin['U3_HNC10CT121'] + self.origin['U3_HNC10CT122']
            #                                + self.origin['U3_HNC10CT123']) / 3
            # self.origin.drop(columns=['U3_HNC10CT121', 'U3_HNC10CT122', 'U3_HNC10CT123'], inplace=True)
            # # U3_HNC10CT13
            # self.origin['U3_HNC10CT13'] = (self.origin['U3_HNC10CT131'] + self.origin['U3_HNC10CT132']
            #                                + self.origin['U3_HNC10CT133']) / 3
            # self.origin.drop(columns=['U3_HNC10CT131', 'U3_HNC10CT132', 'U3_HNC10CT133'], inplace=True)
            # # U3_HNC10CT14
            # self.origin['U3_HNC10CT14'] = (self.origin['U3_HNC10CT141'] + self.origin['U3_HNC10CT142']
            #                                + self.origin['U3_HNC10CT143']) / 3
            # self.origin.drop(columns=['U3_HNC10CT141', 'U3_HNC10CT142', 'U3_HNC10CT143'], inplace=True)
            # # U3_HNC10CT15
            # self.origin['U3_HNC10CT15'] = (self.origin['U3_HNC10CT151'] + self.origin['U3_HNC10CT152']) / 2
            # self.origin.drop(columns=['U3_HNC10CT151', 'U3_HNC10CT152'], inplace=True)
            # # U3_HNC10CT16
            # self.origin['U3_HNC10CT16'] = (self.origin['U3_HNC10CT161'] + self.origin['U3_HNC10CT162']) / 2
            # self.origin.drop(columns=['U3_HNC10CT161', 'U3_HNC10CT162'], inplace=True)
            # # U3_HNC10CT17
            # self.origin['U3_HNC10CT17'] = (self.origin['U3_HNC10CT171'] + self.origin['U3_HNC10CT172']
            #                                + self.origin['U3_HNC10CT173'] + self.origin['U3_HNC10CT174']
            #                                + self.origin['U3_HNC10CT175'] + self.origin['U3_HNC10CT176']) / 6
            # self.origin.drop(columns=['U3_HNC10CT171', 'U3_HNC10CT172', 'U3_HNC10CT173',
            #                           'U3_HNC10CT174', 'U3_HNC10CT175', 'U3_HNC10CT176'], inplace=True)
            # 去除不被使用的属性
            # self.origin.drop(
            #     columns=['U3_HNA10CP107', 'U3_HNV10CP101', 'U3_HNV10CP102', 'U3_HNV10CP103',
            #              'U3_HNV20CP101', 'U3_HNV20CP102', 'U3_HNV20CP103',
            #              'U3_HNC10CY101', 'U3_HNC10CY102', 'U3_HNC10CY111', 'U3_HNC10CY112',
            #              'U3_HNC10CT17'],
            #     inplace=True)

        self.len = len(self.origin)
        self.dim = len(self.origin.columns)

        # 预先配置干净数据
        self.clean = self.origin.copy(deep=True)
        # # 人工清洗U3_HNV20CT111
        # # 用速度约束清洗U3_HNV20CT111
        # values = self.clean['U3_HNV20CT111'].values
        # s = 0.1
        # for i in range(1, len(values)):
        #     if values[i] - values[i-1] > s:
        #         values[i] = values[i-1] + s
        #     elif values[i] - values[i-1] < -s:
        #         values[i] = values[i-1] - s
        # self.clean['U3_HNV20CT111'] = values
        # # 在U3_HNC10CT14注入一个连续错误并尝试用速度约束修复
        # values = self.clean['U3_HNC10CT14'].values
        # for i in range(int(len(self.origin) * 0.33) + 10000, int(len(self.origin) * 0.33) + 11000):
        #     values[i] = values[i] - 0.5 - random.random() * 0.05
        # self.clean['U3_HNC10CT14'] = values
        # # 速度约束清洗
        # s = 0.01
        # for i in range(1, len(values)):
        #     if values[i] - values[i - 1] > s:
        #         values[i] = values[i - 1] + s
        #     elif values[i] - values[i - 1] < -s:
        #         values[i] = values[i - 1] - s
        # self.clean['U3_HNC10CT14'] = values

        # 修复值和标记值随着错误的注入后续会发生变化
        self.modified = self.origin.copy(deep=True)
        self.label = self.origin.copy(deep=True)
        self.isLabel = self.origin.copy(deep=True)
        self.isModified = self.origin.copy(deep=True)
        for col in self.isLabel.columns:
            self.isLabel[col] = False
            self.isModified[col] = False


if __name__ == '__main__':
    fan = MTS('fan')

    print(fan.len)
    print(fan.dim)
    # fan.clean.plot(subplots=True, figsize=(6, 4),
    #                style=['+--', 'o--', '*--'], ms=3, linewidth=1, markerfacecolor='white')

    fig = plt.figure(figsize=(10, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # ['U3_HNV10CT102', 'U3_HNV10CT103', 'U3_HNV10CT104']
    x = fan.clean['U3_HNV10CT102'].values
    y = fan.clean['U3_HNV10CT103'].values
    z = fan.clean['U3_HNV10CT104'].values

    ax.scatter(x, y, z, c='g', label='正确值')
    ax.scatter(x[100:150]+0.07, y[100:150], z[100:150], c='r', label='错误值')
    ax.set_xlabel('U3_HNV10CT102')
    ax.set_ylabel('U3_HNV10CT103')
    ax.set_zlabel('U3_HNV10CT104')
    ax.set_xbound(28.6, 29.4)
    ax.set_ybound(36.0, 38.0)
    ax.set_zbound(31.8, 32.9)
    # 拟合三维直线
    data = np.concatenate((
        x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]
    ), axis=1)
    data_mean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - data_mean)
    linepts = vv[0] * np.mgrid[0.6:-0.6:2j][:, np.newaxis]
    linepts += data_mean
    ax.plot3D(*linepts.T)

    plt.show()
    # print(error(fan))
