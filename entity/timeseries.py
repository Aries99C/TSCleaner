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
                    'U3_HNV20CT111XH52',
                    'U3_HNV10CP103'],
                inplace=True)
            # 取数据整体状态稳定的段
            self.origin = self.origin[int(len(self.origin) * 0.1): int(len(self.origin) * 0.1)+20000] # 长度20k
            # 填充缺失值
            self.origin.replace(0, np.nan, inplace=True)
            self.origin.interpolate(method='time', inplace=True)

        self.len = len(self.origin)
        self.dim = len(self.origin.columns)

        # 预先配置干净数据
        self.clean = self.origin.copy(deep=True)

        # 注入错误数据
        # 注入5%
        for col in [
            'U3_HNC10CT111', 'U3_HNC10CT121', 'U3_HNC10CT131', 'U3_HNC10CT141'
        ]:
            values = self.origin[col].values
            error_len = int(20000 * 0.05)
            # 注入
            for i in range(100, 100+error_len):
                values[i] += -3. + random.random() * 0.05
            # 用注入的错误数据覆盖观测值
            self.origin[col] = values

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

    fan.origin.plot(subplots=True, figsize=(20, 30))
    plt.show()

    print(fan.len)
    print(fan.dim)
    #
    # fig = plt.figure(1, figsize=(10, 6))
    # ax = Axes3D(fig, auto_add_to_figure=False)
    # fig.add_axes(ax)
    # # ['U3_HNV10CT102', 'U3_HNV10CT103', 'U3_HNV10CT104']
    # x = fan.clean['U3_HNV10CT102'].values
    # y = fan.clean['U3_HNV10CT103'].values
    # z = fan.clean['U3_HNV10CT104'].values
    #
    # ax.scatter(x[:100], y[:100], z[:100], c='g')
    # ax.scatter(x[120:], y[120:], z[120:], c='g')
    # ax.scatter(x[100:120]+0.1, y[100:120], z[100:120], c='r', label='错误值')
    # ax.scatter(np.add(x[100:120]+0.02, np.array([random.random()*0.02 for i in range(20)])), y[100:120], z[100:120],
    #            c='b', label='修复值')
    # ax.scatter(x[100:120], y[100:120], z[100:120], c='brown', label='正确值')
    # ax.set_xlabel('U3_HNV10CT102')
    # ax.set_ylabel('U3_HNV10CT103')
    # ax.set_zlabel('U3_HNV10CT104')
    # ax.set_xbound(28.6, 29.4)
    # ax.set_ybound(36.0, 38.0)
    # ax.set_zbound(31.8, 32.9)
    # # 拟合三维直线
    # data = np.concatenate((
    #     x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]
    # ), axis=1)
    # data_mean = data.mean(axis=0)
    # uu, dd, vv = np.linalg.svd(data - data_mean)
    # linepts = vv[0] * np.mgrid[0.6:-0.6:2j][:, np.newaxis]
    # linepts += data_mean
    # ax.plot3D(*linepts.T)
    #
    # fig2 = plt.figure(figsize=(6, 4))
    # t = list(fan.clean.index)[80:140]
    #
    # ax1 = plt.subplot(311)
    # x_e = x.copy()
    # x_e[100:120] += 0.1
    # ax1.plot(t, x_e[80:140], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='darkorange', label='U3_HNV10CT102')
    # plt.ylim(28.8, 29.1)
    # plt.tick_params('x', labelbottom=False)
    #
    # ax2 = plt.subplot(312, sharex=ax1)
    # ax2.plot(t, y[80:140], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='palegreen', label='U3_HNV10CT103')
    # plt.ylim(36.5, 37.2)
    # plt.tick_params('x', labelbottom=False)
    #
    # ax3 = plt.subplot(313, sharex=ax1)
    # ax3.plot(t, z[80:140], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='skyblue', label='U3_HNV10CT104')
    # plt.ylim(32.2, 32.9)
    # plt.xticks(rotation=45)
    # plt.tick_params('x', labelsize=8)
    #
    # plt.tight_layout()
    # plt.show()
    #
    # fig3 = plt.figure(figsize=(5, 5))
    # ct102 = fan.clean['U3_HNV10CT102'].values[80:140]
    # ct103 = fan.clean['U3_HNV10CT103'].values[80:140]
    #
    # delta = ct103 - ct102
    # mean = np.mean(delta)
    # std = np.std(delta)
    #
    # for i in range(len(ct102)):
    #     if not (ct102[i] + mean - 2 * std <= ct103[i] <= ct102[i] + mean + 2 * std):
    #         print(i)
    #
    # plt.scatter(ct102[:59], ct103[:59], marker='o', s=7, c='white', edgecolors='g')
    # plt.scatter(ct102[60:], ct103[60:], marker='o', s=7, c='white', edgecolors='g')
    # plt.scatter(ct102[59], ct103[59], marker='o', s=7, c='white', edgecolors='r')
    #
    # x = np.linspace(28.9, 28.96, 100)
    # y1 = x + mean + 2 * std
    # y2 = x + mean - 2 * std
    #
    # plt.plot(x, y1, '--', c='blue')
    # plt.plot(x, y2, '--', c='blue')
    # plt.fill_between(x, y1, y2, where=y1>y2, color='grey', alpha=0.5)
    #
    # plt.grid(True, linestyle='-.')
    # plt.xlim(28.9, 28.96)
    # plt.ylim(36.86, 36.98)
    # plt.xlabel('U3_HNV10CT102')
    # plt.ylabel('U3_HNV10CT103')
    # plt.show()
