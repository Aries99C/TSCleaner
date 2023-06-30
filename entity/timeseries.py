import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import rcParams

from utils import root_path


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
            # 过滤冗余且高质量的属性值

            # 取数据整体状态稳定的段
            self.origin = self.origin[:int(len(self.origin) * 0.375)]
            # 填充缺失值
            self.origin.replace(0, np.nan, inplace=True)
            self.origin.interpolate(method='time', inplace=True)

        self.len = len(self.origin)
        self.dim = len(self.origin.columns)

        self.clean = self.origin.copy(deep=True)

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
    # for period in [30, 90, 180, 360]:
    #     rd = sm.tsa.seasonal_decompose(fan.origin['U3_DPU56A105'].values, period=period)
    #     rcParams['figure.figsize'] = 30, 15
    #     fig = rd.plot()
    #     plt.show()

    # print(fan.len)
    # print(fan.dim)
    # fan.clean.plot(subplots=True, figsize=(40, 40))
    # plt.show()
