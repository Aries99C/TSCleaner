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
            # 取数据整体状态稳定的段
            self.origin = self.origin[:int(len(self.origin) * 0.36)]
            # 填充缺失值
            self.origin.replace(0, np.nan, inplace=True)
            self.origin.interpolate(method='time', inplace=True)
            # 整合相同物理含义的序列
            # U3_HNC10CT11
            self.origin['U3_HNC10CT11'] = (self.origin['U3_HNC10CT111'] + self.origin['U3_HNC10CT112']
                                           + self.origin['U3_HNC10CT113']) / 3
            self.origin.drop(columns=['U3_HNC10CT111', 'U3_HNC10CT112', 'U3_HNC10CT113'], inplace=True)
            # U3_HNC10CT12
            self.origin['U3_HNC10CT12'] = (self.origin['U3_HNC10CT121'] + self.origin['U3_HNC10CT122']
                                           + self.origin['U3_HNC10CT123']) / 3
            self.origin.drop(columns=['U3_HNC10CT121', 'U3_HNC10CT122', 'U3_HNC10CT123'], inplace=True)
            # U3_HNC10CT13
            self.origin['U3_HNC10CT13'] = (self.origin['U3_HNC10CT131'] + self.origin['U3_HNC10CT132']
                                           + self.origin['U3_HNC10CT133']) / 3
            self.origin.drop(columns=['U3_HNC10CT131', 'U3_HNC10CT132', 'U3_HNC10CT133'], inplace=True)
            # U3_HNC10CT14
            self.origin['U3_HNC10CT14'] = (self.origin['U3_HNC10CT141'] + self.origin['U3_HNC10CT142']
                                           + self.origin['U3_HNC10CT143']) / 3
            self.origin.drop(columns=['U3_HNC10CT141', 'U3_HNC10CT142', 'U3_HNC10CT143'], inplace=True)
            # U3_HNC10CT15
            self.origin['U3_HNC10CT15'] = (self.origin['U3_HNC10CT151'] + self.origin['U3_HNC10CT152']) / 2
            self.origin.drop(columns=['U3_HNC10CT151', 'U3_HNC10CT152'], inplace=True)
            # U3_HNC10CT16
            self.origin['U3_HNC10CT16'] = (self.origin['U3_HNC10CT161'] + self.origin['U3_HNC10CT162']) / 2
            self.origin.drop(columns=['U3_HNC10CT161', 'U3_HNC10CT162'], inplace=True)
            # U3_HNC10CT17
            self.origin['U3_HNC10CT17'] = (self.origin['U3_HNC10CT171'] + self.origin['U3_HNC10CT172']
                                           + self.origin['U3_HNC10CT173'] + self.origin['U3_HNC10CT174']
                                           + self.origin['U3_HNC10CT175'] + self.origin['U3_HNC10CT176']) / 6
            self.origin.drop(columns=['U3_HNC10CT171', 'U3_HNC10CT172', 'U3_HNC10CT173',
                                      'U3_HNC10CT174', 'U3_HNC10CT175', 'U3_HNC10CT176'], inplace=True)
            # 去除不被使用的属性
            self.origin.drop(
                columns=['U3_HNA10CP107', 'U3_HNV10CP101', 'U3_HNV10CP102', 'U3_HNV10CP103',
                         'U3_HNV20CP101', 'U3_HNV20CP102', 'U3_HNV20CP103',
                         'U3_HNC10CY101', 'U3_HNC10CY102', 'U3_HNC10CY111', 'U3_HNC10CY112',
                         'U3_HNC10CT17'],
                inplace=True)

        self.len = len(self.origin)
        self.dim = len(self.origin.columns)

        # 预先配置干净数据
        self.clean = self.origin.copy(deep=True)
        # 人工清洗U3_HNV20CT111
        self.clean = self.clean[int(len(self.origin) * 0.33): int(len(self.origin) * 0.4)]

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

    print(fan.len)
    print(fan.dim)
    fan.clean.plot(subplots=True, figsize=(40, 40))
    plt.show()
