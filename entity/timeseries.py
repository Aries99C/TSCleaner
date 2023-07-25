import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from mindoptpy import *
from utils import root_path

warnings.filterwarnings('ignore')


class MTS(object):
    dataset = None

    len = None  # 时间戳长度
    dim = None  # 属性维度

    origin = None  # 观测值
    clean = None  # 正确值
    modified = None  # 修复值
    label = None  # 标记值

    isLabel = None  # 标记标签
    isModified = None  # 修复标签

    def __init__(self, dataset=None, size=None, ratio=None, file=None):
        filepath = None
        # 根据数据集名称获取文件地址，可自定义
        if dataset == 'fan':
            filepath = root_path() + '/data/fans/fan.csv'
        elif dataset == 'oil':
            filepath = root_path() + '/data/clean_oil/' + file
        elif dataset == 'SWaT':
            filepath = root_path() + '/data/SWaT.csv'
        if filepath is None:
            raise FileExistsError('dataset is Wrong!')

        # 数据集名称
        self.dataset = dataset

        # 对不同的数据集做预处理
        if dataset == 'fan':
            if size is None:
                size = 20000
            if ratio is None:
                ratio = 0.15

            # 1. 读取洁净数据

            # 读取原始值
            self.clean = pd.read_csv(filepath, sep=',', index_col='timestamp')
            # 配置索引
            self.clean.index = pd.DatetimeIndex(self.clean.index)
            # 取根据给定长度取数据段
            self.clean = self.clean[: size]

            # 2. 向观测值中注入错误

            # 清空标记
            self.label = self.clean.copy(deep=True)
            self.isLabel = self.clean.copy(deep=True)
            for col in self.isLabel.columns:
                self.isLabel[col] = False
                self.isLabel[:20] = True

            # 注入错误前计算时序数据的统计信息
            info = {}
            for col in self.clean.columns:
                info[col] = {}
                values = self.clean[col].values
                info[col]['min'] = np.mean(values)
                info[col]['max'] = np.max(values)
                info[col]['mean'] = np.mean(values)
                info[col]['std'] = np.std(values)

            # 拷贝干净数据集
            self.origin = self.clean.copy(deep=True)

            # 分配注入错误的比例和类型
            continuous_error_length = int(ratio / 3 * len(self.origin))
            single_error_length = int(ratio / 3 * len(self.origin))

            # 注入错误的属性
            cols_group = [
                [
                    'U3_DPU56A105',
                    'U3_HNC10AA101XQ01',
                    'U3_HNC10AN001XQ01',
                    'U3_HNC10CF101',
                    'U3_HNC10CF101_F',
                    'U3_HNC10CG101XQ01'
                ],
                [
                    'U3_HNC10CT111',
                    'U3_HNC10CT112',
                    'U3_HNC10CT113',
                    'U3_HNC10CT121',
                    'U3_HNC10CT122',
                    'U3_HNC10CT123',
                    'U3_HNC10CT131',
                    'U3_HNC10CT132',
                    'U3_HNC10CT133',
                    'U3_HNC10CT141',
                    'U3_HNC10CT142',
                    'U3_HNC10CT143',
                ],
                [
                    'U3_HNC10CT171',
                    'U3_HNC10CT172',
                    'U3_HNC10CT173',
                    'U3_HNC10CT174',
                    'U3_HNC10CT175',
                    'U3_HNC10CT176',
                ],
                [
                    'U3_HNC10CY101',
                    'U3_HNC10CY102',
                    'U3_HNC10CY111',
                    'U3_HNC10CY112',
                ],
                [
                    'U3_HNV10CT102',
                    'U3_HNV10CT103',
                    'U3_HNV10CT104',
                    'U3_HNV10CT111',
                    'U3_HNV20CT101',
                    'U3_HNV20CT102',
                    'U3_HNV20CT104',
                    'U3_PGB75CT101',
                    'U3_PGB75CT102',
                ]
            ]

            # 2.1 连续错误注入A

            # 属性采样
            error_cols = []
            for group in cols_group:
                error_cols.extend(random.sample(group, int((len(group) - 1) / 2)))

            # 随机注入位置
            insert_position = random.randrange(0, int(len(self.origin) / 10))

            # 注入连续错误
            for col in error_cols:
                for i in range(insert_position, insert_position + continuous_error_length):
                    self.origin[col].values[i] = info[col]['mean'] + 3 * info[col]['std'] + random.random() * 0.5 * \
                                                 info[col]['std']

            # 随机标注少量标记值
            random_label_index = np.random.randint(insert_position, insert_position + continuous_error_length,
                                                   size=int(continuous_error_length * 0.1))
            self.isLabel[random_label_index] = True

            # 2.2 单点错误注入

            # 属性采样
            error_cols = []
            for group in cols_group:
                error_cols.extend(random.sample(group, int((len(group) - 1) / 2)))

            # 随机注入位置
            insert_position = random.randrange(insert_position + continuous_error_length, int(len(self.origin) / 10 * 4))

            # 注入单点错误
            for col in error_cols:
                for i in range(insert_position, insert_position + single_error_length):
                    if i % 3 == 0:
                        self.origin[col].values[i] = info[col]['mean'] + pow(-1, i % 2) * (2 + random.random() * 0.5) * \
                                                     info[col]['std']
                        if random.random() <= 0.1:
                            self.isLabel[i] = True

            # 2.3 连续错误注入B

            # 属性采样
            error_cols = []
            for group in cols_group:
                error_cols.extend(random.sample(group, int((len(group) - 1) / 2)))

            # 随机注入位置
            insert_position = random.randrange(insert_position + single_error_length, int(len(self.origin) / 10 * 7))

            # 注入连续错误
            for col in error_cols:
                for i in range(insert_position, insert_position + continuous_error_length):
                    self.origin[col].values[i] = info[col]['mean'] + 3 * info[col][
                        'std'] + random.random() * 0.5 * \
                                                 info[col]['std']

            # 随机标注少量标记值
            random_label_index = np.random.randint(insert_position,
                                                   insert_position + continuous_error_length,
                                                   size=int(continuous_error_length * 0.1))
            self.isLabel[random_label_index] = True

            # 修复值重置为观测值
            self.modified = self.origin.copy(deep=True)
            self.isModified = self.origin.copy(deep=True)
            for col in self.isLabel.columns:
                self.isModified[col] = False

        if dataset == 'oil':
            # 读取原始值
            self.origin = pd.read_csv(filepath, sep=',', index_col=0)
            # 过滤不被使用的序列
            self.origin.drop(columns=['O2', 'N2', 'TOTALHYDROCARBON'], inplace=True)
            # 填充缺失值
            self.origin.fillna(method='ffill', inplace=True)
            self.origin.fillna(method='bfill', inplace=True)

            self.len = len(self.origin)
            self.dim = len(self.origin.columns)

            # 预先配置干净数据
            self.clean = self.origin.copy(deep=True)

            # 清空标记
            self.label = self.clean.copy(deep=True)
            self.isLabel = self.origin.copy(deep=True)
            for col in self.isLabel.columns:
                self.isLabel[col] = False
                self.isLabel[:20] = True

            # self.origin.plot(subplots=True, figsize=(8, 6))
            # plt.show()

            # 计算每个属性的最大最小值
            info = {}
            for col in self.clean.columns:
                info[col] = {}
                values = self.clean[col].values
                info[col]['min'] = 0.
                info[col]['max'] = np.max(values)
                info[col]['mean'] = np.mean(values)

            # 注入错误
            # 分配不同类型的错误
            continuous = ratio / 4
            single = ratio / 4
            # 注入连续错误
            error_len = int(self.len * continuous)
            cols = ['CH4', 'C2H4', 'C2H2', 'C2H6']
            col_list = random.sample(cols, 2)
            for col in col_list:
                for i in range(int(self.len * 0.2), int(self.len * 0.2) + error_len):
                    self.origin[col].values[i] += info[col]['mean'] / 2 + random.random() * info[col]['mean'] / 15
            # 随机标注少量标记值
            random_label_index = np.random.randint(int(self.len * 0.2), int(self.len * 0.2) + error_len,
                                                   size=int(error_len * 0.1))
            self.isLabel[random_label_index] = True
            # 注入连续错误
            error_len = int(self.len * continuous)
            cols = ['CH4', 'C2H4', 'C2H2', 'C2H6']
            col_list = random.sample(cols, 2)
            for col in col_list:
                for i in range(int(self.len * 0.4), int(self.len * 0.4) + error_len):
                    self.origin[col].values[i] += info[col]['mean'] / 2 + random.random() * info[col]['mean'] / 15
            # 随机标注少量标记值
            random_label_index = np.random.randint(int(self.len * 0.4), int(self.len * 0.4) + error_len,
                                                   size=int(error_len * 0.1))
            self.isLabel[random_label_index] = True
            # 注入连续错误
            error_len = int(self.len * continuous)
            cols = ['CH4', 'C2H4', 'C2H2', 'C2H6']
            col_list = random.sample(cols, 2)
            for col in col_list:
                for i in range(int(self.len * 0.6), int(self.len * 0.6) + error_len):
                    self.origin[col].values[i] += info[col]['mean'] / 2 + random.random() * info[col]['mean'] / 15
            # 随机标注少量标记值
            random_label_index = np.random.randint(int(self.len * 0.6), int(self.len * 0.6) + error_len,
                                                   size=int(error_len * 0.1))
            self.isLabel[random_label_index] = True

            # 注入单点错误
            error_len = int(self.len * single)
            cols = ['CH4', 'C2H4', 'C2H2', 'C2H6']
            for i in range(int(self.len * 0.8), int(self.len * 0.8) + error_len):
                if i % 3 == 0 and random.random() <= 0.1:
                    self.isLabel[i] = True
                    col_list = random.sample(cols, 2)
                    for col in col_list:
                        self.origin[col].values[i] += info[col]['mean'] / 3 + random.random() * info[col]['mean'] / 15

            # 修复值重置为观测值
            self.modified = self.origin.copy(deep=True)
            self.isModified = self.origin.copy(deep=True)
            for col in self.isLabel.columns:
                self.isModified[col] = False


if __name__ == '__main__':
    # 测试引风机数据集错误注入效果

    fan = MTS('fan')

    fan.origin.plot(subplots=True, figsize=(20, 30))
    plt.show()

    # print(fan.len)
    # print(fan.dim)

    # for file in os.listdir('../data/clean_oil/'):
    #     oil = MTS('oil', size=None, ratio=0.2, file=file)
    #     if oil.origin.isna().sum().sum() > 0:
    #         print('nan')

    # oil = MTS('oil', size=None, ratio=0.2, file='01M10000000039263.csv')

    # oil.origin.plot(subplots=True, figsize=(8, 6))
    # plt.show()

    # print(oil.len)
    # print(oil.dim)
