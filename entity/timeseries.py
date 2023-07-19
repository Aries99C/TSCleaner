import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from mindoptpy import *
from utils import root_path

warnings.filterwarnings('ignore')


class MTS(object):
    len = None  # 时间戳长度
    dim = None  # 属性维度

    origin = None  # 观测值
    clean = None  # 正确值
    modified = None  # 修复值
    label = None  # 标记值

    isLabel = None  # 标记标签
    isModified = None  # 修复标签

    def __init__(self, dataset=None, size=None, ratio=None):
        filepath = None
        # 根据数据集名称获取文件地址，可自定义
        if dataset == 'fan':
            filepath = root_path() + '/data/fan.csv'
        if dataset == 'oil':
            filepath = root_path() + '/data/oil/01M10000000038959.csv'
        if filepath is None:
            raise FileExistsError('dataset is Wrong!')

        # 对不同的数据集做预处理
        if dataset == 'fan':
            if size is None:
                size = 20000

            # 读取原始值
            self.origin = pd.read_csv(filepath, sep=',', index_col='timestamp')
            # 配置索引
            self.origin.index = pd.DatetimeIndex(self.origin.index)
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
            # 过滤没有规则关联的信号
            self.origin.drop(
                columns=[
                    'U3_HNC10CY101',
                    'U3_HNC10CY102',
                    'U3_HNC10CY111',
                    'U3_HNC10CY112',
                    'U3_HNV10CP101',
                    'U3_HNV10CP102'],
                inplace=True)
            # 取数据整体状态稳定的段
            self.origin = self.origin[int(len(self.origin) * 0.1): int(len(self.origin) * 0.1) + size]
            # 填充缺失值
            self.origin.replace(0, np.nan, inplace=True)
            self.origin.interpolate(method='time', inplace=True)

            self.len = len(self.origin)
            self.dim = len(self.origin.columns)

            # 预先配置干净数据
            self.clean = self.origin.copy(deep=True)

            # 提取时序数据信息
            # info = {}
            # for col in self.clean.columns:
            #     info[col] = {}
            #     info[col]['min'] = np.min(self.clean[col].values)
            #     info[col]['max'] = np.min(self.clean[col].values)
            #     for another_col in self.clean.columns:
            #         info[col][another_col] = {}
            #         info[col][another_col]['min'] = np.mean(
            #             self.clean[col].values - self.clean[another_col].values) - 2 * np.std(
            #             self.clean[col].values - self.clean[another_col].values)
            #         info[col][another_col]['max'] = np.mean(
            #             self.clean[col].values - self.clean[another_col].values) + 2 * np.std(
            #             self.clean[col].values - self.clean[another_col].values)

            # # 清洗U3_HNV20CT104中的小段错误
            # for i in range(1, len(self.clean)):
            #     model = MdoModel()
            #     try:
            #         # 目标函数最小化
            #         model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
            #         # 变量
            #         vars = []
            #         for col in info.keys():
            #             vars.append(model.add_var(0, MdoModel.get_infinity(), 1., None, 'u_' + col, False))
            #             vars.append(model.add_var(0, MdoModel.get_infinity(), 1., None, 'v_' + col, False))
            #         # 引风机约束
            #         model.add_cons(
            #             info['U3_HNV20CT104']['U3_HNV20CT101']['min'] - self.clean.iat[i, 40] +
            #             self.clean.iat[i, 38],
            #             info['U3_HNV20CT104']['U3_HNV20CT101']['max'] - self.clean.iat[i, 40] +
            #             self.clean.iat[i, 38],
            #             (vars[80] - vars[81]) - (vars[76] - vars[77]),
            #             'U3_HNV20CT104 - U3_HNV20CT101')
            #         model.add_cons(
            #             info['U3_HNV20CT104']['U3_HNV20CT102']['min'] - self.clean.iat[i, 40] +
            #             self.clean.iat[i, 39],
            #             info['U3_HNV20CT104']['U3_HNV20CT102']['max'] - self.clean.iat[i, 40] +
            #             self.clean.iat[i, 39],
            #             (vars[80] - vars[81]) - (vars[78] - vars[79]),
            #             'U3_HNV20CT104 - U3_HNV20CT102')
            #         model.solve_prob()
            #         # 修复
            #         for idx, col in enumerate(info.keys()):
            #             u = vars[idx * 2].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
            #             v = vars[idx * 2 + 1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
            #             self.clean.iat[i, idx] += (u - v)
            #     except MdoError as e:
            #         print("Received Mindopt exception.")
            #         # print(" - Code          : {}".format(e.code))
            #         # print(" - Reason        : {}".format(e.message))
            #     finally:
            #         model.free_mdl()

            # 清空标记
            self.label = self.clean.copy(deep=True)
            self.isLabel = self.origin.copy(deep=True)
            for col in self.isLabel.columns:
                self.isLabel[col] = False

                self.isLabel[:50] = True

            # 注入错误数据
            # 注入5%的连续错误
            for col in [
                'U3_HNC10CT111', 'U3_HNC10CT121', 'U3_HNC10CT131', 'U3_HNC10CT141'
            ]:
                values = self.origin[col].values
                error_len = int(self.len * 0.05)
                # 注入
                for i in range(int(self.len * 0.1), int(self.len * 0.1) + error_len):
                    values[i] += -3. + random.random() * 0.05
                # 用注入的错误数据覆盖观测值
                self.origin[col] = values
                # 随机标注少量标记值
                random_label_index = np.random.randint(int(self.len * 0.1), int(self.len * 0.1) + error_len,
                                                       size=int(error_len * 0.1))
                self.isLabel[random_label_index] = True

            # # 注入5%的连续错误
            # for col in [
            #     'U3_HNC10CT113', 'U3_HNC10CT123', 'U3_HNC10CT133', 'U3_HNC10CT143'
            # ]:
            #     values = self.origin[col].values
            #     error_len = int(20000 * 0.05)
            #     # 注入
            #     for i in range(2378, 2378+error_len):
            #         values[i] += -3. + random.random() * 0.05
            #     # 用注入的错误数据覆盖观测值
            #     self.origin[col] = values
            #     # 随机标注少量标记值
            #     random_label_index = np.random.randint(2378, 2378 + error_len, size=int(error_len * 0.1))
            #     self.isLabel[random_label_index] = True

            # # 注入5%的连续错误
            # for col in [
            #     'U3_HNC10CT172', 'U3_HNC10CT173', 'U3_HNV20CT104'
            # ]:
            #     values = self.origin[col].values
            #     error_len = int(20000 * 0.05)
            #     # 注入
            #     for i in range(4269, 4269 + error_len):
            #         values[i] += -3. + random.random() * 0.05
            #         if col == 'U3_HNV20CT104':
            #             values[i] += 2.
            #     # 用注入的错误数据覆盖观测值
            #     self.origin[col] = values
            #     # 随机标注少量标记值
            #     random_label_index = np.random.randint(4269, 4269 + error_len, size=int(error_len * 0.1))
            #     self.isLabel[random_label_index] = True

            # 注入5%的小错误
            cols = [
                ['U3_HNC10CT111', 'U3_HNC10CT112', 'U3_HNC10CT113'],
                ['U3_HNC10CT121', 'U3_HNC10CT122', 'U3_HNC10CT123'],
                ['U3_HNC10CT131', 'U3_HNC10CT132', 'U3_HNC10CT133'],
                ['U3_HNC10CT141', 'U3_HNC10CT142', 'U3_HNC10CT143'],
                ['U3_HNC10CT171', 'U3_HNC10CT172', 'U3_HNC10CT173', 'U3_HNC10CT174', 'U3_HNC10CT175', 'U3_HNC10CT176'],
            ]
            error_len = int(self.len * 0.05)
            for i in range(int(self.len * 0.4), int(self.len * 0.4) + error_len):
                for group in cols:
                    col_list = random.sample(group, int((len(group) - 1) / 2))
                    for col in col_list:
                        self.origin[col].values[i] += -5. + random.random() * 1.5

            # # 注入5%的小错误
            # cols = [
            #     ['U3_HNC10CT111', 'U3_HNC10CT112', 'U3_HNC10CT113'],
            #     ['U3_HNC10CT121', 'U3_HNC10CT122', 'U3_HNC10CT123'],
            #     ['U3_HNC10CT131', 'U3_HNC10CT132', 'U3_HNC10CT133'],
            #     ['U3_HNC10CT141', 'U3_HNC10CT142', 'U3_HNC10CT143'],
            #     ['U3_HNC10CT171', 'U3_HNC10CT172', 'U3_HNC10CT173', 'U3_HNC10CT174', 'U3_HNC10CT175',
            #      'U3_HNC10CT176'],
            # ]
            # error_len = int(20000 * 0.05)
            # for i in range(9918, 9918 + error_len):
            #     for group in cols:
            #         col_list = random.sample(group, int((len(group) - 1) / 2))
            #         for col in col_list:
            #             self.origin[col].values[i] += -3. + random.random() * 1.5

            # # 注入5%的小错误
            # cols = [
            #     ['U3_HNC10CT111', 'U3_HNC10CT112', 'U3_HNC10CT113'],
            #     ['U3_HNC10CT121', 'U3_HNC10CT122', 'U3_HNC10CT123'],
            #     ['U3_HNC10CT131', 'U3_HNC10CT132', 'U3_HNC10CT133'],
            #     ['U3_HNC10CT141', 'U3_HNC10CT142', 'U3_HNC10CT143'],
            #     ['U3_HNC10CT171', 'U3_HNC10CT172', 'U3_HNC10CT173', 'U3_HNC10CT174', 'U3_HNC10CT175',
            #      'U3_HNC10CT176'],
            # ]
            # error_len = int(20000 * 0.05)
            # for i in range(11392, 11392 + error_len):
            #     for group in cols:
            #         col_list = random.sample(group, int((len(group) - 1) / 2))
            #         for col in col_list:
            #             self.origin[col].values[i] += -4. + random.random() * 2.0

            # # 注入5%的小错误
            # cols = [
            #     ['U3_HNC10CT111', 'U3_HNC10CT112', 'U3_HNC10CT113'],
            #     ['U3_HNC10CT121', 'U3_HNC10CT122', 'U3_HNC10CT123'],
            #     ['U3_HNC10CT131', 'U3_HNC10CT132', 'U3_HNC10CT133'],
            #     ['U3_HNC10CT141', 'U3_HNC10CT142', 'U3_HNC10CT143'],
            #     ['U3_HNC10CT171', 'U3_HNC10CT172', 'U3_HNC10CT173', 'U3_HNC10CT174', 'U3_HNC10CT175',
            #      'U3_HNC10CT176'],
            # ]
            # error_len = int(20000 * 0.05)
            # for group in cols:
            #     col_list = random.sample(group, int((len(group) - 1) / 2))
            #     for col in col_list:
            #         values = self.origin[col].values
            #         for i in range(13238, 13238 + error_len):
            #             values[i] += -3. + random.random() * 0.5

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
            for col in self.origin.columns:
                if col != 'C2H2':
                    self.origin[col].replace(0, np.nan, inplace=True)
            self.origin.interpolate(method='linear', inplace=True)

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

            # 注入错误

            # 修复值重置为观测值
            self.modified = self.origin.copy(deep=True)
            self.isModified = self.origin.copy(deep=True)
            for col in self.isLabel.columns:
                self.isModified[col] = False


if __name__ == '__main__':
    # fan = MTS('fan')
    #
    # fan.origin.plot(subplots=True, figsize=(20, 30))
    # plt.show()
    #
    # print(fan.len)
    # print(fan.dim)

    oil = MTS('oil')

    oil.origin.plot(subplots=True, figsize=(8, 6))
    plt.show()

    print(oil.len)
    print(oil.dim)

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
