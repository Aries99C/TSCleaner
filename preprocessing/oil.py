import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mindoptpy import *
from utils import root_path

if __name__ == '__main__':
    dirs = os.listdir('../data/oil/')
    for file in dirs:
        data = pd.read_csv(root_path() + '/data/oil/' + file, sep=',', index_col=0)
        data.drop(columns=['O2', 'N2', 'TOTALHYDROCARBON'], inplace=True)

        # 填充缺失值
        for col in data.columns:
            if col != 'C2H2':
                data[col].replace(0, np.nan, inplace=True)
        data.interpolate(method='linear', inplace=True)

        # 计算每个属性的最大最小值
        info = {}
        for col in data.columns:
            info[col] = {}
            values = data[col].values
            info[col]['min'] = 0.
            info[col]['max'] = np.max(values)

        # 清洗
        for i in range(len(data)):
            model = MdoModel()
            try:
                # 目标函数最小化
                model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
                # 变量
                vars = []
                for col in data.columns:
                    vars.append(model.add_var(0, MdoModel.get_infinity(), 1. / info[col]['max'], None, 'u_' + col, False))
                    vars.append(model.add_var(0, MdoModel.get_infinity(), 1. / info[col]['max'], None, 'v_' + col, False))
                # 油色谱约束
                # 所有变量必须大于等于0.
                model.add_cons(
                    info[col]['min'] - data.iat[i, 0],
                    info[col]['max'] - data.iat[i, 0],
                    vars[0] - vars[1],
                    'min < H2 < max'
                )
                model.add_cons(
                    info[col]['min'] - data.iat[i, 1],
                    info[col]['max'] - data.iat[i, 1],
                    vars[2] - vars[3],
                    'min < CH4 < max'
                )
                model.add_cons(
                    info[col]['min'] - data.iat[i, 2],
                    info[col]['max'] - data.iat[i, 2],
                    vars[4] - vars[5],
                    'min < C2H4 < max'
                )
                model.add_cons(
                    info[col]['min'] - data.iat[i, 3],
                    info[col]['max'] - data.iat[i, 3],
                    vars[6] - vars[7],
                    'min < C2H2 < max'
                )
                model.add_cons(
                    info[col]['min'] - data.iat[i, 4],
                    info[col]['max'] - data.iat[i, 4],
                    vars[8] - vars[9],
                    'min < C2H6 < max'
                )
                model.add_cons(
                    info[col]['min'] - data.iat[i, 5],
                    info[col]['max'] - data.iat[i, 5],
                    vars[10] - vars[11],
                    'min < CO < max'
                )
                model.add_cons(
                    info[col]['min'] - data.iat[i, 6],
                    info[col]['max'] - data.iat[i, 6],
                    vars[12] - vars[13],
                    'min < CO2 < max'
                )
                # 三比值法
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -data.iat[i, 3] + 0.099 * data.iat[i, 2],
                    (vars[6] - vars[7]) - 0.099 * (vars[4] - vars[5]),
                    '0 < C2H2 / C2H4 < 0.1'
                )
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -data.iat[i, 1] + 0.99 * data.iat[i, 0],
                    (vars[2] - vars[3]) - 0.99 * (vars[0] - vars[1]),
                    'CH4 / H2 < 1'
                )
                model.add_cons(
                    -data.iat[i, 2] + 1.01 * data.iat[i, 4],
                    MdoModel.get_infinity(),
                    (vars[4] - vars[5]) - 1.01 * (vars[8] - vars[9]),
                    '1 <= C2H4 / C2H6'
                )
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -data.iat[i, 2] + 2.99 * data.iat[i, 4],
                    (vars[4] - vars[5]) - 2.99 * (vars[8] - vars[9]),
                    'C2H4 / C2H6 < 3'
                )
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -data.iat[i, 6] + 6.99 * data.iat[i, 5],
                    (vars[12] - vars[13]) - 6.99 * (vars[10] - vars[11]),
                    'CO2 / CO < 7'
                )
                # 求解
                model.solve_prob()
                # 修复
                for idx, col in enumerate(data.columns):
                    u = vars[idx * 2].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
                    v = vars[idx * 2 + 1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
                    data.iat[i, idx] += (u - v)
                    if u - v > 0 or u - v < 0:
                        print(" delta of {0} at {1}: {2}".format(col, data.index.values[idx],
                                                                 round(u - v, 2)))
            except MdoError as e:
                print("Received Mindopt exception.")
                print(" - Code          : {}".format(e.code))
                print(" - Reason        : {}".format(e.message))
            finally:
                model.free_mdl()

        # data['label'] = 0
        # for i in range(len(data)):
        #     r1 = data.iat[i, 3] / data.iat[i, 2]
        #     r2 = data.iat[i, 1] / data.iat[i, 0]
        #     r3 = data.iat[i, 2] / data.iat[i, 4]
        #     r4 = data.iat[i, 6] / data.iat[i, 5]
        #
        #     if r1 < 0.1 and 0.1 <= r2 < 1 and r3 < 1 and r4 > 7:
        #         data.iat[i, 7] = 1
        #     elif r1 < 0.1 and r2 >= 1 and r3 < 1:
        #         data.iat[i, 7] = 1
        #     elif r1 < 0.1 and r2 >= 1 and 1 <= r3 < 3:
        #         data.iat[i, 7] = 1
        #     elif r1 < 0.1 and r3 >= 3:
        #         data.iat[i, 7] = 1
        #     elif r1 < 0.1 and r2 < 0.1 and r3 < 1:
        #         data.iat[i, 7] = 1
        #     elif r1 >= 0.1:
        #         data.iat[i, 7] = 1

        data.plot(subplots=True, figsize=(8, 6))
        plt.title(file)
        plt.show()

        # data.to_csv('../data/clean_oil/' + file, index_label='timestamp')
