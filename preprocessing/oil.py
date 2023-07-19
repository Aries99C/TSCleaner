import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mindoptpy import *
from utils import root_path

if __name__ == '__main__':
    data = pd.read_csv(root_path() + '/data/oil/01M10000000038959.csv', sep=',', index_col=0)
    data.drop(columns=['O2', 'N2', 'TOTALHYDROCARBON'], inplace=True)

    # 填充缺失值
    for col in data.columns:
        if col != 'C2H2':
            data[col].replace(0, np.nan, inplace=True)
    data.interpolate(method='linear', inplace=True)

    # 计算每个属性的最大最小值
    info = {}
    for col in data.columns:
        values = data[col].values
        info[col]['min'] = max(0., np.min(values))
        info[col]['max'] = np.min(values)

    # 清洗
    for i in range(len(data)):
        model = MdoModel()
        try:
            # 目标函数最小化
            model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
            # 变量
            vars = []
            for col in data.columns:
                vars.append(model.add_var(0, MdoModel.get_infinity(), 1., None, 'u_' + col, False))
                vars.append(model.add_var(0, MdoModel.get_infinity(), 1., None, 'v_' + col, False))
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
            # TODO
        except MdoError as e:
            print("Received Mindopt exception.")
            # print(" - Code          : {}".format(e.code))
            # print(" - Reason        : {}".format(e.message))
        finally:
            model.free_mdl()

    data.plot(subplots=True, figsize=(8, 6))
    plt.plot()
