import time
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error
from mindoptpy import *


class Constraint(Cleaner):

    def clean(self):
        info = {}
        for col in self.dataset.clean.columns:
            info[col] = {}
            info[col]['min'] = np.min(self.dataset.clean[col].values)
            info[col]['max'] = np.min(self.dataset.clean[col].values)
            for another_col in self.dataset.clean.columns:
                info[col][another_col] = {}
                info[col][another_col]['min'] = np.mean(
                    self.dataset.clean[col].values - self.dataset.clean[another_col].values) - 2 * np.std(
                    self.dataset.clean[col].values - self.dataset.clean[another_col].values)
                info[col][another_col]['max'] = np.mean(
                    self.dataset.clean[col].values - self.dataset.clean[another_col].values) + 2 * np.std(
                    self.dataset.clean[col].values - self.dataset.clean[another_col].values)

        start = time.perf_counter()

        for i in range(1, len(self.dataset.origin)):
            model = MdoModel()
            try:
                # 目标函数最小化
                model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
                # 变量
                vars = []
                for col in info.keys():
                    vars.append(model.add_var(0, MdoModel.get_infinity(), 1., None, 'u_' + col, False))
                    vars.append(model.add_var(0, MdoModel.get_infinity(), 1., None, 'v_' + col, False))
                # 约束
                model.add_cons(
                    info['U3_HNC10CT111']['U3_HNC10CT112']['min'] - self.dataset.modified.iat[i, 9] +
                    self.dataset.modified.iat[i, 10],
                    info['U3_HNC10CT111']['U3_HNC10CT112']['max'] - self.dataset.modified.iat[i, 9] +
                    self.dataset.modified.iat[i, 10],
                    (vars[18] - vars[19]) - (vars[20] - vars[21]),
                    'U3_HNC10CT111 - U3_HNC10CT112')
                model.add_cons(
                    info['U3_HNC10CT111']['U3_HNC10CT113']['min'] - self.dataset.modified.iat[i, 9] +
                    self.dataset.modified.iat[i, 11],
                    info['U3_HNC10CT111']['U3_HNC10CT113']['max'] - self.dataset.modified.iat[i, 9] +
                    self.dataset.modified.iat[i, 11],
                    (vars[18] - vars[19]) - (vars[22] - vars[23]),
                    'U3_HNC10CT111 - U3_HNC10CT113')
                model.add_cons(
                    info['U3_HNC10CT121']['U3_HNC10CT122']['min'] - self.dataset.modified.iat[i, 12] +
                    self.dataset.modified.iat[i, 13],
                    info['U3_HNC10CT121']['U3_HNC10CT122']['max'] - self.dataset.modified.iat[i, 12] +
                    self.dataset.modified.iat[i, 13],
                    (vars[24] - vars[25]) - (vars[26] - vars[27]),
                    'U3_HNC10CT121 - U3_HNC10CT122')
                model.add_cons(
                    info['U3_HNC10CT121']['U3_HNC10CT123']['min'] - self.dataset.modified.iat[i, 12] +
                    self.dataset.modified.iat[i, 14],
                    info['U3_HNC10CT121']['U3_HNC10CT123']['max'] - self.dataset.modified.iat[i, 12] +
                    self.dataset.modified.iat[i, 14],
                    (vars[24] - vars[25]) - (vars[28] - vars[29]),
                    'U3_HNC10CT121 - U3_HNC10CT123')
                model.add_cons(
                    info['U3_HNC10CT131']['U3_HNC10CT132']['min'] - self.dataset.modified.iat[i, 15] +
                    self.dataset.modified.iat[i, 16],
                    info['U3_HNC10CT131']['U3_HNC10CT132']['max'] - self.dataset.modified.iat[i, 15] +
                    self.dataset.modified.iat[i, 16],
                    (vars[30] - vars[31]) - (vars[32] - vars[33]),
                    'U3_HNC10CT131 - U3_HNC10CT132')
                model.add_cons(
                    info['U3_HNC10CT131']['U3_HNC10CT133']['min'] - self.dataset.modified.iat[i, 15] +
                    self.dataset.modified.iat[i, 17],
                    info['U3_HNC10CT131']['U3_HNC10CT133']['max'] - self.dataset.modified.iat[i, 15] +
                    self.dataset.modified.iat[i, 17],
                    (vars[30] - vars[31]) - (vars[34] - vars[35]),
                    'U3_HNC10CT131 - U3_HNC10CT133')
                # 求解
                model.solve_prob()
                # model.display_results()
                # 修复
                for idx, col in enumerate(info.keys()):
                    u = vars[idx * 2].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
                    v = vars[idx * 2 + 1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
                    self.dataset.modified.iat[i, idx] += (u - v)
                    if u - v > 0 or u - v < 0:
                        print(" delta of {0} at {1}: {2}".format(col, self.dataset.modified.index.values[idx],
                                                                 round(u - v, 2)))
            except MdoError as e:
                print("Received Mindopt exception.")
                # print(" - Code          : {}".format(e.code))
                # print(" - Reason        : {}".format(e.message))
            finally:
                model.free_mdl()

        end = time.perf_counter()
        print(end - start, 'ms')

        return info


if __name__ == '__main__':
    fan = MTS('fan')
    # 测试SCREEN修复效果
    # 修复前
    before_fix = error(fan)

    # fan.origin.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    # 修复后
    cleaner = Constraint(fan)
    info = cleaner.clean()

    # fan.modified.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    after_fix = error(fan)

    print(before_fix, after_fix)
