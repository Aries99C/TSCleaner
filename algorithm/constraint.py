import time
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error
from mindoptpy import *


class Constraint(Cleaner):

    def clean(self):
        # 先重新拷贝观测值
        self.dataset.modified = self.dataset.origin.copy(deep=True)

        # info = {}
        # for col in self.dataset.clean.columns:
        #     info[col] = {}
        #     info[col]['min'] = np.min(self.dataset.clean[col].values)
        #     info[col]['max'] = np.min(self.dataset.clean[col].values)
        #     for another_col in self.dataset.clean.columns:
        #         info[col][another_col] = {}
        #         info[col][another_col]['min'] = np.mean(
        #             self.dataset.clean[col].values - self.dataset.clean[another_col].values) - 2 * np.std(
        #             self.dataset.clean[col].values - self.dataset.clean[another_col].values)
        #         info[col][another_col]['max'] = np.mean(
        #             self.dataset.clean[col].values - self.dataset.clean[another_col].values) + 2 * np.std(
        #             self.dataset.clean[col].values - self.dataset.clean[another_col].values)
        #
        # t = 0.
        #
        # for i in range(1, len(self.dataset.origin)):
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
        #         # U3_HNC10CT11组
        #         model.add_cons(
        #             info['U3_HNC10CT111']['U3_HNC10CT112']['min'] - self.dataset.modified.iat[i, 9] +
        #             self.dataset.modified.iat[i, 10],
        #             info['U3_HNC10CT111']['U3_HNC10CT112']['max'] - self.dataset.modified.iat[i, 9] +
        #             self.dataset.modified.iat[i, 10],
        #             (vars[18] - vars[19]) - (vars[20] - vars[21]),
        #             'U3_HNC10CT111 - U3_HNC10CT112')
        #         model.add_cons(
        #             info['U3_HNC10CT111']['U3_HNC10CT113']['min'] - self.dataset.modified.iat[i, 9] +
        #             self.dataset.modified.iat[i, 11],
        #             info['U3_HNC10CT111']['U3_HNC10CT113']['max'] - self.dataset.modified.iat[i, 9] +
        #             self.dataset.modified.iat[i, 11],
        #             (vars[18] - vars[19]) - (vars[22] - vars[23]),
        #             'U3_HNC10CT111 - U3_HNC10CT113')
        #         # U3_HNC10CT12组
        #         model.add_cons(
        #             info['U3_HNC10CT121']['U3_HNC10CT122']['min'] - self.dataset.modified.iat[i, 12] +
        #             self.dataset.modified.iat[i, 13],
        #             info['U3_HNC10CT121']['U3_HNC10CT122']['max'] - self.dataset.modified.iat[i, 12] +
        #             self.dataset.modified.iat[i, 13],
        #             (vars[24] - vars[25]) - (vars[26] - vars[27]),
        #             'U3_HNC10CT121 - U3_HNC10CT122')
        #         model.add_cons(
        #             info['U3_HNC10CT121']['U3_HNC10CT123']['min'] - self.dataset.modified.iat[i, 12] +
        #             self.dataset.modified.iat[i, 14],
        #             info['U3_HNC10CT121']['U3_HNC10CT123']['max'] - self.dataset.modified.iat[i, 12] +
        #             self.dataset.modified.iat[i, 14],
        #             (vars[24] - vars[25]) - (vars[28] - vars[29]),
        #             'U3_HNC10CT121 - U3_HNC10CT123')
        #         # U3_HNC10CT13组
        #         model.add_cons(
        #             info['U3_HNC10CT131']['U3_HNC10CT132']['min'] - self.dataset.modified.iat[i, 15] +
        #             self.dataset.modified.iat[i, 16],
        #             info['U3_HNC10CT131']['U3_HNC10CT132']['max'] - self.dataset.modified.iat[i, 15] +
        #             self.dataset.modified.iat[i, 16],
        #             (vars[30] - vars[31]) - (vars[32] - vars[33]),
        #             'U3_HNC10CT131 - U3_HNC10CT132')
        #         model.add_cons(
        #             info['U3_HNC10CT131']['U3_HNC10CT133']['min'] - self.dataset.modified.iat[i, 15] +
        #             self.dataset.modified.iat[i, 17],
        #             info['U3_HNC10CT131']['U3_HNC10CT133']['max'] - self.dataset.modified.iat[i, 15] +
        #             self.dataset.modified.iat[i, 17],
        #             (vars[30] - vars[31]) - (vars[34] - vars[35]),
        #             'U3_HNC10CT131 - U3_HNC10CT133')
        #         # U3_HNC10CT14组
        #         model.add_cons(
        #             info['U3_HNC10CT141']['U3_HNC10CT142']['min'] - self.dataset.modified.iat[i, 18] +
        #             self.dataset.modified.iat[i, 19],
        #             info['U3_HNC10CT141']['U3_HNC10CT142']['max'] - self.dataset.modified.iat[i, 18] +
        #             self.dataset.modified.iat[i, 19],
        #             (vars[36] - vars[37]) - (vars[38] - vars[39]),
        #             'U3_HNC10CT141 - U3_HNC10CT142')
        #         model.add_cons(
        #             info['U3_HNC10CT141']['U3_HNC10CT143']['min'] - self.dataset.modified.iat[i, 18] +
        #             self.dataset.modified.iat[i, 20],
        #             info['U3_HNC10CT141']['U3_HNC10CT143']['max'] - self.dataset.modified.iat[i, 18] +
        #             self.dataset.modified.iat[i, 20],
        #             (vars[36] - vars[37]) - (vars[40] - vars[41]),
        #             'U3_HNC10CT141 - U3_HNC10CT143')
        #         # U3_HNC10CT15组
        #         model.add_cons(
        #             info['U3_HNC10CT151']['U3_HNC10CT152']['min'] - self.dataset.modified.iat[i, 21] +
        #             self.dataset.modified.iat[i, 22],
        #             info['U3_HNC10CT151']['U3_HNC10CT152']['max'] - self.dataset.modified.iat[i, 21] +
        #             self.dataset.modified.iat[i, 22],
        #             (vars[42] - vars[43]) - (vars[44] - vars[45]),
        #             'U3_HNC10CT151 - U3_HNC10CT152')
        #         # U3_HNC10CT17组
        #         model.add_cons(
        #             info['U3_HNC10CT171']['U3_HNC10CT172']['min'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 26],
        #             info['U3_HNC10CT171']['U3_HNC10CT172']['max'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 26],
        #             (vars[50] - vars[51]) - (vars[52] - vars[53]),
        #             'U3_HNC10CT171 - U3_HNC10CT172')
        #         model.add_cons(
        #             info['U3_HNC10CT171']['U3_HNC10CT173']['min'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 27],
        #             info['U3_HNC10CT171']['U3_HNC10CT173']['max'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 27],
        #             (vars[50] - vars[51]) - (vars[54] - vars[55]),
        #             'U3_HNC10CT171 - U3_HNC10CT173')
        #         model.add_cons(
        #             info['U3_HNC10CT171']['U3_HNC10CT174']['min'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 28],
        #             info['U3_HNC10CT171']['U3_HNC10CT174']['max'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 28],
        #             (vars[50] - vars[51]) - (vars[56] - vars[57]),
        #             'U3_HNC10CT171 - U3_HNC10CT174')
        #         model.add_cons(
        #             info['U3_HNC10CT171']['U3_HNC10CT175']['min'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 29],
        #             info['U3_HNC10CT171']['U3_HNC10CT175']['max'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 29],
        #             (vars[50] - vars[51]) - (vars[58] - vars[59]),
        #             'U3_HNC10CT171 - U3_HNC10CT175')
        #         model.add_cons(
        #             info['U3_HNC10CT171']['U3_HNC10CT176']['min'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 30],
        #             info['U3_HNC10CT171']['U3_HNC10CT176']['max'] - self.dataset.modified.iat[i, 25] +
        #             self.dataset.modified.iat[i, 30],
        #             (vars[50] - vars[51]) - (vars[60] - vars[61]),
        #             'U3_HNC10CT171 - U3_HNC10CT176')
        #         # U3_HNV20CT10组
        #         model.add_cons(
        #             info['U3_HNV20CT104']['U3_HNV20CT101']['min'] - self.dataset.modified.iat[i, 40] +
        #             self.dataset.modified.iat[i, 38],
        #             info['U3_HNV20CT104']['U3_HNV20CT101']['max'] - self.dataset.modified.iat[i, 40] +
        #             self.dataset.modified.iat[i, 38],
        #             (vars[80] - vars[81]) - (vars[76] - vars[77]),
        #             'U3_HNV20CT104 - U3_HNV20CT101')
        #         model.add_cons(
        #             info['U3_HNV20CT104']['U3_HNV20CT102']['min'] - self.dataset.modified.iat[i, 40] +
        #             self.dataset.modified.iat[i, 39],
        #             info['U3_HNV20CT104']['U3_HNV20CT102']['max'] - self.dataset.modified.iat[i, 40] +
        #             self.dataset.modified.iat[i, 39],
        #             (vars[80] - vars[81]) - (vars[78] - vars[79]),
        #             'U3_HNV20CT104 - U3_HNV20CT102')
        #         # 求解
        #         start = time.perf_counter()
        #         model.solve_prob()
        #         end = time.perf_counter()
        #         t += end - start
        #         # model.display_results()
        #         # 修复
        #         for idx, col in enumerate(info.keys()):
        #             u = vars[idx * 2].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
        #             v = vars[idx * 2 + 1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
        #             self.dataset.modified.iat[i, idx] += (u - v)
        #             if u - v > 0 or u - v < 0:
        #                 print(" delta of {0} at {1}: {2}".format(col, self.dataset.modified.index.values[idx],
        #                                                          round(u - v, 2)))
        #     except MdoError as e:
        #         print("Received Mindopt exception.")
        #         print(" - Code          : {}".format(e.code))
        #         print(" - Reason        : {}".format(e.message))
        #     finally:
        #         model.free_mdl()

        # 计算每个属性的最大最小值
        info = {}
        for col in self.dataset.clean.columns:
            info[col] = {}
            values = self.dataset.clean[col].values
            info[col]['min'] = 0.
            info[col]['max'] = np.max(values)
            info[col]['mean'] = np.mean(values)

        t = 0.

        # 清洗
        for i in range(len(self.dataset.modified)):
            model = MdoModel()
            try:
                # 目标函数最小化
                model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
                # 变量
                vars = []
                for col in self.dataset.modified.columns:
                    vars.append(
                        model.add_var(0, MdoModel.get_infinity(), 1. / info[col]['mean'], None, 'u_' + col, False))
                    vars.append(
                        model.add_var(0, MdoModel.get_infinity(), 1. / info[col]['mean'], None, 'v_' + col, False))
                # 油色谱约束
                # 所有变量必须大于等于0.
                model.add_cons(
                    info[col]['min'] - self.dataset.modified.iat[i, 0],
                    info[col]['max'] - self.dataset.modified.iat[i, 0],
                    vars[0] - vars[1],
                    'min < H2 < max'
                )
                model.add_cons(
                    info[col]['min'] - self.dataset.modified.iat[i, 1],
                    info[col]['max'] - self.dataset.modified.iat[i, 1],
                    vars[2] - vars[3],
                    'min < CH4 < max'
                )
                model.add_cons(
                    info[col]['min'] - self.dataset.modified.iat[i, 2],
                    info[col]['max'] - self.dataset.modified.iat[i, 2],
                    vars[4] - vars[5],
                    'min < C2H4 < max'
                )
                model.add_cons(
                    info[col]['min'] - self.dataset.modified.iat[i, 3],
                    info[col]['max'] - self.dataset.modified.iat[i, 3],
                    vars[6] - vars[7],
                    'min < C2H2 < max'
                )
                model.add_cons(
                    info[col]['min'] - self.dataset.modified.iat[i, 4],
                    info[col]['max'] - self.dataset.modified.iat[i, 4],
                    vars[8] - vars[9],
                    'min < C2H6 < max'
                )
                model.add_cons(
                    info[col]['min'] - self.dataset.modified.iat[i, 5],
                    info[col]['max'] - self.dataset.modified.iat[i, 5],
                    vars[10] - vars[11],
                    'min < CO < max'
                )
                model.add_cons(
                    info[col]['min'] - self.dataset.modified.iat[i, 6],
                    info[col]['max'] - self.dataset.modified.iat[i, 6],
                    vars[12] - vars[13],
                    'min < CO2 < max'
                )
                # 三比值法
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -self.dataset.modified.iat[i, 3] + 0.099 * self.dataset.modified.iat[i, 2],
                    (vars[6] - vars[7]) - 0.099 * (vars[4] - vars[5]),
                    '0 < C2H2 / C2H4 < 0.1'
                )
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -self.dataset.modified.iat[i, 1] + 0.99 * self.dataset.modified.iat[i, 0],
                    (vars[2] - vars[3]) - 0.99 * (vars[0] - vars[1]),
                    'CH4 / H2 < 1'
                )
                model.add_cons(
                    -self.dataset.modified.iat[i, 2] + 1.01 * self.dataset.modified.iat[i, 4],
                    MdoModel.get_infinity(),
                    (vars[4] - vars[5]) - 1.01 * (vars[8] - vars[9]),
                    '1 <= C2H4 / C2H6'
                )
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -self.dataset.modified.iat[i, 2] + 2.99 * self.dataset.modified.iat[i, 4],
                    (vars[4] - vars[5]) - 2.99 * (vars[8] - vars[9]),
                    'C2H4 / C2H6 < 3'
                )
                model.add_cons(
                    -MdoModel.get_infinity(),
                    -self.dataset.modified.iat[i, 6] + 6.99 * self.dataset.modified.iat[i, 5],
                    (vars[12] - vars[13]) - 6.99 * (vars[10] - vars[11]),
                    'CO2 / CO < 7'
                )
                # 求解
                start = time.perf_counter()
                model.solve_prob()
                end = time.perf_counter()
                t += end - start
                # 修复
                for idx, col in enumerate(self.dataset.modified.columns):
                    u = vars[idx * 2].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
                    v = vars[idx * 2 + 1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
                    self.dataset.modified.iat[i, idx] += (u - v)
                    if u - v > 0 or u - v < 0:
                        print(" delta of {0} at {1}: {2}".format(col, self.dataset.modified.index.values[idx],
                                                                 round(u - v, 2)))
            except MdoError as e:
                print("Received Mindopt exception.")
                print(" - Code          : {}".format(e.code))
                print(" - Reason        : {}".format(e.message))
            finally:
                model.free_mdl()

        print(round(t, 2), 'ms')

        return t, error(self.dataset)


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
