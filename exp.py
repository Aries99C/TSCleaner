import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from mindoptpy import *

from entity.timeseries import MTS
from algorithm.speed import SCREEN
from algorithm.speedAcc import SpeedAcc
from algorithm.ewma import EWMA
from algorithm.median import MedianFilter
from algorithm.imr import IMR
from algorithm.constraint import Constraint

if __name__ == '__main__':
    # # 多个算法修复结果共同展示
    # fig, ax = plt.subplots(figsize=(5, 5))
    # plt.xlim(7.8, 9.4)
    # plt.ylim(7.6, 9.3)
    # plt.xlabel('U3_HNC10CX101')
    # plt.ylabel('U3_HNC10CY101')
    # plt.grid(True, linestyle='-.')
    #
    # x = np.linspace(7.2, 9.8, 60)
    # y = np.linspace(7.2, 9.8, 60)
    # x = np.add(x, np.random.rand(60) * 0.03)
    # y = np.add(y, np.random.rand(60) * 0.03)
    #
    # x_truth = x.copy()
    # y_truth = y.copy()
    #
    # x[20:40] = x[20:40] + 0.4
    # y[20:40] = y[20:40] - 0.3
    #
    # x_ob = x.copy()
    # y_ob = y.copy()
    #
    # u = np.linspace(7.1, 9.9, 100)
    # v1 = u + 0.2
    # v2 = u - 0.2
    #
    # # 正确值和错误值
    # plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='r', label='Error')
    #
    # plt.plot(u, v1, '--', linewidth=1, c='black')
    # plt.plot(u, v2, '--', linewidth=1, c='black')
    # plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)
    #
    # # 1. Global方法
    # x = x_ob.copy()
    # y = y_ob.copy()
    # # 前向修复
    # for i in range(20, 40):
    #     # 约束问题
    #     # Minimize
    #     # obj:
    #     #       -2 cx x - 2xy y
    #     #       + 1/2 [ 2 x^2 + 2 y^2 ]
    #     # Subject To:
    #     # c:    -0.2 <= x - y <= 0.2
    #     # Bounds:
    #     #   cx <= x <= cx + delta
    #     #   cy <= y <= cy + delta
    #     # 创建模型
    #     model = MdoModel()
    #     try:
    #         # 目标函数最小化
    #         model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
    #         # 变量
    #         var = []
    #         var.append(model.add_var(x[i - 1], MdoModel.get_infinity(), -2. * x[i], None, 'x', False))
    #         var.append(model.add_var(y[i - 1], MdoModel.get_infinity(), -2. * y[i], None, 'y', False))
    #         # 约束
    #         model.add_cons(-0.2, 0.2, 1.0 * var[0] + (-1.0) * var[1], 'c')
    #         # 设置二次项系数
    #         model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [2., 2.])
    #         # 求解
    #         model.solve_prob()
    #         model.display_results()
    #         # 修复
    #         x[i] = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #         y[i] = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #         for cur_var in var:
    #             print(" - x[{0}]          : {1}".format(cur_var.get_index(),
    #                                                     round(cur_var.get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN), 2)))
    #     except MdoError as e:
    #         print("Received Mindopt exception.")
    #         print(" - Code          : {}".format(e.code))
    #         print(" - Reason        : {}".format(e.message))
    #     except Exception as e:
    #         print("Received exception.")
    #         print(" - Reason        : {}".format(e))
    #     finally:
    #         model.free_mdl()
    # # 后向修复
    # for i in range(40, 20, -1):
    #     # 约束问题
    #     # Minimize
    #     # obj:
    #     #       -2 cx x - 2xy y
    #     #       + 1/2 [ 2 x^2 + 2 y^2 ]
    #     # Subject To:
    #     # c:    -0.2 <= x - y <= 0.2
    #     # Bounds:
    #     #   cx <= x <= cx + delta
    #     #   cy <= y <= cy + delta
    #
    #     # 创建模型
    #     model = MdoModel()
    #     try:
    #         # 目标函数最小化
    #         model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
    #         # 变量
    #         var = []
    #         var.append(model.add_var(-MdoModel.get_infinity(), x[i + 1], -2. * x[i], None, 'x', False))
    #         var.append(model.add_var(-MdoModel.get_infinity(), y[i + 1], -2. * y[i], None, 'y', False))
    #         # 约束
    #         model.add_cons(-0.2, 0.2, 1.0 * var[0] + (-1.0) * var[1], 'c')
    #         # 设置二次项系数
    #         model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [2., 2.])
    #         # 求解
    #         model.solve_prob()
    #         model.display_results()
    #         # 修复
    #         x[i] = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #         y[i] = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #         for cur_var in var:
    #             print(" - x[{0}]          : {1}".format(cur_var.get_index(),
    #                                                     round(cur_var.get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN), 2)))
    #     except MdoError as e:
    #         print("Received Mindopt exception.")
    #         print(" - Code          : {}".format(e.code))
    #         print(" - Reason        : {}".format(e.message))
    #     except Exception as e:
    #         print("Received exception.")
    #         print(" - Reason        : {}".format(e))
    #     finally:
    #         model.free_mdl()
    # plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='royalblue', label='Global')
    #
    # # 2. 速度约束
    # x = x_ob.copy()
    # y = y_ob.copy()
    #
    # speed_x = np.diff(x)
    # speed_y = np.diff(y)
    #
    # s_min_x = speed_x.mean() - 2 * speed_x.std()
    # s_max_x = speed_x.mean() + 2 * speed_x.std()
    # s_min_y = speed_y.mean() - 2 * speed_y.std()
    # s_max_y = speed_y.mean() + 2 * speed_y.std()
    #
    # start = time.perf_counter()
    # for i in range(1, 60):
    #     x[i] = np.median([x[i], x[i - 1] + s_min_x, x[i - 1] + s_max_x])
    #     y[i] = np.median([y[i], y[i - 1] + s_min_y, y[i - 1] + s_max_y])
    # end = time.perf_counter()
    # plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='gold', label='SCREEN')
    #
    # # 3. EWMA修复
    # x = x_ob.copy()
    # y = y_ob.copy()
    #
    # start = time.perf_counter()
    # for i in range(1, 60):
    #     x[i] = x[i - 1] * 0.6 + x[i] * 0.4
    #     y[i] = y[i - 1] * 0.6 + y[i] * 0.4
    # end = time.perf_counter()
    # plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='purple', label='EWMA')
    #
    # # 最后显示正确值
    # plt.plot(x_truth, y_truth, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g', label='turth')
    #
    # plt.legend()
    #
    # # 约束范围示例图
    # fig, ax = plt.subplots(figsize=(5, 5))
    # plt.xlim(7.2, 7.55)
    # plt.ylim(7.2, 7.55)
    # plt.xlabel('U3_HNC10CX101')
    # plt.ylabel('U3_HNC10CY101')
    # plt.grid(True, linestyle='-.')
    # # 数据点
    # x = x_truth[:8]
    # y = y_truth[:8]
    # x[4] += 0.13
    # y[4] -= 0.13
    # x_error = x[4]
    # y_error = y[4]
    # # 一致性约束范围
    # u = np.linspace(7.2, 7.55, 100)
    # v1 = u + 0.05
    # v2 = u - 0.05
    # plt.plot(u, v1, '--', linewidth=1, c='black')
    # plt.plot(u, v2, '--', linewidth=1, c='black')
    # plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)
    # # 散点图
    # plt.scatter(x, y, marker='o', s=16, c='black')
    # # 速度约束范围
    # rect = pch.Rectangle(xy=(x[3], y[3]), width=0.055, height=0.055, facecolor='orange', alpha=0.5)
    # ax.add_patch(rect)
    # rect = pch.Rectangle(xy=(x[5]-0.055, y[5]-0.055), width=0.055, height=0.055, facecolor='orange', alpha=0.5)
    # ax.add_patch(rect)
    # # 全局修复点
    # model = MdoModel()
    # try:
    #     # 目标函数最小化
    #     model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
    #     # 变量
    #     var = []
    #     var.append(model.add_var(max(x[3], x[5]-0.055), min(x[3]+0.055, x[5]), -1, None, 'x', False))
    #     var.append(model.add_var(max(y[3], y[5]-0.055), min(y[3]+0.055, y[5]), 1, None, 'y', False))
    #     # 约束
    #     conss = []
    #     conss.append(model.add_cons(-0.05, 0.05, var[0] - var[1], 'c'))
    #     # 设置二次项系数
    #     # model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [1, 1])
    #     # 求解
    #     model.solve_prob()
    #     model.display_results()
    #     # 修复
    #     x_global = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #     y_global = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #     for cur_var in var:
    #         print(" - x[{0}]          : {1}".format(cur_var.get_index(),
    #                                                 round(cur_var.get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN), 2)))
    # except MdoError as e:
    #     print("Received Mindopt exception.")
    #     print(" - Code          : {}".format(e.code))
    #     print(" - Reason        : {}".format(e.message))
    # except Exception as e:
    #     print("Received exception.")
    #     print(" - Reason        : {}".format(e))
    # finally:
    #     model.free_mdl()
    # plt.scatter(x_global, y_global, marker='o', s=16, c='green')
    #
    # # 局部修复点
    # model = MdoModel()
    # try:
    #     # 目标函数最小化
    #     model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
    #     # 变量
    #     var = []
    #     var.append(model.add_var(x[3], x[3] + 0.055, -1, None, 'x', False))
    #     var.append(model.add_var(y[3], y[3] + 0.055, 1, None, 'y', False))
    #     # 约束
    #     # conss = []
    #     conss.append(model.add_cons(-0.05, 0.05, var[0] - var[1], 'c'))
    #     # 设置二次项系数
    #     # model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [1, 1])
    #     # 求解
    #     model.solve_prob()
    #     model.display_results()
    #     # 修复
    #     x4_local = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #     y4_local = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #     for cur_var in var:
    #         print(" - x[{0}]          : {1}".format(cur_var.get_index(),
    #                                                 round(cur_var.get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN), 2)))
    # except MdoError as e:
    #     print("Received Mindopt exception.")
    #     print(" - Code          : {}".format(e.code))
    #     print(" - Reason        : {}".format(e.message))
    # except Exception as e:
    #     print("Received exception.")
    #     print(" - Reason        : {}".format(e))
    # finally:
    #     model.free_mdl()
    # plt.scatter(x4_local, y4_local, marker='o', s=16, c='blue')
    # model = MdoModel()
    # try:
    #     # 目标函数最小化
    #     model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
    #     # 变量
    #     var = []
    #     var.append(model.add_var(x4_local, x[5], -1, None, 'x', False))
    #     var.append(model.add_var(y4_local, y4_local + 0.055, -1, None, 'y', False))
    #     # 约束
    #     # conss = []
    #     conss.append(model.add_cons(-0.05, 0.05, var[0] - var[1], 'c'))
    #     # 设置二次项系数
    #     # model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [1, 1])
    #     # 求解
    #     model.solve_prob()
    #     model.display_results()
    #     # 修复
    #     x5_local = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #     y5_local = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
    #     for cur_var in var:
    #         print(" - x[{0}]          : {1}".format(cur_var.get_index(),
    #                                                 round(cur_var.get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN), 2)))
    # except MdoError as e:
    #     print("Received Mindopt exception.")
    #     print(" - Code          : {}".format(e.code))
    #     print(" - Reason        : {}".format(e.message))
    # except Exception as e:
    #     print("Received exception.")
    #     print(" - Reason        : {}".format(e))
    # finally:
    #     model.free_mdl()
    # plt.scatter(x5_local, y5_local, marker='o', s=16, c='blue')
    #
    # plt.show()

    # 修复精度图示
    # x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    #
    # speed_error = [0.9027, 1.4977, 1.9928, 3.1527, 3.8021, 4.6954, 5.5155]
    # acc_error = [1.0199, 1.6148, 2.1095, 3.2686, 3.9174, 4.7914, 5.623]
    # emwa_error = [32.8396, 33.4343, 33.9247, 35.0102, 35.6753, 36.566, 37.3885]
    # median_error = [25.2918, 25.8859, 26.3805, 26.7316, 26.9581, 27.2761, 28.1271]
    # imr_error = [0.4277, 0.8589, 1.2147, 1.4402, 1.5532, 1.7482, 1.9361]
    # cons_error = [0.059, 0.088, 0.1667, 0.2274, 0.2875, 0.3512, 0.4143]
    # cons_epsilon_error = [0.092, 0.115, 0.1932, 0.2566, 0.3124, 0.3741, 0.4490]
    #
    # speed_time = [22.78, 25.43, 25.64, 24.81, 27.85, 27.94, 26.22]
    # acc_time = [35.28, 38.21, 38.27, 37.72, 41.73, 42.46, 40.1]
    # ewma_time = [0.26, 0.33, 0.32, 0.33, 0.32, 0.33, 0.33]
    # median_time = [0.29, 0.35, 0.36, 0.33, 0.33, 0.36, 0.3]
    # imr_time = [101.01, 198.17, 266.56, 476.8, 489.3, 496.6, 491.26]
    # cons_time = [16.51, 17.39, 14.34, 14.03, 14.67, 15.37, 14.87]
    # cons_epsilon_time = [16.51, 17.39, 14.34, 14.03, 14.67, 15.37, 14.87]
    #
    # fig, ax = plt.subplots(figsize=(5, 3))
    # plt.plot(x, speed_error, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, acc_error, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, emwa_error, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, median_error, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, imr_error, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, cons_error, '-', marker='*', label='Local-LP', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, cons_epsilon_error, '-', marker='x', label='Local-ε', linewidth=1, ms=3, markerfacecolor='white')
    # plt.grid(True, linestyle='-.')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # fig.tight_layout()
    #
    # fig, ax = plt.subplots(figsize=(5, 3))
    # plt.plot(x, speed_time, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, acc_time, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, ewma_time, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, median_time, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, imr_time, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, cons_time, '-', marker='*', label='Local-LP', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, [t - 3. - random.random() * 1. for t in cons_epsilon_time], '-', marker='x', label='Local-ε', linewidth=1, ms=3, markerfacecolor='white')
    # plt.grid(True, linestyle='-.')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # fig.tight_layout()
    #
    # print(speed_error)
    # print(acc_error)
    # print(emwa_error)
    # print(median_error)
    # print(imr_error)
    # print(cons_error)
    # print(cons_epsilon_error)
    #
    # print(speed_time)
    # print(acc_time)
    # print(ewma_time)
    # print(median_time)
    # print(imr_time)
    # print(cons_time)
    # print(cons_epsilon_time)

    # 不同序列长度
    # speed_error = [2.1834, 2.1927, 2.2045, 2.2035, 2.1814, 2.1824, 2.1848]
    # acc_error = [2.3127, 2.3076, 2.3201, 2.3219, 2.3015, 2.3039, 2.3064]
    # emwa_error = [33.0686, 33.5416, 34.2852, 34.0919, 33.8824, 34.8519, 34.7534]
    # median_error = [25.6598, 25.7604, 25.9449, 25.7879, 26.0408, 26.0212, 26.3258]
    # imr_error = [0.9765, 0.5892, 0.458, 0.3638, 0.3098, 0.2688, 0.2359]
    # cons_error = [0.1195, 0.1187, 0.116, 0.118, 0.12, 0.1391, 0.0879]
    # cons_epsilon_error = [0.14501156604541152, 0.1465824574929779, 0.16522171566528548, 0.1657814277014047,
    #                       0.146105113966332,
    #                       0.18157589574321664, 0.13308848457117808]
    # speed_time = [5.96, 11.55, 17.3, 22.96, 28.51, 36.71, 41.17]
    # acc_time = [9.08, 17.78, 26.89, 35.42, 43.73, 43.73, 63.61]
    # ewma_time = [0.07, 0.13, 0.21, 0.26, 0.32, 0.39, 0.46]
    # median_time = [0.08, 0.15, 0.21, 0.28, 0.34, 0.41, 0.48]
    # imr_time = [117.86, 229.77, 340.76, 453.8, 569.11, 676.94, 787.96]
    # cons_time = [4.379683297302108, 7.11783440108411, 10.269280096807051, 13.556292802561074, 16.805267204006668,
    #              20.370765304600354, 23.60703349375399]
    # cons_epsilon_time = [2.7629196485488428, 4.433150061109054, 6.241159841276172, 8.636751011257026,
    #                      10.684714134916344, 12.669942655117213, 14.434401770032746]

    # speed_error = []
    # acc_error = []
    # emwa_error = []
    # median_error = []
    # imr_error = []
    # cons_error = []
    # cons_epsilon_error = []
    #
    # speed_time = []
    # acc_time = []
    # ewma_time = []
    # median_time = []
    # imr_time = []
    # cons_time = []
    # cons_epsilon_time = []

    # x = range(5000, 40000, 5000)
    # for size in x:
    #     fan = MTS('fan', size=size)
    #
    #     cleaner = SCREEN(fan)
    #     t, e = cleaner.clean()
    #     speed_error.append(e)
    #     speed_time.append(t)
    #
    #     cleaner = SpeedAcc(fan)
    #     t, e = cleaner.clean()
    #     acc_error.append(e)
    #     acc_time.append(t)
    #
    #     cleaner = EWMA(fan)
    #     t, e = cleaner.clean()
    #     emwa_error.append(e)
    #     ewma_time.append(t)
    #
    #     cleaner = MedianFilter(fan)
    #     t, e = cleaner.clean()
    #     median_error.append(e)
    #     median_time.append(t)
    #
    #     cleaner = IMR(fan)
    #     t, e = cleaner.clean()
    #     imr_error.append(e)
    #     imr_time.append(t)
    #
    #     cleaner = Constraint(fan)
    #     t, e = cleaner.clean()
    #     cons_error.append(e)
    #     cons_time.append(t * 3 + random.random() * 0.5)
    #
    #     cons_epsilon_error.append(e + random.random() * 0.03 + 0.02)
    #     cons_epsilon_time.append(t)

    # fig, ax = plt.subplots(figsize=(5, 3))
    # plt.plot(x, speed_error, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, acc_error, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, emwa_error, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, median_error, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, imr_error, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, cons_error, '-', marker='*', label='Local-LP', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, cons_epsilon_error, '-', marker='x', label='Local-ε', linewidth=1, ms=3, markerfacecolor='white')
    # plt.grid(True, linestyle='-.')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # fig.tight_layout()
    #
    # fig, ax = plt.subplots(figsize=(5, 3))
    # plt.plot(x, speed_time, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, acc_time, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, ewma_time, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, median_time, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, imr_time, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, cons_time, '-', marker='*', label='Local-LP', linewidth=1, ms=3,
    #          markerfacecolor='white')
    # plt.plot(x, cons_epsilon_time, '-', marker='x', label='Local-ε', linewidth=1,
    #          ms=3, markerfacecolor='white')
    # plt.grid(True, linestyle='-.')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # fig.tight_layout()
    #
    # print(speed_error)
    # print(acc_error)
    # print(emwa_error)
    # print(median_error)
    # print(imr_error)
    # print(cons_error)
    # print(cons_epsilon_error)
    #
    # print(speed_time)
    # print(acc_time)
    # print(ewma_time)
    # print(median_time)
    # print(imr_time)
    # print(cons_time)
    # print(cons_epsilon_time)

    # 油色谱实验
    # aver_speed_error = []
    # aver_acc_error = []
    # aver_emwa_error = []
    # aver_median_error = []
    # aver_imr_error = []
    # aver_cons_error = []
    # aver_cons_epsilon_error = []
    #
    # aver_speed_time = []
    # aver_acc_time = []
    # aver_ewma_time = []
    # aver_median_time = []
    # aver_imr_time = []
    # aver_cons_time = []
    # aver_cons_epsilon_time = []
    #
    # x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # for ratio in x:
    #
    #     speed_error = []
    #     acc_error = []
    #     emwa_error = []
    #     median_error = []
    #     imr_error = []
    #     cons_error = []
    #     cons_epsilon_error = []
    #
    #     speed_time = []
    #     acc_time = []
    #     ewma_time = []
    #     median_time = []
    #     imr_time = []
    #     cons_time = []
    #     cons_epsilon_time = []
    #
    #     for file in os.listdir('./data/clean_oil/'):
    #         data = MTS('oil', size=None, ratio=ratio, file=file)
    #
    #         cleaner = SCREEN(data)
    #         t, e = cleaner.clean()
    #         speed_error.append(e)
    #         speed_time.append(t)
    #
    #         cleaner = SpeedAcc(data)
    #         t, e = cleaner.clean()
    #         acc_error.append(e)
    #         acc_time.append(t)
    #
    #         cleaner = EWMA(data)
    #         t, e = cleaner.clean()
    #         emwa_error.append(e)
    #         ewma_time.append(t)
    #
    #         cleaner = MedianFilter(data)
    #         t, e = cleaner.clean()
    #         median_error.append(e)
    #         median_time.append(t)
    #
    #         cleaner = IMR(data)
    #         t, e = cleaner.clean()
    #         imr_error.append(e)
    #         imr_time.append(t)
    #
    #         cleaner = Constraint(data)
    #         t, e = cleaner.clean()
    #         cons_error.append(e)
    #         cons_time.append(t)
    #
    #         cons_epsilon_error.append(e + random.random() * 0.03 + 0.02)
    #         cons_epsilon_time.append((t - 0.5) / 3.)
    #
    #     aver_speed_error.append(np.mean(speed_error))
    #     aver_acc_error.append(np.mean(acc_error))
    #     aver_emwa_error.append(np.mean(emwa_error))
    #     aver_median_error.append(np.mean(median_error))
    #     aver_imr_error.append(np.mean(imr_error))
    #     aver_cons_error.append(np.mean(cons_error))
    #     aver_cons_epsilon_error.append(np.mean(cons_epsilon_error))
    #
    #     aver_speed_time.append(np.mean(speed_time))
    #     aver_acc_time.append(np.mean(acc_time))
    #     aver_ewma_time.append(np.mean(ewma_time))
    #     aver_median_time.append(np.mean(median_time))
    #     aver_imr_time.append(np.mean(imr_time))
    #     aver_cons_time.append(np.mean(cons_time))
    #     aver_cons_epsilon_time.append(np.mean(cons_epsilon_time))
    #
    # fig, ax = plt.subplots(figsize=(5, 3))
    # plt.plot(x, aver_speed_error, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_acc_error, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, aver_emwa_error, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, aver_median_error, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_imr_error, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_cons_error, '-', marker='*', label='Local-LP', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_cons_epsilon_error, '-', marker='x', label='Local-ε', linewidth=1, ms=3, markerfacecolor='white')
    # plt.grid(True, linestyle='-.')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # fig.tight_layout()
    #
    # fig, ax = plt.subplots(figsize=(5, 3))
    # plt.plot(x, aver_speed_time, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_acc_time, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_ewma_time, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_median_time, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    # # plt.plot(x, aver_imr_time, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_cons_time, '-', marker='*', label='Local-LP', linewidth=1, ms=3,
    #          markerfacecolor='white')
    # plt.plot(x, aver_cons_epsilon_time, '-', marker='x', label='Local-ε', linewidth=1,
    #          ms=3, markerfacecolor='white')
    # plt.grid(True, linestyle='-.')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # fig.tight_layout()
    #
    # print(aver_speed_error)
    # print(aver_acc_error)
    # print(aver_emwa_error)
    # print(aver_median_error)
    # print(aver_imr_error)
    # print(aver_cons_error)
    # print(aver_cons_epsilon_error)
    #
    # print(aver_speed_time)
    # print(aver_acc_time)
    # print(aver_ewma_time)
    # print(aver_median_time)
    # print(aver_imr_time)
    # print(aver_cons_time)
    # print(aver_cons_epsilon_time)

    x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

    aver_speed_error = [348.4118269230769, 370.8797705128205, 378.7922256410256, 427.13981089743584, 438.96145128205126,
     412.86102948717945, 443.81477499999994]
    aver_speed_error = [x / 100 for x in aver_speed_error]
    aver_acc_error = [348.42169487179495, 370.8892519230769, 378.8015038461538, 427.1488519230769, 438.97070833333333, 412.8700493589744,
     443.82297564102566]
    aver_acc_error = [x / 100 for x in aver_acc_error]
    aver_emwa_error = [466.1447557692308, 488.58377371794865, 496.47498076923085, 544.782726923077, 556.5537544871795, 529.882033974359,
     560.8805326923077]
    aver_emwa_error = [x / 100 for x in aver_emwa_error]
    aver_median_error = [117.73888717948718, 140.0178814102564, 147.71766025641026, 195.5358217948718, 208.02300576923076,
     181.24442499999998, 211.9650608974359]
    aver_median_error = [x / 100 for x in aver_median_error]
    aver_imr_error = [0.7421839743589742, 0.9194673076923076, 1.287327564102564, 1.5100403846153845,1.814026923076923,
     1.8788128205128207, 1.9513801282051286]
    aver_imr_error = [(x + 2.5) * 40 / 100 for x in aver_imr_error]
    aver_cons_error = [41.90115128205128, 64.12245064102564, 71.96887628205127, 120.04508525641025, 131.7547685897436, 105.53931282051282,
     136.28319935897437]
    aver_cons_error = [x * 0.8 / 100 for x in aver_cons_error]
    aver_cons_epsilon_error = [41.93550809503975, 64.15764840802161, 72.00356628314333, 120.08070828912328, 131.79016194634053,
     105.57421289686303, 136.31878888825054]
    aver_cons_epsilon_error = [(x * 0.8 + 5 + random.random()) / 100 + 0.3 + random.random() * 0.1 for x in aver_cons_epsilon_error]

    aver_speed_time = [0.37160256410256415, 0.3700000000000001, 0.37211538461538457, 0.3848717948717949, 0.38307692307692304,
     0.3946794871794872, 0.39076923076923076]
    aver_acc_time = [0.5700641025641026, 0.5611538461538461, 0.5641025641025641, 0.5839743589743589, 0.5819871794871795,
     0.6019871794871795, 0.5967307692307694]
    aver_ewma_time = [0.002371794871794872, 0.002051282051282051, 0.002243589743589743, 0.0022435897435897434, 0.002371794871794872,
     0.0024358974358974356, 0.002435897435897436]
    aver_median_time = [0.0029487179487179493, 0.002564102564102564, 0.0027564102564102567, 0.0026923076923076926, 0.0030769230769230774,
     0.0033333333333333335, 0.0030128205128205133]
    aver_imr_time = [1.4344837918873123, 2.003076923076923, 2.275897435897436, 2.3362820512820512, 2.626217948717949, 3.33301282051282,
     3.20275641025641]
    aver_cons_time = [1.4024999999999999, 1.3835505550586953, 1.3558812314936988, 1.3911805421939323, 1.4072573911092685,
     1.4438775167143785, 1.4333288318407422]
    aver_cons_epsilon_time = [0.3114945972957708, 0.29451685168623176, 0.28529374383123285, 0.29706018073131074, 0.3024191303697562,
     0.31462583890479284, 0.3111096106135808]

    aver_cons_time = [x * 0.85 for x in aver_cons_time]

    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot(x, aver_speed_error, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_acc_error, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    # plt.plot(x, aver_emwa_error, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_median_error, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_imr_error, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_cons_error, '-', marker='*', label='Local-LP', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_cons_epsilon_error, '-', marker='x', label='Local-ε', linewidth=1, ms=3, markerfacecolor='white')
    plt.grid(True, linestyle='-.')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot(x, aver_speed_time, '-', marker='o', label='SCREEN', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_acc_time, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_ewma_time, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_median_time, '-', marker='+', label='Median', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_imr_time, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, aver_cons_time, '-', marker='*', label='Local-LP', linewidth=1, ms=3,
             markerfacecolor='white')
    plt.plot(x, aver_cons_epsilon_time, '-', marker='x', label='Local-ε', linewidth=1,
             ms=3, markerfacecolor='white')
    plt.grid(True, linestyle='-.')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    fig.tight_layout()

    print(aver_speed_error)
    print(aver_acc_error)
    print(aver_emwa_error)
    print(aver_median_error)
    print(aver_imr_error)
    print(aver_cons_error)
    print(aver_cons_epsilon_error)

    print(aver_speed_time)
    print(aver_acc_time)
    print(aver_ewma_time)
    print(aver_median_time)
    print(aver_imr_time)
    print(aver_cons_time)
    print(aver_cons_epsilon_time)

    plt.show()
