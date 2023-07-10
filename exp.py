import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from mindoptpy import *


if __name__ == '__main__':
    # 多个算法修复结果共同展示
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.xlim(7.8, 9.4)
    plt.ylim(7.6, 9.3)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')

    x = np.linspace(7.2, 9.8, 60)
    y = np.linspace(7.2, 9.8, 60)
    x = np.add(x, np.random.rand(60) * 0.03)
    y = np.add(y, np.random.rand(60) * 0.03)

    x_truth = x.copy()
    y_truth = y.copy()

    x[20:40] = x[20:40] + 0.4
    y[20:40] = y[20:40] - 0.3

    x_ob = x.copy()
    y_ob = y.copy()

    u = np.linspace(7.1, 9.9, 100)
    v1 = u + 0.2
    v2 = u - 0.2

    # 正确值和错误值
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='r')

    plt.plot(u, v1, '--', linewidth=1, c='black')
    plt.plot(u, v2, '--', linewidth=1, c='black')
    plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)

    # 1. Global方法
    x = x_ob.copy()
    y = y_ob.copy()
    # 前向修复
    for i in range(20, 40):
        # 约束问题
        # Minimize
        # obj:
        #       -2 cx x - 2xy y
        #       + 1/2 [ 2 x^2 + 2 y^2 ]
        # Subject To:
        # c:    -0.2 <= x - y <= 0.2
        # Bounds:
        #   cx <= x <= cx + delta
        #   cy <= y <= cy + delta
        # 创建模型
        model = MdoModel()
        try:
            # 目标函数最小化
            model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
            # 变量
            var = []
            var.append(model.add_var(x[i - 1], MdoModel.get_infinity(), -2. * x[i], None, 'x', False))
            var.append(model.add_var(y[i - 1], MdoModel.get_infinity(), -2. * y[i], None, 'y', False))
            # 约束
            model.add_cons(-0.2, 0.2, 1.0 * var[0] + (-1.0) * var[1], 'c')
            # 设置二次项系数
            model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [2., 2.])
            # 求解
            model.solve_prob()
            model.display_results()
            # 修复
            x[i] = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
            y[i] = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
            for cur_var in var:
                print(" - x[{0}]          : {1}".format(cur_var.get_index(),
                                                        round(cur_var.get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN), 2)))
        except MdoError as e:
            print("Received Mindopt exception.")
            print(" - Code          : {}".format(e.code))
            print(" - Reason        : {}".format(e.message))
        except Exception as e:
            print("Received exception.")
            print(" - Reason        : {}".format(e))
        finally:
            model.free_mdl()
    # 后向修复
    for i in range(40, 20, -1):
        # 约束问题
        # Minimize
        # obj:
        #       -2 cx x - 2xy y
        #       + 1/2 [ 2 x^2 + 2 y^2 ]
        # Subject To:
        # c:    -0.2 <= x - y <= 0.2
        # Bounds:
        #   cx <= x <= cx + delta
        #   cy <= y <= cy + delta

        # 创建模型
        model = MdoModel()
        try:
            # 目标函数最小化
            model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
            # 变量
            var = []
            var.append(model.add_var(-MdoModel.get_infinity(), x[i + 1], -2. * x[i], None, 'x', False))
            var.append(model.add_var(-MdoModel.get_infinity(), y[i + 1], -2. * y[i], None, 'y', False))
            # 约束
            model.add_cons(-0.2, 0.2, 1.0 * var[0] + (-1.0) * var[1], 'c')
            # 设置二次项系数
            model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [2., 2.])
            # 求解
            model.solve_prob()
            model.display_results()
            # 修复
            x[i] = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
            y[i] = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
            for cur_var in var:
                print(" - x[{0}]          : {1}".format(cur_var.get_index(),
                                                        round(cur_var.get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN), 2)))
        except MdoError as e:
            print("Received Mindopt exception.")
            print(" - Code          : {}".format(e.code))
            print(" - Reason        : {}".format(e.message))
        except Exception as e:
            print("Received exception.")
            print(" - Reason        : {}".format(e))
        finally:
            model.free_mdl()
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='royalblue')

    # 速度约束
    x = x_ob.copy()
    y = y_ob.copy()

    start = time.perf_counter()
    for i in range(1, 60):
        x[i] = np.median([x[i], x[i - 1] - 2.6 / 60, x[i - 1] + 2.6 / 60])
        y[i] = np.median([y[i], y[i - 1] - 2.6 / 60, y[i - 1] + 2.6 / 60])
    end = time.perf_counter()
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='gold')

    # 用ewma修复
    x = x_ob.copy()
    y = y_ob.copy()

    start = time.perf_counter()
    for i in range(1, 60):
        x[i] = x[i - 1] * 0.6 + x[i] * 0.4
        y[i] = y[i - 1] * 0.6 + y[i] * 0.4
    end = time.perf_counter()
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='tomato')

    # 最后显示正确值
    plt.plot(x_truth, y_truth, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')

    plt.show()
