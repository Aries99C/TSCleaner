import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from mindoptpy import *

if __name__ == '__main__':
    x = np.linspace(7.2, 9.8, 60)
    y = np.linspace(7.2, 9.8, 60)
    x = np.add(x, np.random.rand(60) * 0.03)
    y = np.add(y, np.random.rand(60) * 0.03)

    x[20:40] = x[20:40] + 0.4
    y[20:40] = y[20:40] - 0.3

    u = np.linspace(7.1, 9.9, 100)
    v1 = u + 0.2
    v2 = u - 0.2

    fig, ax = plt.subplots(figsize=(5, 5))

    # 修复前范围展示
    rect = pch.Rectangle(xy=(x[19], y[19]), width=x[40] - x[19], height=y[40] - y[19], facecolor='orange', alpha=0.5)
    ax.add_patch(rect)
    plt.plot(x[:21], y[:21], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[39:], y[39:], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[20:40], y[20:40], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='r')
    plt.plot(u, v1, '--', linewidth=1, c='black')
    plt.plot(u, v2, '--', linewidth=1, c='black')
    plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)
    plt.xlim(7.1, 9.9)
    plt.ylim(7.1, 9.9)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')

    # 修复数据
    # 前向修复
    for i in range(20, 25):
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
            var.append(model.add_var(x[i-1], MdoModel.get_infinity(), -2.*x[i], None, 'x', False))
            var.append(model.add_var(y[i-1], MdoModel.get_infinity(), -2.*y[i], None, 'y', False))
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
    for i in range(40, 35, -1):
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
            var.append(model.add_var(-MdoModel.get_infinity(), x[i+1], -2.*x[i], None, 'x', False))
            var.append(model.add_var(-MdoModel.get_infinity(), y[i+1], -2.*y[i], None, 'y', False))
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

    # 修复后范围展示
    fig, ax = plt.subplots(figsize=(5, 5))
    rect = pch.Rectangle(xy=(x[24], y[24]), width=x[36] - x[24], height=y[36] - y[24], facecolor='orange', alpha=0.5)
    ax.add_patch(rect)
    plt.plot(x[:21], y[:21], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[34:], y[34:], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[20:26], y[20:26], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='b')
    plt.plot(x[25:36], y[25:36], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='r')
    plt.plot(u, v1, '--', linewidth=1, c='black')
    plt.plot(u, v2, '--', linewidth=1, c='black')
    plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)
    plt.xlim(7.1, 9.9)
    plt.ylim(7.1, 9.9)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')
    plt.show()
