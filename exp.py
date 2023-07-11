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
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='r', label='Error')

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
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='royalblue', label='Global')

    # 2. 速度约束
    x = x_ob.copy()
    y = y_ob.copy()

    speed_x = np.diff(x)
    speed_y = np.diff(y)

    s_min_x = speed_x.mean() - 2 * speed_x.std()
    s_max_x = speed_x.mean() + 2 * speed_x.std()
    s_min_y = speed_y.mean() - 2 * speed_y.std()
    s_max_y = speed_y.mean() + 2 * speed_y.std()

    start = time.perf_counter()
    for i in range(1, 60):
        x[i] = np.median([x[i], x[i - 1] + s_min_x, x[i - 1] + s_max_x])
        y[i] = np.median([y[i], y[i - 1] + s_min_y, y[i - 1] + s_max_y])
    end = time.perf_counter()
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='gold', label='SCREEN')

    # 3. EWMA修复
    x = x_ob.copy()
    y = y_ob.copy()

    start = time.perf_counter()
    for i in range(1, 60):
        x[i] = x[i - 1] * 0.6 + x[i] * 0.4
        y[i] = y[i - 1] * 0.6 + y[i] * 0.4
    end = time.perf_counter()
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='purple', label='EWMA')

    # 最后显示正确值
    plt.plot(x_truth, y_truth, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g', label='turth')

    plt.legend()

    # 约束范围示例图
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.xlim(7.2, 7.55)
    plt.ylim(7.2, 7.55)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')
    # 数据点
    x = x_truth[:8]
    y = y_truth[:8]
    x[4] += 0.13
    y[4] -= 0.13
    x_error = x[4]
    y_error = y[4]
    # 一致性约束范围
    u = np.linspace(7.2, 7.55, 100)
    v1 = u + 0.05
    v2 = u - 0.05
    plt.plot(u, v1, '--', linewidth=1, c='black')
    plt.plot(u, v2, '--', linewidth=1, c='black')
    plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)
    # 散点图
    plt.scatter(x, y, marker='o', s=16, c='black')
    # 速度约束范围
    rect = pch.Rectangle(xy=(x[3], y[3]), width=0.055, height=0.055, facecolor='orange', alpha=0.5)
    ax.add_patch(rect)
    rect = pch.Rectangle(xy=(x[5]-0.055, y[5]-0.055), width=0.055, height=0.055, facecolor='orange', alpha=0.5)
    ax.add_patch(rect)
    # 全局修复点
    model = MdoModel()
    try:
        # 目标函数最小化
        model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
        # 变量
        var = []
        var.append(model.add_var(max(x[3], x[5]-0.055), min(x[3]+0.055, x[5]), -1, None, 'x', False))
        var.append(model.add_var(max(y[3], y[5]-0.055), min(y[3]+0.055, y[5]), 1, None, 'y', False))
        # 约束
        conss = []
        conss.append(model.add_cons(-0.05, 0.05, var[0] - var[1], 'c'))
        # 设置二次项系数
        # model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [1, 1])
        # 求解
        model.solve_prob()
        model.display_results()
        # 修复
        x_global = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
        y_global = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
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
    plt.scatter(x_global, y_global, marker='o', s=16, c='green')

    # 局部修复点
    model = MdoModel()
    try:
        # 目标函数最小化
        model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
        # 变量
        var = []
        var.append(model.add_var(x[3], x[3] + 0.055, -1, None, 'x', False))
        var.append(model.add_var(y[3], y[3] + 0.055, 1, None, 'y', False))
        # 约束
        # conss = []
        conss.append(model.add_cons(-0.05, 0.05, var[0] - var[1], 'c'))
        # 设置二次项系数
        # model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [1, 1])
        # 求解
        model.solve_prob()
        model.display_results()
        # 修复
        x4_local = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
        y4_local = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
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
    plt.scatter(x4_local, y4_local, marker='o', s=16, c='blue')
    model = MdoModel()
    try:
        # 目标函数最小化
        model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)
        # 变量
        var = []
        var.append(model.add_var(x4_local, x[5], -1, None, 'x', False))
        var.append(model.add_var(y4_local, y4_local + 0.055, -1, None, 'y', False))
        # 约束
        # conss = []
        conss.append(model.add_cons(-0.05, 0.05, var[0] - var[1], 'c'))
        # 设置二次项系数
        # model.set_quadratic_elements([var[0], var[1]], [var[0], var[1]], [1, 1])
        # 求解
        model.solve_prob()
        model.display_results()
        # 修复
        x5_local = var[0].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
        y5_local = var[1].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN)
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
    plt.scatter(x5_local, y5_local, marker='o', s=16, c='blue')

    plt.show()
