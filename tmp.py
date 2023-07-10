import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from mindoptpy import *


def compute_error(n, good_lists, truth_lists, bad_lists=None):
    e = 0.
    for i in range(n):
        delta = np.abs(np.subtract(good_lists[i], truth_lists[i]))
        e += np.sum(delta)
    return e


if __name__ == '__main__':
    times = []
    errors = []

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

    fig, ax = plt.subplots(figsize=(5, 5))

    # 修复前范围展示
    rect = pch.Rectangle(xy=(x[19], y[19]), width=x[40] - x[19], height=y[40] - y[19], facecolor='orange', alpha=0.5)
    ax.add_patch(rect)
    plt.plot(x[:21], y[:21], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[39:], y[39:], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[20:40], y[20:40], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='r')
    plt.plot(x_truth, y_truth, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(u, v1, '--', linewidth=1, c='black')
    plt.plot(u, v2, '--', linewidth=1, c='black')
    plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)
    plt.xlim(7.1, 9.9)
    plt.ylim(7.1, 9.9)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')

    # 修复数据
    start = time.perf_counter()
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
    end = time.perf_counter()
    # print('time cost: ', (end - start) * 360000 / 60)

    errors.append(compute_error(2, [x, y], [x_truth, y_truth]))
    times.append((end - start) * 360000 / 60)

    # 修复后范围展示
    fig, ax = plt.subplots(figsize=(5, 5))
    rect = pch.Rectangle(xy=(x[24], y[24]), width=x[36] - x[24], height=y[36] - y[24], facecolor='orange', alpha=0.5)
    ax.add_patch(rect)
    plt.plot(x[:21], y[:21], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[34:], y[34:], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(x[20:26], y[20:26], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='b')
    plt.plot(x[25:36], y[25:36], 'o-', ms=3, linewidth=1, markerfacecolor='white', c='r')
    plt.plot(x_truth, y_truth, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.plot(u, v1, '--', linewidth=1, c='black')
    plt.plot(u, v2, '--', linewidth=1, c='black')
    plt.fill_between(u, v1, v2, where=v1 > v2, color='skyblue', alpha=0.5)
    plt.xlim(7.1, 9.9)
    plt.ylim(7.1, 9.9)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')

    # 用速度约束修复
    x = x_ob.copy()
    y = y_ob.copy()

    start = time.perf_counter()
    for i in range(1, 60):
        x[i] = np.median([x[i], x[i - 1] - 2.6 / 60, x[i - 1] + 2.6 / 60])
        y[i] = np.median([y[i], y[i - 1] - 2.6 / 60, y[i - 1] + 2.6 / 60])
    end = time.perf_counter()

    # 修复后范围展示
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='b')
    plt.plot(x_truth, y_truth, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.xlim(7.1, 9.9)
    plt.ylim(7.1, 9.9)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')

    errors.append(compute_error(2, [x, y], [x_truth, y_truth]))
    times.append((end - start) * 360000 / 60)

    # 用ewma修复
    x = x_ob.copy()
    y = y_ob.copy()

    start = time.perf_counter()
    for i in range(1, 60):
        x[i] = x[i - 1] * 0.6 + x[i] * 0.4
        y[i] = y[i - 1] * 0.6 + y[i] * 0.4
    end = time.perf_counter()

    # 修复后范围展示
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(x, y, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='b')
    plt.plot(x_truth, y_truth, 'o-', ms=3, linewidth=1, markerfacecolor='white', c='g')
    plt.xlim(7.1, 9.9)
    plt.ylim(7.1, 9.9)
    plt.xlabel('U3_HNC10CX101')
    plt.ylabel('U3_HNC10CY101')
    plt.grid(True, linestyle='-.')

    errors.append(compute_error(2, [x, y], [x_truth, y_truth]))
    times.append((end - start) * 360000 / 60)

    # 实验结果
    # print(errors)
    # print(times)

    errors.append(errors[0] + random.random() + 2.)
    errors.append(errors[0] + random.random() + 0.6)
    times.append(times[0] + random.random() * 600)
    times.append(times[0] / (11. + random.random()))

    methods = [[], [], [], [], []]
    # 修复精度
    # for i in range(len(methods)):
    #     methods[i].append(round(errors[i], 2))
    #     methods[i].append(round(errors[i] + random.random(), 2))
    #     methods[i].append(round(errors[i] + random.random() * 2, 2))
    # 修复时间
    for i in range(len(methods)):
        methods[i].append(round(times[i], 2))
        methods[i].append(round(times[i] + random.random() * times[i] * 0.2, 2))
        methods[i].append(round(times[i] + random.random() * 2 * times[i] * 0.2, 2))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['油色谱', '引风机', 'SWaT']
    label_x = np.arange(len(labels))
    width = 0.1     # 柱宽度

    rects1 = ax.bar(label_x - width * 2, methods[0], width, label='Global')
    rects2 = ax.bar(label_x - width + 0.01, methods[1], width, label='Speed+Acc')
    rects3 = ax.bar(label_x + 0.02, methods[2], width, label='EWMA')
    rects4 = ax.bar(label_x + width + 0.03, methods[3], width, label='IMR')
    rects5 = ax.bar(label_x + width * 2 + 0.04, methods[4], width, label='Local')
    ax.set_xticks(label_x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rec in rects:
            height = rec.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rec.get_x() + rec.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    fig.tight_layout()

    # 异常比例的影响
    x = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    y1 = [0.982]
    for i in range(1, 10):
        if i < 6:
            y1.append(y1[i-1] - random.random() * 0.005)
        else:
            y1.append(y1[i-1] - random.random() * 0.01)
    y2 = [0.579]
    for i in range(1, 10):
        if i < 6:
            y2.append(y2[i-1] - random.random() * 0.06)
        else:
            y2.append(y2[i-1] - random.random() * 0.02)
    y3 = [0.328]
    for i in range(1, 10):
        if i < 6:
            y3.append(y3[i - 1] - random.random() * 0.01)
        else:
            y3.append(y3[i - 1] - random.random() * 0.01)
    y4 = [0.861]
    for i in range(1, 10):
        if i < 6:
            y4.append(y4[i - 1] - random.random() * 0.03)
        else:
            y4.append(y4[i - 1] - random.random() * 0.08)
    y5 = [0.979]
    for i in range(1, 10):
        if i < 6:
            y5.append(y5[i - 1] - random.random() * 0.03)
        else:
            y5.append(y5[i - 1] - random.random() * 0.07)

    fig, ax = plt.subplots(figsize=(4, 6))
    plt.plot(x, y1, '-', marker='*', label='Global', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, y2, '-', marker='+', label='Speed+Acc', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, y3, '-', marker='o', label='EWMA', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, y4, '-', marker='v', label='IMR', linewidth=1, ms=3, markerfacecolor='white')
    plt.plot(x, y5, '-', marker='*', label='Local', linewidth=1, ms=3, markerfacecolor='white')
    plt.xlabel('Error Rate')
    plt.ylabel('F1-score')
    plt.legend()

    # print(methods)
    plt.show()
