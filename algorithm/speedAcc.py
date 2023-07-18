import time
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error


class SpeedAcc(Cleaner):
    def clean(self):
        # 对每个序列进行修复
        # 先重新拷贝观测值
        self.dataset.modified = self.dataset.origin.copy(deep=True)
        # 记录执行时间
        start = time.perf_counter()
        for col in self.dataset.modified.columns:
            values = self.dataset.modified[col].values
            # 定义速度约束
            speed_list = np.diff(values)  # 差分
            s_min = np.mean(speed_list) - 3 * np.std(speed_list)
            s_max = np.mean(speed_list) + 3 * np.std(speed_list)
            # 定义加速度约束
            acc_list = np.diff(speed_list)  # 差分
            a_min = np.mean(acc_list) - 3 * np.std(acc_list)
            a_max = np.mean(acc_list) + 3 * np.std(acc_list)
            w = 10 # 滑动窗口长度
            for k in range(2, len(values)):
                X_k_min = set()
                X_k_max = set()
                # Compute x_k_min & x_k_max
                # x_k_min = x_k,k-1_min = max(x_k,k-1,s_min, x_k,k-1,a_min)
                x_k_min = max(values[k - 1] + s_min, a_min + (values[k - 1] - values[k - 2]) + values[k - 1])
                x_k_max = min(values[k - 1] + s_max, a_max + (values[k - 1] - values[k - 2]) + values[k - 1])
                for i in range(k + 1, len(values)):
                    if i > k + w:
                        break
                    # Compute z_k,i,a_min & z_k,i,a_max
                    z_k_i_a_min = (values[k - 1] * (i - k) - (a_min * (i - k) * (i - k) - values[i])) / (i - k + 1)
                    z_k_i_a_max = (values[k - 1] * (i - k) - (a_max * (i - k) * (i - k) - values[i])) / (i - k + 1)
                    # Compute z_k,i,s_min & z_k,i,s_max
                    z_k_i_s_min = values[i] - s_min * (i - k)
                    z_k_i_s_max = values[i] - s_max * (i - k)
                    X_k_min.add(min(z_k_i_s_min, z_k_i_a_min))
                    X_k_max.add(max(z_k_i_s_max, z_k_i_a_max))
                # Compute x_k_mid
                x_k_mid = np.median(list(X_k_min | X_k_max | {values[k]}))
                if x_k_max < x_k_mid:
                    values[k] = x_k_max
                elif x_k_min > x_k_mid:
                    values[k] = x_k_min
                else:
                    values[k] = x_k_mid
        end = time.perf_counter()
        print(round(end - start, 2), 'ms')


if __name__ == '__main__':
    fan = MTS('fan')
    # 测试SCREEN修复效果
    # 修复前
    before_fix = error(fan)

    # fan.origin.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    # 修复后
    cleaner = SpeedAcc(fan)
    cleaner.clean()

    # fan.modified.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    after_fix = error(fan)

    print(before_fix, after_fix)
