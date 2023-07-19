import time
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error


class SCREEN(Cleaner):
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
            w = 10  # 滑动窗口长度
            for k in range(len(values)):
                X_k_min = set()
                X_k_max = set()
                x_k_min = values[k-1] + s_min if k > 0 else -np.inf
                x_k_max = values[k-1] + s_max if k > 0 else np.inf
                for i in range(k + 1, len(values)):
                    if i > k + w:
                        break
                    X_k_min.add(values[i] + s_min * (k - i))
                    X_k_max.add(values[i] + s_max * (k - i))
                # print(X_k_min | X_k_max | {values[k]})
                x_k_mid = np.median(list(X_k_min | X_k_max | {values[k]}))
                # print(x_k_mid)
                if x_k_max < x_k_mid:
                    values[k] = x_k_max
                elif x_k_min > x_k_mid:
                    values[k] = x_k_min
                else:
                    values[k] = x_k_mid
        end = time.perf_counter()
        print(round(end - start, 2), 'ms')

        return round(end - start, 2), error(self.dataset)


if __name__ == '__main__':
    fan = MTS('fan')
    # 测试SCREEN修复效果
    # 修复前
    before_fix = error(fan)

    # fan.origin.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    # 修复后
    cleaner = SCREEN(fan)
    cleaner.clean()

    # fan.modified.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    after_fix = error(fan)

    print(before_fix, after_fix)
