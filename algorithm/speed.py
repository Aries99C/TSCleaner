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
            speed_list = np.diff(values)    # 差分
            s_min = np.mean(speed_list) - 3 * np.std(speed_list)
            s_max = np.mean(speed_list) + 3 * np.std(speed_list)
            # print('min speed: ', s_min)
            # print('max speed: ', s_max)
            for i in range(1, len(values)):
                if values[i] > values[i-1] + s_max:
                    values[i] = values[i-1] + s_max
                elif values[i] < values[i-1] + s_min:
                    values[i] = values[i - 1] + s_min
        end = time.perf_counter()
        print(end - start, 'ms')


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
