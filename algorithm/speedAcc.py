import time
import numpy as np

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
            s_min = np.mean(speed_list) - 2 * np.std(speed_list)
            s_max = np.mean(speed_list) + 2 * np.std(speed_list)
            # 定义加速度约束
            acc_list = np.diff(speed_list)  # 差分
            a_min = np.mean(acc_list) - 2 * np.std(acc_list)
            a_max = np.mean(acc_list) + 2 * np.std(acc_list)
            for i in range(2, len(values)):
                candidates = [values[i-1] + s_min, values[i-1] + s_max,
                              values[i-1] + a_min + (values[i-1] - values[i-2]),
                              values[i-1] + a_max + (values[i-1] - values[i-2]),
                              values[i]]
                values[i] = np.median(candidates)
        end = time.perf_counter()
        print(end - start, 'ms')


if __name__ == '__main__':
    fan = MTS('fan')
    # 测试SCREEN修复效果
    # 修复前误差
    print(error(fan))
    # 修复后误差
    speedAcc = SpeedAcc(fan)
    speedAcc.clean()

    print(error(fan))
