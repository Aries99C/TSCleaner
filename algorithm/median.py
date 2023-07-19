import time
import scipy.signal as signal
from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error


class MedianFilter(Cleaner):
    def clean(self):
        # 先重新拷贝观测值
        self.dataset.modified = self.dataset.origin.copy(deep=True)
        # 记录执行时间
        start = time.perf_counter()
        w = 9  # 滑动窗口长度
        for col in self.dataset.modified.columns:
            values = self.dataset.modified[col].values
            modified = signal.medfilt(values, w)
            self.dataset.modified[col] = modified

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
    cleaner = MedianFilter(fan)
    cleaner.clean()

    # fan.modified.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    after_fix = error(fan)

    print(before_fix, after_fix)
