import time
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import numpy as np
import pmdarima as pm

from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error


class ARIMA(Cleaner):
    def clean(self):
        # 对每个序列进行修复
        # 先重新拷贝观测值
        self.dataset.modified = self.dataset.origin.copy(deep=True)
        # 记录执行时间
        start = time.perf_counter()
        for col in self.dataset.modified.columns:
            # 自动选参模型
            series = self.dataset.modified[col]
            n = len(series)
            series = series[:int(n*0.3)]
            model = pm.auto_arima(series,
                                  start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                                  start_P=0, seasonal=False,
                                  max_d=3, trace=True,
                                  information_criterion='aic',
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=False)
            forecast = model.predict(n - int(n*0.3))
            modified = np.concatenate((series.values, forecast.values), axis=0)
            self.dataset.modified[col] = modified
        end = time.perf_counter()
        print(end - start, 'ms')


if __name__ == '__main__':
    fan = MTS('fan')
    # 修复前误差
    print(error(fan))

    # 修复后误差
    arima = ARIMA(fan)
    arima.clean()
    print(error(fan))
