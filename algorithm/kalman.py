import time
import numpy as np

from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error
from pykalman import KalmanFilter


class Kalman(Cleaner):
    def clean(self):
        # 滤波器参数
        damping = 1

