import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mindoptpy import *
from utils import root_path


if __name__ == '__main__':
    data = pd.read_csv(root_path() + '/data/SWaT_Dataset_Attack_v0.csv', sep=',', index_col=0)
    data.drop(columns=[
        ' MV101', 'P101', 'P102', ' MV201', ' P201', ' P202', 'P203', ' P204', 'P205', 'P206',
        'MV301', 'MV302', ' MV303', 'MV304', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401',
        'P501', 'P502', 'P601', 'P602', 'P603'
    ], inplace=True)

    # 填充缺失值
    for col in data.columns:
        data[col].replace(0, np.nan, inplace=True)
    data.interpolate(method='linear', inplace=True)

    data['label'] = [0 if x == 'Normal' else 1 for x in data['Normal/Attack'].values]
    data.drop(columns=['Normal/Attack'], inplace=True)

    data = data[:5000]

    data.plot(subplots=True, figsize=(10, 16))
    plt.show()
