import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 1. 修复时间戳
    print('修复引风机数据时间戳')

    # 读取数据
    fanA = pd.read_csv('../data/fans/A.csv', sep=',')
    fanB = pd.read_csv('../data/fans/B.csv', sep=',')

    # 重命名时间戳列
    fanA.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    fanB.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)

    # 长度对齐
    fanB = fanB[:-4]

    # 合并
    fan = pd.merge(fanA, fanB, how='inner', on='timestamp', sort=False)

    # 时间戳为索引
    fan['timestamp'] = pd.to_datetime(fan['timestamp'], format='%Y-%m-%d|%H:%M:%S.%f')
    fan.set_index('timestamp', inplace=True)

    # 2. 数据预处理
    print('引风机数据预处理')

    # 过滤模拟0-1信号
    fan.drop(
        columns=[
            'U3_HNV10CT111XH01',
            'U3_HNV10CT111XH52',
            'U3_HNV20CF001',
            'U3_HNV20CF002',
            'U3_HNV20CL001XH52',
            'U3_HNV20CL001XH54',
            'U3_HNV20CP101XH01',
            'U3_HNV20CP101XH52',
            'U3_HNV20CP102XH52',
            'U3_HNV20CP103XH01',
            'U3_HNV20CT111XH01',
            'U3_HNV20CT111XH52',
            'U3_HNV10CP103'],
        inplace=True
    )

    # 过滤无关信号
    fan.drop(
        columns=[
            'U3_HNA10CP107',
            'U3_HNA10CT110',
            'U3_HNV10CP101',
            'U3_HNV10CP102',
            'U3_HNV20CP101',
            'U3_HNV20CP102',
            'U3_HNV20CP103',
            'U3_HNV20CT111',
            'U3_HNC10CT151',
            'U3_HNC10CT152',
            'U3_HNC10CT161',
            'U3_HNC10CT162',
            'U3_HNA10CP106'
        ],
        inplace=True
    )

    # 填充缺失值
    fan.replace(0, np.nan, inplace=True)
    fan.interpolate(method='time', inplace=True)

    # 保留概念一致的数据
    fan = fan[int(len(fan) * 0.05): int(len(fan) * 0.3)]

    # 3. 人工清洗数据


    # 存储文件
    print('引风机数据预处理完毕')
    fan.to_csv('../data/fans/fan.csv', index_label='timestamp')

    # 3. 绘图观察
    fan.plot(subplots=True, figsize=(20, 30))
    plt.show()
