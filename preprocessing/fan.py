import pandas as pd


if __name__ == '__main__':
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

    fan.to_csv('../data/fans/fan.csv', index_label='timestamp')
