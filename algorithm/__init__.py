import abc

import numpy as np


def error(data, order=1):
    e = 0.
    for col in data.modified.columns:
        modified = np.array(data.modified[col].values)
        clean = np.array(data.clean[col].values)
        # 消除量纲
        modified = (modified - np.min(modified)) / (np.max(modified) - np.min(modified))
        clean = (clean - np.min(clean)) / (np.max(clean) - np.min(clean))
        # 累加误差平均值
        e = e + np.linalg.norm(modified - clean, ord=order)
    return e / len(data.modified.columns)


class Cleaner:

    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def clean(self):
        pass
