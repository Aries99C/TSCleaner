import abc

import numpy as np


def error(data, order=1):
    e = 0.
    for col in data.modified.columns:
        modified = np.array(data.modified[col].values)
        clean = np.array(data.clean[col].values)
        # 累加误差平均值
        delta = modified - clean
        if np.std(delta) == 0.:
            continue
        e = e + np.linalg.norm(delta, ord=order) / np.mean(clean)
    return e / len(data.clean)


class Cleaner:

    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def clean(self):
        pass
