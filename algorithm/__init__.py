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
        if col == 'U3_DPI56A105':
            e = e + np.linalg.norm(delta, ord=order) / 120.
        elif col == 'U3_HNA10CT110':
            e = e + np.linalg.norm(delta, ord=order) / 2.5
        elif col == 'U3_HNC10AN001XQ01':
            e = e + np.linalg.norm(delta, ord=order) / 8.
        elif col == 'U3_HNC10CF101_F':
            e = e + np.linalg.norm(delta, ord=order) / 16.
        elif col == 'C2H4':
            e = e + np.linalg.norm(delta, ord=order) / 0.25
        elif col == 'C2H2':
            e = e + np.linalg.norm(delta, ord=order) / 0.1
        elif col == 'C2H6':
            e = e + np.linalg.norm(delta, ord=order) / 0.5
        elif col == 'CO':
            e = e + np.linalg.norm(delta, ord=order) / 50
        elif col == 'CO2':
            e = e + np.linalg.norm(delta, ord=order) / 500
        else:
            e = e + np.linalg.norm(delta, ord=order)
    return round(e / len(data.clean), 4)


class Cleaner:

    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def clean(self):
        pass
