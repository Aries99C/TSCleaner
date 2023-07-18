import time
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Cleaner
from entity.timeseries import MTS
from algorithm import error


class IMR(Cleaner):
    def __init__(self, dataset):
        super().__init__(dataset)

        # IMR参数
        self.p = 3
        self.delta = 0.1
        self.maxNum = 1000
        self.MINVAL = np.inf
        self.MAXVAL = -np.inf

        # 标注信息
        self.label_list = None
        self.labels = None

    def learnParamsOLS(self, xMatrix, yMatrix):
        phi = np.zeros((self.p, 1))

        xMatrixT = xMatrix.T

        middleMatrix = xMatrixT.dot(xMatrix)
        phi = np.linalg.pinv(middleMatrix).dot(xMatrixT).dot(yMatrix)

        return phi

    def combine(self, phi, xMatrix):
        yhatMatrix = xMatrix.dot(phi)

        return yhatMatrix

    def repairAMin(self, yhatMatrix, yMatrix):
        rowNum = yhatMatrix.shape[0]
        residualMatrix = yhatMatrix - yMatrix

        aMin = self.MINVAL
        target_index = -1
        yhat = None
        yhat_abs = None

        for i in range(rowNum):
            if self.label_list[i + self.p]:
                continue
            if abs(residualMatrix[i, 0] < self.delta):
                continue

            yhat = yhatMatrix[i, 0]
            yhat_abs = abs(yhat)

            if yhat_abs < aMin:
                aMin = yhat_abs
                target_index = i

        return target_index

    def clean(self):
        # 先重新拷贝观测值
        self.dataset.modified = self.dataset.origin.copy(deep=True)
        # 记录执行时间
        start = time.perf_counter()
        for col in self.dataset.modified.columns:
            # 获取标注信息
            self.label_list = self.dataset.isLabel[col].values
            self.labels = self.dataset.label[col].values
            # 获取待修复序列
            modified = self.dataset.modified[col].values

            size = len(modified)
            rowNum = size - self.p

            # form z
            zs = []
            for i in range(size):
                zs.append(self.labels[i] - modified[i])

            # build x,y for params estimation
            x = np.zeros((rowNum, self.p))
            y = np.zeros((rowNum, 1))
            for i in range(rowNum):
                y[i, 0] = zs[self.p + i]
                for j in range(self.p):
                    x[i, j] = zs[self.p + i - j - 1]

            # iteration
            index = -1
            xMatrix = np.matrix(x)
            yMatrix = np.matrix(y)
            iterationNum = 0
            val = 0

            phi = None
            while True:
                iterationNum += 1

                phi = self.learnParamsOLS(xMatrix, yMatrix)

                yhatMatrix = self.combine(phi, xMatrix)

                index = self.repairAMin(yhatMatrix, yMatrix)

                if index == -1:
                    break

                val = yhatMatrix[index, 0]
                # update y
                yMatrix[index, 0] = val
                # update x
                for j in range(self.p):
                    i = index + 1 + j
                    if i >= rowNum:
                        break
                    if i < 0:
                        continue

                    xMatrix[i, j] = val

                # 迭代控制
                if iterationNum > self.maxNum:
                    break

            print('Stop after {0} iterations'.format(iterationNum))

            # 修复
            for i in range(size):
                if self.label_list[i]:
                    modified[i] = self.labels[i]
                else:
                    modified[i] = modified[i] + yMatrix[i - self.p, 0]

        end = time.perf_counter()
        print(round(end - start, 2), 'ms')


if __name__ == '__main__':
    fan = MTS('fan')
    # 测试SCREEN修复效果
    # 修复前
    before_fix = error(fan)

    # fan.origin.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    # 修复后
    cleaner = IMR(fan)
    cleaner.clean()

    # fan.modified.plot(subplots=True, figsize=(20, 30))
    # plt.show()

    after_fix = error(fan)

    print(before_fix, after_fix)

