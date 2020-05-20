# Copyright (c) 2020 hanbing
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import numpy as np


class NativeBayes(object):

    def __init__(self, laplace=1):
        """初始化朴素贝叶斯分类器

        Keyword Arguments:
            laplace {int} -- 拉普拉斯平滑系数 (default: {1})
        """
        self.laplace = laplace
        self.catagories = None  # 存储所有类别
        self.priori = None  # 先验概率 p(y)
        self.conditional_p = None  # 条件概率 p(X|y)

    def fit(self, X, y, X_space):
        """拟合数据

        Arguments:
            X {np.ndarray} -- 输入数据
            y {np.ndarray} -- 输出标签
            X_space {np.ndarray} -- 输入参数空间
        """
        self.catagories, count = np.unique(y, return_counts=True)

        # 计算先验概率，使用概率的对数进行存储，计算乘法可以简化为计算加法
        self.priori = np.log(count / count.sum())

        self.conditional_p = dict()

        for c in self.catagories:
            x_c = X[y == c]  # 选出所有标签为 c 的输入样本

            for i, (column, space) in enumerate(zip(x_c.T, X_space)):
                values, counts = np.unique(column, return_counts=True)
                for item in space:
                    if item in values:
                        index = np.argwhere(values == item)[0][0]
                        count = counts[index]
                    else:
                        count = 0
                    #  计算条件概率 (用概率的对数表示，计算乘法可以简化为计算加法)
                    logp = np.log((count + self.laplace) /
                                  (x_c.shape[0] +
                                   space.shape[0] * self.laplace))

                    self.conditional_p[(i, item, c)] = logp

    def predict(self, X):
        # 计算每个类别对应的后验概率
        posteriors = np.empty(len(self.catagories))
        for index, c in enumerate(self.catagories):
            p = self.priori[index]
            for i, x in enumerate(X):
                p += self.conditional_p[(i, x, c)]
            posteriors[index] = p

        # 取后验概率最大值对应的类别作为分类类别
        argmax = posteriors.argmax()
        return self.catagories[argmax]


if __name__ == "__main__":
    # 书中的例4.2
    X = np.array([[1, "S"],
                  [1, "M"],
                  [1, "M"],
                  [1, "S"],
                  [1, "S"],
                  [2, "S"],
                  [2, "M"],
                  [2, "M"],
                  [2, "L"],
                  [2, "L"],
                  [3, "L"],
                  [3, "M"],
                  [3, "M"],
                  [3, "L"],
                  [3, "L"]])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    X_space = np.array([[1, 2, 3], ["M", "S", "L"]])

    model = NativeBayes()
    model.fit(X, y, X_space)
    y_ = model.predict(np.array([2, "S"]))
    print(y_)
