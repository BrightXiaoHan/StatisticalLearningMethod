# Copyright 2020 hanbing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset


# 使用pytorch搭建搭建感知机模型
class Perceptron(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.randn(n_features, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, x):
        # y = w * x + b
        y = torch.matmul(x, self.w) + self.b
        return y

    def predict(self, x):
        # y_ = sign(w * x + b)
        y_ = self(x)
        return y_.sign()


def perceptron_loss(model, output, y):
    # loss = -sum(y * (w*x+b) / ||w||)
    index = (output * y) < 0
    loss = -((y[index] * output[index]) / torch.norm(model.w)).sum()
    # 这里可以不考虑w的正则化项
    # loss = -((y[index] * output[index]).mean()
    return loss


def train(model, x, y, epoch=20):
    x_data = torch.from_numpy(x)
    y_data = torch.from_numpy(y)
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=x_data.shape[0], shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0)

    for _ in range(epoch):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = perceptron_loss(model, model(x_batch), y_batch)
            loss.backward()
            optimizer.step()


def evaluate(model, x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    y_ = model.predict(x)
    correct = y[y == y_].shape[0]
    accuracy = correct / x.shape[0]
    return accuracy


def plot(x, y, w, b):
    x_pos = x[y == 1]
    x_neg = x[y == -1]

    surface_x = np.arange(x[:, 0].min() - 1, x[:, 0].max() + 1, 1)
    surface_y = np.arange(x[:, 1].min() - 1, x[:, 1].max() + 1, 1)

    surface_x, surface_y = np.meshgrid(surface_x, surface_y)
    surface_z = surface_x * (-w[0] / w[2]) - \
        surface_y * (-w[1] / w[2]) - b

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x_pos[:, 0], x_pos[:, 1], x_pos[:, 2],
               c="red", label='Positive Examples')
    ax.scatter(x_neg[:, 0], x_neg[:, 1], x_neg[:, 2],
               c="black", label='Negative Examples')
    ax.plot_surface(surface_x, surface_y, surface_z, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, alpha=0.5)
    ax.legend()
    plt.show()


# 使用sklearn生成随机的三维线性可分样本
x, y = make_classification(n_samples=100,  # n_samples:生成样本的数量
                           n_features=3,  # 生成样本的特征数
                           n_redundant=0,  # 冗余信息特征
                           n_informative=1,  # 多信息特征的个数
                           n_clusters_per_class=1)  # 每个特征的聚类
y[y == 0] = -1
model = Perceptron(x.shape[1])
train(model, x, y)
acc = evaluate(model, x, y)
# 当分类数据集线性可分时，可以收敛到100%准确率，若数据集本身不是线性可分，则不存在可以完全分离正负样本的超平面
print("感知机超平面分类准确率为%f" % acc)
w = model.w.data.numpy()
b = model.b.data.numpy()
plot(x, y, w, b)
