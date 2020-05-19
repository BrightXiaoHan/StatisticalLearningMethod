# Copyright (c) 2020 hanbing
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# %%
import heapq
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.datasets import make_classification


class Node(object):
    """KDTree的节点
    """

    def __init__(self, value=None, split=None, left=None, right=None):
        """Node 构造方法

        Arguments:
            value {numpy.ndarray} -- 该节点上的值
            left {Node} -- 左子数根节点
            right {Node} -- 又子树根节点
        """
        self.value = value  # (x, y)
        self.split = split  # (dim, value)
        self.left = left
        self.right = right

    def is_leaf(self):
        """判断当前节点是否为叶子节点

        Returns:
            bool -- True：是叶子节点 False：不是叶子节点
        """
        return self.left is None and self.left is None


class KNN(object):

    def __init__(self,
                 topk=1,
                 p=2):
        """注，本代码仅实现了 p=2 的情况，即以欧式距离作为度量

        Keyword Arguments:
            topk {int} -- 以最近邻topk结果作为分类标准 (default: {1})
            p {int} -- 距离度量洗漱，p=2时为欧式距离 (default: {2})
        """
        self.k = topk
        self.p = p
        self.kdtree = None
        self.nodes = None
        self.x = None
        self.y = None

    def fit(self, x, y):
        """学习数据特征，构造KNN分类器

        Arguments:
            x {[type]} -- 输入数据 (n, dim)
            y {[type]} -- 分类标签
        """
        nodes = []
        assert x.shape[0] > self.k, "训练数据数量必须大于k"

        def construct(x, y, depth):
            if (len(x) == 0):
                return Node()
            i = depth % x.shape[1]
            x_i = x[:, i]
            sorted_index = x_i.argsort()
            mid_point = x.shape[0] // 2

            mid_index = sorted_index[mid_point]
            left_index = sorted_index[:mid_point]
            right_index = sorted_index[mid_point+1:]

            node = Node((x[mid_index], y[mid_index]),
                        (i, x[mid_index][i]),
                        construct(x[left_index], y[left_index], depth+1),
                        construct(x[right_index], y[right_index], depth+1)
                        )
            nodes.append(node)
            return node
        self.kdtree = construct(x, y, 0)

        # 记录所有节点
        self.nodes = nodes

        # 存储训练数据
        self.x = x
        self.y = y

    def predict(self, x):
        """给定数据，搜索topk的节点

        Arguments:
            x {np.ndarray} -- 待预测向量

        Returns:
            list -- 元素结构为 （distance, y) list按照大根堆顺序排列
        """
        if self.kdtree is None:
            raise RuntimeError("调用predict之前请调用fit方法进行训练")

        # heap 中的元素格式为 (distance, y)即距离目标点的距离，和当前距离下目标点的分类
        heap = []

        def l2distance(x1, x2):
            # 计算两点之间的欧式距离
            return np.linalg.norm(x1 - x2)

        def traverse(root):
            # 从根节点开始递归的查找，根据p在节点的左边还是右边，决定递归方向
            # 若到达叶节点,则将其作为当前最优节点
            node = root
            path = []
            while (not node.is_leaf()):
                path.append(node)
                dim, value = node.split  # 获取当前节点分割的维度和分割点
                if x[dim] < value:
                    node = node.left
                else:
                    node = node.right

            path.reverse()

            for node in path:
                distance = l2distance(x, node.value[0])

                if len(heap) < self.k:
                    heapq.heappush(heap, (distance, node))
                elif distance < heap[0][0]:
                    heapq.heappushpop(heap, (distance, node))

                tmpx, tmpy = heap[0][1].value
                tmp_distance = heap[0][0]
                dim = node.split[0]
                if (abs(x[dim] - tmpx[dim]) < tmp_distance):
                    traverse(node.left) if node.right in path else traverse(
                        node.right)

        traverse(self.kdtree)

        # 使用投票的方式进行分类，这里也可以用加权距离，核方法等进行分类
        catagory = [i[1].value[1] for i in heap]
        catagory = Counter(catagory).most_common(1)[0][0]
        return catagory


def plot_knn_2d(model):
    assert model.x.shape[1] == 2, "plot_knn_2d只能绘制特征维度为2的数据"

    # 计算分割线坐标范围
    left_most = model.x[:, 0].min()
    right_most = model.x[:, 0].max()
    width = right_most - left_most
    left_most -= width / 5
    right_most += width / 5
    bottom_most = model.x[:, 1].min()
    top_most = model.x[:, 1].max()
    height = top_most - bottom_most
    bottom_most -= height / 5
    top_most += height / 5

    def plot(node, left_most, right_most, bottom_most, top_most):

        if node is None:
            return

        if node.is_leaf():
            return
        dim, value = node.split
        x, y = node.value
        start_point = [left_most, value] if dim == 1 else [
            value, bottom_most]
        end_point = [right_most, value] if dim == 1 else [
            value, top_most]
        plt.plot([start_point[0], end_point[0]],
                 [start_point[1], end_point[1]], c='black', alpha=0.5)

        if dim == 0:
            plot(node.left, left_most, value, bottom_most, top_most)
            plot(node.right, value, right_most, bottom_most, top_most)
        else:
            plot(node.left, left_most, right_most, bottom_most, value)
            plot(node.right, left_most, right_most, value, top_most)

    plot(model.kdtree, left_most, right_most, bottom_most, top_most)

    plt.scatter(model.x[model.y == 0, 0],
                model.x[model.y == 0, 1], label="Negative Example")
    plt.scatter(model.x[model.y == 1, 0],
                model.x[model.y == 1, 1], label="Positive Example")
    plt.legend()
    plt.show()


# 使用sklearn生成随机的三维线性可分样本
x, y = make_classification(n_samples=10,  # n_samples:生成样本的数量
                           n_features=2,  # 生成样本的特征数
                           n_redundant=0,  # 冗余信息特征
                           n_informative=1,  # 多信息特征的个数
                           n_clusters_per_class=1)  # 每个特征的聚类

# 训练knn
knn = KNN(1)
knn.fit(x, y)

# 画出生成的kd树
plot_knn_2d(knn)

# 预测结果
knn.predict(np.array([1., 1.]))
