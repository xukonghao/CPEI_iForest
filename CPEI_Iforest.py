# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 00:22:12 2021

@author: 文柯力
@improve: 许孔豪
"""
import numpy as np
import pandas as pd
import random


def C(psi):
    """
    异常分数
    anomaly score 
    why C? 为什么用这个公式呢？
    """
    H = lambda _psi: np.log(_psi) + 0.5772156649
    if psi > 2:
        return 2 * H(psi - 1) - 2 * (psi - 1) / psi
    elif psi == 2:
        return 1
    return 0


def generate_batch(n, batch_size,random_state=None):
    random.seed(random_state)
    batch_index = random.sample(range(n), batch_size)
    return batch_index

class LeafNode:
    """ 叶子 """

    def __init__(self, size, data):
        # services for pathLength calculate below. 
        self.size = size
        self.data = data


class DecisionNode:
    """ split node  """

    def __init__(self, left, right, splitAtt, splitVal):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitVal = splitVal


class iTree:
    def __init__(self, height, height_limit,random_state = None):
        self.height = height
        self.height_limit = height_limit
        self.split_feature = None
        self.random_state = random_state

    def fit(self, X: np.ndarray):
        if self.height >= self.height_limit or X.shape[0] <= 1:
            self.root = LeafNode(X.shape[0], X)
            return self.root
            # Choose Random Split Attributes and Value
        num_features = X.shape[1]
        # Note the high is exclusive
        self.random_state = generate_batch(1000000000, self.height+1,self.random_state)[self.height]
        random.seed(self.random_state)
        splitAtt = random.randint(0, num_features-1)
        # splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))
        self.split_feature = splitAtt
        splitVal = self.evaluate_split(X)

        # Make X_left and X_right
        X_left = X[X[:, splitAtt] < splitVal]
        X_right = X[X[:, splitAtt] >= splitVal]

        random_state_left = generate_batch(2000000000, self.height + 1, self.random_state)[self.height]
        random_state_right = generate_batch(1500000000, self.height + 1, self.random_state)[self.height]
        left = iTree(self.height + 1, self.height_limit,random_state_left)
        right = iTree(self.height + 1, self.height_limit,random_state_right)
        left.fit(X_left)
        right.fit(X_right)

        self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)
        # i don't know why we need to count it 
        # self.n_nodes = self.count_nodes(self.root)
        return self.root

    def evaluate_split(self, X):
        if len(X) < 6:
            spile_value = np.median(X[:, self.split_feature])
            return spile_value

        self.split_value = np.median(X[:, self.split_feature])
        left_X = X[X[:, self.split_feature] < self.split_value]
        right_X = X[X[:, self.split_feature] >= self.split_value]

        if len(left_X) == 0 or len(right_X) == 0:
            return self.split_value

        score_middle = self.score(X)

        self.split_value = np.median(left_X[:, self.split_feature])
        score_left = self.score(left_X)

        self.split_value = np.median(right_X[:, self.split_feature])
        score_right = self.score(right_X)

        if score_left == score_right or (score_middle > score_left and score_middle > score_right):
            self.split_value = np.median(X[:, self.split_feature]) #直接输出分割点
            return self.split_value
        elif score_middle < score_left:
            return self.evaluate_split(left_X) # 左分割
        else:
            return self.evaluate_split(right_X) #右分割

    def score(self, X):
        left_X = X[X[:, self.split_feature] < self.split_value]
        right_X = X[X[:, self.split_feature] >= self.split_value]
        if len(left_X) == 0 or len(right_X) == 0:
            return 0

        left_mean = left_X.mean(axis=0)[self.split_feature]
        right_mean = right_X.mean(axis=0)[self.split_feature]
        mean_diff = np.abs(left_mean - right_mean)

        left_std = left_X.std(axis=0)[self.split_feature]
        right_std = right_X.std(axis=0)[self.split_feature]
        std_sum = left_std + right_std
        if std_sum==0:
            return float('inf')

        # score = mean_diff + std_sum
        # score = np.abs(mean_diff - std_sum)
        score = mean_diff
        # print("mean_diff:",mean_diff," std_sum:",std_sum," score:",score)
        return score

class IsolationTreeEnsemble:
    def __init__(self, sample_size=256, n_trees=10,random_state = None):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.random_state = random_state

    def fit(self, X: np.ndarray):
        """
        Given a 2D matrix of observations, creat an ensemble of IsolationTree objects
        and store them in a list: self.trees. Convert DataFrames to ndarray objects.
        给定一个2D观测矩阵，创建一个IsolationTree对象集合,并将它们存储在列表中：self.trees。将数据帧转换为ndarray对象。
        """
        self.trees = []
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_rows = X.shape[0]
        height_limit = np.ceil(np.log2(self.sample_size))

        for i in range(self.n_trees):
            # using Bagging methods. 
            # do u think about that if smaple_size greater than n_rows ?
            self.random_state=generate_batch(100000000,i+1,self.random_state)[i]
            np.random.seed(self.random_state)
            data_index = np.random.randint(0, n_rows, self.sample_size)
            X_sub = X[data_index]
            tree = iTree(0, height_limit,random_state=self.random_state)
            tree.fit(X_sub)
            self.trees.append(tree)

        # allow chaning 
        return self

    def PathLength(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  
        Compute the path length for x_i using every tree in self.trees then 
        compute the average for each x_i.  Return an ndarray of shape (len(X),1).
        给定观测值的2D矩阵X，计算平均路径长度,对于X中的每个观察。使用self.trees中的每棵树计算x_i的路径长度，
        然后计算每个x_i的平均值。返回形状为（len（X），1）的ndarray。
        """
        paths = []
        for row in X:
            avg_path, cnt = 0, 0
            for tree in self.trees:
                node = tree.root
                cur_length = 0
                while isinstance(node, DecisionNode):
                    if (row[node.splitAtt] < node.splitVal):
                        node = node.left
                    else:
                        node = node.right
                    cur_length += 1

                # The adjustment(`C(leaf_size)`) accounts for an unbuilt
                # subtree beyond the tree height limit. 
                leaf_size = node.size
                pathLength = cur_length + C(leaf_size)

                # dynamic updates the average pathLength 
                avg_path = (avg_path * cnt + pathLength) / (cnt + 1)
                cnt += 1
            paths.append(avg_path)

        return np.array(paths)

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the 
        anomaly score for each x_i observaton, returning an 
        ndarry of them.
        给定观测值的2D矩阵X，计算每个x_i观测的异常分数，返回他们中的ndarry。
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = self.PathLength(X)
        scores = np.array([1-np.power(2, -ech / C(self.sample_size)) for ech in avg_length])
        return scores

    def predict(self, X: np.ndarray, threshold=0.5):
        """
        Givend an 2D matrix of observations, X 
        using anomaly_score() to calculate scores and 
        return an array of prediction: 1 for any score >= threshold and 0 otherwise.
        u can see that the default value of threshold is 0.5
        给出观测值的二维矩阵，X使用anomaly_score（）计算分数,返回一个预测数组：任何分数>=阈值为1，否则为0。可以看到阈值的默认值是0.5
        """
        scores = self.anomaly_score(X)
        prediction = np.array(
            [1 if s >= threshold else 0 for s in scores]
            # [0 if s >= threshold else 1 for s in scores]
        )
        return prediction

