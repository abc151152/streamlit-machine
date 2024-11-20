import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

st.title("XGBoost")
st.markdown("""
     随机森林的基础是决策树。\\N
     决策树是一种树状结构，每个节点表示一个特征，每个叶子节点表示一个类别或一个数值。\\N
     学习过程是递归的，根据选择的特征将数据划分成子集，直到达到停止条件。   \\N
     随机性引入  •  随机抽样： 针对每个决策树的训练集，从原始数据集中进行随机抽样（有放回抽样），形成不同的训练子集。这使得每棵树的训练集都是略有不同的。  •  随机特征选择：在每次决策树的节点划分时，随机选择一个特征进行划分。这防止了某个特定特征对模型的过度依赖。     \n 
     Bootstrap Aggregating (Bagging)  •  针对每个随机抽样得到的训练子集，训练一个独立的决策树。 \n
     •  预测时，对所有决策树的输出取平均（回归问题）或进行投票（分类问题）。     预测  •  对于回归问题，将所有决策树的预测结果取平均。  \n
     •  对于分类问题，进行投票，选择得票最多的类别作为最终预测。
""")


st.header('示例代码')
st.code("""
    class XGBoostFromScratch:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, lambda_reg=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.trees = []
        self.losses = []  # 新增一个用于保存每轮损失的列表

    def fit(self, X, y):
        n_samples = X.shape[0]
        # 初始化预测值为0
        y_pred = np.zeros(n_samples)
        
        # 迭代生成树
        for _ in range(self.n_estimators):
            # 计算梯度和Hessian
            grad = gradient(y, y_pred)
            hess = hessian(y, y_pred)

            # 构建一颗树并拟合
            tree = self.build_tree(X, grad, hess, depth=0)
            self.trees.append(tree)
            
            # 更新预测值
            y_pred += self.learning_rate * self.predict_tree(tree, X)
            
            # 每轮迭代后计算当前的损失，并保存
            current_loss = squared_loss(y, y_pred)
            self.losses.append(current_loss)

    def build_tree(self, X, grad, hess, depth):
        # 这里我们简单处理为单层分裂的树，后续可扩展为多层
        n_samples, n_features = X.shape
        best_split = None
        best_gain = -float('inf')
        
        # 遍历所有特征，找到最佳分裂点
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if len(left_mask) == 0 or len(right_mask) == 0:
                    continue

                # 计算增益
                gain = self.compute_gain(grad, hess, left_mask, right_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)

        # 计算叶子节点权重
        if best_split is not None:
            feature_idx, threshold = best_split
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            left_weight = self.compute_leaf_weight(grad[left_mask], hess[left_mask])
            right_weight = self.compute_leaf_weight(grad[right_mask], hess[right_mask])

            return {"split_feature": feature_idx, "threshold": threshold,
                    "left_weight": left_weight, "right_weight": right_weight}
        else:
            return None

    def compute_gain(self, grad, hess, left_mask, right_mask):
        G_L, H_L = np.sum(grad[left_mask]), np.sum(hess[left_mask])
        G_R, H_R = np.sum(grad[right_mask]), np.sum(hess[right_mask])
        gain = 0.5 * (G_L**2 / (H_L + self.lambda_reg) + G_R**2 / (H_R + self.lambda_reg))
        return gain

    def compute_leaf_weight(self, grad, hess):
        return -np.sum(grad) / (np.sum(hess) + self.lambda_reg)

    def predict_tree(self, tree, X):
        predictions = np.zeros(X.shape[0])
        if tree is not None:
            feature_idx, threshold = tree["split_feature"], tree["threshold"]
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            predictions[left_mask] = tree["left_weight"]
            predictions[right_mask] = tree["right_weight"]
        return predictions

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * self.predict_tree(tree, X)
        return y_pred
""")



# 训练模型
if st.button("开始训练"):
    np.random.seed(42)

    # 生成虚拟数据集
    X = np.random.rand(1000, 5)  # 1000个样本，5个特征
    true_weights = np.array([2.5, -1.7, 0.5, 1.2, -0.9])
    y = X.dot(true_weights) + np.random.randn(1000) * 0.5  # 线性关系加噪声

    # 创建DataFrame以便后续处理
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 6)])
    data['target'] = y

    # 打印数据集前5行
    print(data.head())

    # 定义平方损失函数
    def squared_loss(y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    # 梯度和Hessian计算
    def gradient(y_true, y_pred):
        return -(y_true - y_pred)

    def hessian(y_true, y_pred):
        return np.ones_like(y_true)

    class XGBoostFromScratch:
        def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, lambda_reg=1):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.lambda_reg = lambda_reg
            self.trees = []
            self.losses = []  # 新增一个用于保存每轮损失的列表

        def fit(self, X, y):
            n_samples = X.shape[0]
            # 初始化预测值为0
            y_pred = np.zeros(n_samples)
            
            # 迭代生成树
            for _ in range(self.n_estimators):
                # 计算梯度和Hessian
                grad = gradient(y, y_pred)
                hess = hessian(y, y_pred)

                # 构建一颗树并拟合
                tree = self.build_tree(X, grad, hess, depth=0)
                self.trees.append(tree)
                
                # 更新预测值
                y_pred += self.learning_rate * self.predict_tree(tree, X)
                
                # 每轮迭代后计算当前的损失，并保存
                current_loss = squared_loss(y, y_pred)
                self.losses.append(current_loss)

        def build_tree(self, X, grad, hess, depth):
            n_samples, n_features = X.shape
            best_split = None
            best_gain = -float('inf')
            
            # 遍历所有特征，找到最佳分裂点
            for feature_idx in range(n_features):
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = ~left_mask
                    
                    if len(left_mask) == 0 or len(right_mask) == 0:
                        continue

                    # 计算增益
                    gain = self.compute_gain(grad, hess, left_mask, right_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_idx, threshold)

            # 计算叶子节点权重
            if best_split is not None:
                feature_idx, threshold = best_split
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                left_weight = self.compute_leaf_weight(grad[left_mask], hess[left_mask])
                right_weight = self.compute_leaf_weight(grad[right_mask], hess[right_mask])

                return {
                    "split_feature": feature_idx,
                    "threshold": threshold,
                    "left_weight": left_weight,
                    "right_weight": right_weight
                }
            else:
                return None

        def compute_gain(self, grad, hess, left_mask, right_mask):
            G_L, H_L = np.sum(grad[left_mask]), np.sum(hess[left_mask])
            G_R, H_R = np.sum(grad[right_mask]), np.sum(hess[right_mask])
            gain = 0.5 * (G_L**2 / (H_L + self.lambda_reg) + G_R**2 / (H_R + self.lambda_reg))
            return gain

        def compute_leaf_weight(self, grad, hess):
            return -np.sum(grad) / (np.sum(hess) + self.lambda_reg)

        def predict_tree(self, tree, X):
            predictions = np.zeros(X.shape[0])
            if tree is not None:
                feature_idx, threshold = tree["split_feature"], tree["threshold"]
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                predictions[left_mask] = tree["left_weight"]
                predictions[right_mask] = tree["right_weight"]
            return predictions

        def predict(self, X):
            y_pred = np.zeros(X.shape[0])
            for tree in self.trees:
                y_pred += self.learning_rate * self.predict_tree(tree, X)
            return y_pred

    model = XGBoostFromScratch(n_estimators=10, learning_rate=0.1)
    model.fit(X, y)

    # 预测结果
    y_pred = model.predict(X)

    # 绘制数据分析图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 图1：实际值与预测值对比
    axes[0, 0].scatter(y, y_pred, color='blue')
    axes[0, 0].set_title("Actual vs Predicted")
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")

    # 图2：误差分布
    error = y - y_pred
    axes[0, 1].hist(error, bins=20, color='green')
    axes[0, 1].set_title("Error Distribution")

    # 图3：特征对目标值的相关性
    for i in range(5):
        axes[1, 0].scatter(data[f'feature_{i+1}'], data['target'], label=f'Feature {i+1}')
    axes[1, 0].set_title("Feature vs Target")
    axes[1, 0].legend()

    # 图4：损失随迭代次数的变化
    axes[1, 1].plot(range(1, len(model.losses) + 1), model.losses, color='red')
    axes[1, 1].set_title("Loss vs Iterations")
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("Loss")

    plt.tight_layout()
    st.pyplot(plt)