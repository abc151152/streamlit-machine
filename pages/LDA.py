import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

# Streamlit 标题和描述
st.title("LSTM 时间序列预测")
st.markdown("""
    Adam是一种自适应学习率的优化算法，结合了动量和自适应学习率的特性。\n
    主要思想是根据参数的梯度来动态调整每个参数的学习率。\n
    核心原理包括：\n
    1. 动量（Momentum  ）：Adam算法引入了动量项，以平滑梯度更新的方向。这有助于加速收敛并减少震荡。\n
    2. 自适应学习率：Adam算法计算每个参数的自适应学习率，允许不同参数具有不同的学习速度。 \n
    3. 偏差修正（Bias Correction）：Adam算法在初期迭代中可能受到偏差的影响，因此它使用偏差修正来纠正这个问题。
""")


st.header('示例代码')
st.code("""
     # 计算类内散布矩阵 SW
    SW = np.zeros((5, 5))
    for i, mean_vec in enumerate(mean_vectors):
        class_scatter = np.cov(X[y == i].T) * (X[y == i].shape[0] - 1)
        SW += class_scatter

    # 计算类间散布矩阵 SB
    SB = np.zeros((5, 5))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i].shape[0]
        mean_diff = (mean_vec - mean_overall).reshape(5, 1)
        SB += n * (mean_diff).dot(mean_diff.T)

    # 求解广义特征值问题，获取投影方向
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))

    # 对特征值进行排序，选择前两个特征向量（降维到2维）
    sorted_indices = np.argsort(eigvals)[::-1]
    w = eigvecs[:, sorted_indices[:2]]  # 取前2个特征向量

    # 投影数据
    X_lda = X.dot(w)

    # 可视化
    # 可视化
    plt.figure(figsize=(14, 10))

    # 原始数据分布
    plt.subplot(2, 3, 1)
    plt.title("Original High-dimensional Data Distribution")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # 类内散布矩阵可视化
    plt.subplot(2, 3, 2)
    plt.title("Class Means and Within-class Scatter")
    for i in range(3):
        sns.scatterplot(x=X[y == i][:, 0], y=X[y == i][:, 1], label=f"Class {i}")
        plt.scatter(*mean_vectors[i][:2], marker='x', s=100, c="black")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # 类间散布矩阵可视化
    plt.subplot(2, 3, 3)
    plt.title("Between-class Scatter")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")
    plt.scatter(*mean_overall[:2], marker="o", s=150, c="red", label="Overall Mean")
    for i, mean_vec in enumerate(mean_vectors):
        plt.plot([mean_overall[0], mean_vec[0]], [mean_overall[1], mean_vec[1]], 'k--')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # 投影后的数据分布
    plt.subplot(2, 3, 4)
    plt.title("Data Distribution after LDA Projection (2D)")
    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette="Set1")
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")

    # 高维数据和低维数据的散点图对比
    plt.subplot(2, 3, 5)
    plt.title("High-dim Data vs Low-dim Data (Projection Comparison)")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", alpha=0.5)
    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette="Set1", marker="X", s=70)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # LDA如何有效地将高维数据映射到2维
    plt.subplot(2, 3, 6)
    plt.title("Projection Path Analysis")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")
    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette="Set1", marker="X", s=70)
    for i in range(3):
        plt.annotate(f"Class {i}", (X_lda[y == i][:, 0].mean(), X_lda[y == i][:, 1].mean()))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.show()
""")



# 训练模型
if st.button("开始训练"):
    np.random.seed(0)
    # 类别1
    mean1, cov1 = [2, 2, 3, 1, 1], np.diag([1, 0.5, 0.8, 1.2, 0.9])  # 类别1
    mean2, cov2 = [5, 6, 1, 4, 2], np.diag([1, 0.5, 0.7, 1.3, 1.0])  # 类别2
    mean3, cov3 = [8, 7, 9, 3, 6], np.diag([1, 0.6, 0.8, 1.1, 1.2])  # 类别3

    # 生成三类数据
    data1 = np.random.multivariate_normal(mean1, cov1, 1000)
    data2 = np.random.multivariate_normal(mean2, cov2, 1000)
    data3 = np.random.multivariate_normal(mean3, cov3, 1000)

    # 合并数据和标签
    X = np.vstack((data1, data2, data3))  # 合并数据
    y = np.array([0]*1000 + [1]*1000 + [2]*1000)  # 标签：类别0、1、2

    # 计算每类的均值和总体均值
    mean_overall = np.mean(X, axis=0)
    mean_vectors = [np.mean(X[y == i], axis=0) for i in range(3)]

    # 计算类内散布矩阵 SW
    SW = np.zeros((5, 5))
    for i, mean_vec in enumerate(mean_vectors):
        class_scatter = np.cov(X[y == i].T) * (X[y == i].shape[0] - 1)
        SW += class_scatter

    # 计算类间散布矩阵 SB
    SB = np.zeros((5, 5))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i].shape[0]
        mean_diff = (mean_vec - mean_overall).reshape(5, 1)
        SB += n * (mean_diff).dot(mean_diff.T)

    # 求解广义特征值问题，获取投影方向
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))

    # 对特征值进行排序，选择前两个特征向量（降维到2维）
    sorted_indices = np.argsort(eigvals)[::-1]
    w = eigvecs[:, sorted_indices[:2]]  # 取前2个特征向量

    # 投影数据
    X_lda = X.dot(w)

    # 可视化
    plt.figure(figsize=(14, 10))

    # 原始数据分布
    plt.subplot(2, 3, 1)
    plt.title("Original High-dimensional Data Distribution")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # 类内散布矩阵可视化
    plt.subplot(2, 3, 2)
    plt.title("Class Means and Within-class Scatter")
    for i in range(3):
        sns.scatterplot(x=X[y == i][:, 0], y=X[y == i][:, 1], label=f"Class {i}")
        plt.scatter(*mean_vectors[i][:2], marker='x', s=100, c="black")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # 类间散布矩阵可视化
    plt.subplot(2, 3, 3)
    plt.title("Between-class Scatter")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")
    plt.scatter(*mean_overall[:2], marker="o", s=150, c="red", label="Overall Mean")
    for i, mean_vec in enumerate(mean_vectors):
        plt.plot([mean_overall[0], mean_vec[0]], [mean_overall[1], mean_vec[1]], 'k--')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # 投影后的数据分布
    plt.subplot(2, 3, 4)
    plt.title("Data Distribution after LDA Projection (2D)")
    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette="Set1")
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")

    # 高维数据和低维数据的散点图对比
    plt.subplot(2, 3, 5)
    plt.title("High-dim Data vs Low-dim Data (Projection Comparison)")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", alpha=0.5)
    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette="Set1", marker="X", s=70)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # LDA如何有效地将高维数据映射到2维
    plt.subplot(2, 3, 6)
    plt.title("Projection Path Analysis")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")
    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette="Set1", marker="X", s=70)
    for i in range(3):
        plt.annotate(f"Class {i}", (X_lda[y == i][:, 0].mean(), X_lda[y == i][:, 1].mean()))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    st.pyplot(plt)
