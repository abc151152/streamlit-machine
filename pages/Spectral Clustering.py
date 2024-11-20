import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.cluster import KMeans

# Streamlit 标题和描述
st.title("谱聚类")
st.markdown("""
    1.相似性矩阵：设想你有一堆数据点（例如一群同学），我们首先需要知道哪些同学关系更近，比如谁和谁更像好朋友。\n
    所以，我们给每对同学打一个分数，表示他们有多亲近，这就形成了一个“相似性矩阵”。分数越高，表示两个人关系越密切，越可能属于同一个小组。\n
    2.  构建图：接下来，我们可以想象每个同学是一个点，如果两个同学的相似度很高（关系很密切），我们就给他们之间画一条边，形成一个图。点就是同学，边表示他们的关系。通过这个图，我们可以看到有些同学联系很紧密，而有些则很疏远。\n
    3.  切割图：谱聚类的关键步骤是如何把图切开，分成几个部分，使得每个部分里的同学关系更密切，而不同部分的同学之间关系较疏远。这时候，我们利用数学中的“谱分解”技术，把这个图分割开。换句话说，我们利用图的结构信息，找到一个最好的分割方式。\n
    4. 聚类结果：最后，根据切割的结果，我们可以把同学们分成几个小组。每个小组里的同学关系密切，他们是“朋友”，而不同小组之间的联系较少。
""")


st.header('示例代码')
st.code("""
        np.random.seed(0)
        n_samples = 500
        data1 = np.random.randn(n_samples // 2, 2) + [2, 2]
        data2 = np.random.randn(n_samples // 2, 2) + [-2, -2]
        X = np.vstack([data1, data2])

        # 高斯核相似性矩阵
        def gaussian_similarity(X, sigma=1.0):
            pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
            return np.exp(-pairwise_dists / (2 * sigma ** 2))

        # 计算相似性矩阵
        S = gaussian_similarity(X, sigma=1.5)

        # 度矩阵 D
        D = np.diag(np.sum(S, axis=1))

        # 拉普拉斯矩阵 L
        L = D - S

        # 标准化拉普拉斯矩阵
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L_sym = np.eye(len(S)) - D_inv_sqrt @ S @ D_inv_sqrt

        # 特征值分解
        eigvals, eigvecs = eigh(L_sym)

        # 取前两个最小特征值对应的特征向量
        k = 2
        U = eigvecs[:, :k]

        # 对 U 的每一行进行归一化
        U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

        # 在新的特征空间上用 KMeans 聚类
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(U_normalized)

        # 绘制图形
        plt.figure(figsize=(16, 12))

        # 原始数据集
        plt.subplot(2, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c='blue', s=50, cmap='viridis')
        plt.title("Original Data", fontsize=15)

        # 相似性矩阵 S 的热力图
        plt.subplot(2, 2, 2)
        plt.imshow(S, cmap='hot', interpolation='nearest')
        plt.title("Similarity Matrix (S)", fontsize=15)

        # 拉普拉斯矩阵的热力图
        plt.subplot(2, 2, 3)
        plt.imshow(L_sym, cmap='coolwarm', interpolation='nearest')
        plt.title("Normalized Laplacian (L_sym)", fontsize=15)

        # 聚类结果
        plt.subplot(2, 2, 4)
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
        plt.title("Spectral Clustering Result", fontsize=15)

        plt.tight_layout()
        plt.show()
""")



# 训练模型
if st.button("开始训练"):
    np.random.seed(0)
    n_samples = 500
    data1 = np.random.randn(n_samples // 2, 2) + [2, 2]
    data2 = np.random.randn(n_samples // 2, 2) + [-2, -2]
    X = np.vstack([data1, data2])

    # 高斯核相似性矩阵
    def gaussian_similarity(X, sigma=1.0):
        pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
        return np.exp(-pairwise_dists / (2 * sigma ** 2))

    # 计算相似性矩阵
    S = gaussian_similarity(X, sigma=1.5)

    # 度矩阵 D
    D = np.diag(np.sum(S, axis=1))

    # 拉普拉斯矩阵 L
    L = D - S

    # 标准化拉普拉斯矩阵
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_sym = np.eye(len(S)) - D_inv_sqrt @ S @ D_inv_sqrt

    # 特征值分解
    eigvals, eigvecs = eigh(L_sym)

    # 取前两个最小特征值对应的特征向量
    k = 2
    U = eigvecs[:, :k]

    # 对 U 的每一行进行归一化
    U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

    # 在新的特征空间上用 KMeans 聚类
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(U_normalized)

    # 绘制图形
    plt.figure(figsize=(16, 12))

    # 原始数据集
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', s=50, cmap='viridis')
    plt.title("Original Data", fontsize=15)

    # 相似性矩阵 S 的热力图
    plt.subplot(2, 2, 2)
    plt.imshow(S, cmap='hot', interpolation='nearest')
    plt.title("Similarity Matrix (S)", fontsize=15)

    # 拉普拉斯矩阵的热力图
    plt.subplot(2, 2, 3)
    plt.imshow(L_sym, cmap='coolwarm', interpolation='nearest')
    plt.title("Normalized Laplacian (L_sym)", fontsize=15)

    # 聚类结果
    plt.subplot(2, 2, 4)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.title("Spectral Clustering Result", fontsize=15)

    plt.tight_layout()
    st.pyplot(plt)