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
st.title("SVM")
st.markdown("""
  SVM 常用于：
图像分类（比如猫狗分类）
文本分类（比如垃圾邮件分类）
生物信息学（比如基因数据分类）
这样，SVM 就是一个非常实用的工具，能帮我们在各种复杂的场景下进行分类工作。

只要记住 SVM 就是在帮我们找到一条线来分开不同的东西，而且它会尽可能让这条线分得稳妥，这样就容易理解了！
""")


st.header('示例代码')
st.code("""
    # 定义线性核函数
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)


    # 构建SVM的优化目标函数（拉格朗日乘子法）
    def fit_svm(X, y, C=1.0):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = linear_kernel(X[i], X[j])

        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
        A = y.reshape(1, -1)
        b = np.zeros(1)

        # 使用cvxopt求解器求解二次优化问题
        from cvxopt import matrix, solvers
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A, (1, n_samples), 'd')
        b = matrix(b)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(solution['x'])

        # 计算权重向量w
        w = np.sum(alphas[:, None] * y[:, None] * X, axis=0)

        # 计算偏置b
        sv = (alphas > 1e-5)
        b = np.mean(y[sv] - np.dot(X[sv], w))

        return w, b, alphas


    # 训练SVM
    w, b, alphas = fit_svm(X, y)

    # 画出支持向量
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color='red', label='Positive (+1)')
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color='blue', label='Negative (-1)')
    plt.scatter(X[alphas > 1e-5][:, 0], X[alphas > 1e-5][:, 1], s=100, facecolors='none', edgecolors='yellow',
                label='Support Vectors')

    # 画出分割超平面
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    plt.fill_between(xx[0], y_min, y_max, where=Z[0] > 0, color='red', alpha=0.1)
    plt.fill_between(xx[0], y_min, y_max, where=Z[0] < 0, color='blue', alpha=0.1)
    plt.title("SVM Decision Boundary and Support Vectors")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # 画出拉格朗日乘子的分布
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(alphas)), alphas, 'ro', label='Lagrange Multipliers')
    plt.title("Distribution of Lagrange Multipliers")
    plt.xlabel('Data Index')
    plt.ylabel('Alpha Value')
    plt.legend()
    plt.show()

    # 分类结果的3D图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pos[:, 0], X_pos[:, 1], alphas[:n_points], color='red', label='Positive (+1)')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], alphas[n_points:], color='blue', label='Negative (-1)')
    ax.set_title("3D View of Data with Lagrange Multipliers")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Alpha')
    ax.legend()
    plt.show()
""")



# 训练模型
if st.button("开始训练"):
    # 设置随机种子和数据点数量
    np.random.seed(1)
    n_points = 1000

    # 生成正负两类数据
    X_pos = np.random.randn(n_points, 2) + [2, 2]  # 正类 (+1)
    X_neg = np.random.randn(n_points, 2) + [-2, -2]  # 负类 (-1)

    # 合并数据
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_points), -1 * np.ones(n_points)))

    # 画出初始数据分布
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color='red', label='Positive (+1)')
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color='blue', label='Negative (-1)')
    plt.title("Initial Data Distribution")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


    # 定义线性核函数
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)


    # 构建SVM的优化目标函数（拉格朗日乘子法）
    def fit_svm(X, y, C=1.0):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = linear_kernel(X[i], X[j])

        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
        A = y.reshape(1, -1)
        b = np.zeros(1)

        # 使用cvxopt求解器求解二次优化问题
        from cvxopt import matrix, solvers
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A, (1, n_samples), 'd')
        b = matrix(b)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(solution['x'])

        # 计算权重向量w
        w = np.sum(alphas[:, None] * y[:, None] * X, axis=0)

        # 计算偏置b
        sv = (alphas > 1e-5)
        b = np.mean(y[sv] - np.dot(X[sv], w))

        return w, b, alphas


    # 训练SVM
    w, b, alphas = fit_svm(X, y)

    # 画出支持向量
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color='red', label='Positive (+1)')
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color='blue', label='Negative (-1)')
    plt.scatter(X[alphas > 1e-5][:, 0], X[alphas > 1e-5][:, 1], s=100, facecolors='none', edgecolors='yellow',
                label='Support Vectors')

    # 画出分割超平面
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    plt.fill_between(xx[0], y_min, y_max, where=Z[0] > 0, color='red', alpha=0.1)
    plt.fill_between(xx[0], y_min, y_max, where=Z[0] < 0, color='blue', alpha=0.1)
    plt.title("SVM Decision Boundary and Support Vectors")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # 画出拉格朗日乘子的分布
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(alphas)), alphas, 'ro', label='Lagrange Multipliers')
    plt.title("Distribution of Lagrange Multipliers")
    plt.xlabel('Data Index')
    plt.ylabel('Alpha Value')
    plt.legend()
    plt.show()

    # 分类结果的3D图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pos[:, 0], X_pos[:, 1], alphas[:n_points], color='red', label='Positive (+1)')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], alphas[n_points:], color='blue', label='Negative (-1)')
    ax.set_title("3D View of Data with Lagrange Multipliers")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Alpha')
    ax.legend()
    st.pyplot(plt)