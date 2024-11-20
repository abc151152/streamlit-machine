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
st.title("Adam")
st.markdown("""
   Adam是一种自适应学习率的优化算法，结合了动量和自适应学习率的特性。\n
   主要思想是根据参数的梯度来动态调整每个参数的学习率。\n
    核心原理包括：\n
    1.  动量（Momentum  ）：Adam算法引入了动量项，以平滑梯度更新的方向。这有助于加速收敛并减少震荡。\n
    2.  自适应学习率：Adam算法计算每个参数的自适应学习率，允许不同参数具有不同的学习速度。 \n
    3. 偏差修正（Bias Correction）：Adam算法在初期迭代中可能受到偏差的影响，因此它使用偏差修正来纠正这个问题。
""")


st.header('示例代码')
st.code("""
     # 定义神经网络模型
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict(X, weights):
        return sigmoid(np.dot(X, weights))

    # 初始化参数和超参数
    theta = np.random.rand(2)  # 参数初始化
    alpha = 0.1  # 学习率
    beta1 = 0.9  # 一阶矩衰减因子
    beta2 = 0.999  # 二阶矩衰减因子
    epsilon = 1e-8  # 用于防止分母为零

    # 初始化Adam算法所需的中间变量
    m = np.zeros(2)
    v = np.zeros(2)
    t = 0

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        for i in range(len(X)):
            t += 1
            gradient = (predict(X[i], theta) - y[i]) * X[i]
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    # 输出训练后的参数
    print("训练完成后的参数：", theta)

    # 定义损失函数
    def loss(X, y, weights):
        y_pred = predict(X, weights)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # 记录损失值
    loss_history = []
    for i in range(len(X)):
        loss_history.append(loss(X[i], y[i], theta))
""")



# 训练模型
if st.button("开始训练"):
    # 设置随机种子
    np.random.seed(42)

    # 特征数据和标签
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # 定义sigmoid激活函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 定义预测函数
    def predict(X, weights):
        return sigmoid(np.dot(X, weights))

    # 定义损失函数
    def loss(X, y, weights):
        y_pred = predict(X, weights)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # 初始化参数和超参数
    theta = np.random.rand(2)  # 参数初始化
    alpha = 0.1  # 学习率
    beta1 = 0.9  # 一阶矩衰减因子
    beta2 = 0.999  # 二阶矩衰减因子
    epsilon = 1e-8  # 用于防止分母为零

    # 初始化Adam算法所需的中间变量
    m = np.zeros(2)
    v = np.zeros(2)
    t = 0

    # 训练模型
    num_epochs = 100
    loss_history = []  # 用于记录每个epoch的损失

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(len(X)):
            t += 1
            gradient = (predict(X[i], theta) - y[i]) * X[i]
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # 计算并记录每个样本的损失
            epoch_loss += loss(X[i], y[i], theta)
        
        # 记录每个epoch的平均损失
        loss_history.append(epoch_loss / len(X))

    # 输出训练后的参数
    print("训练完成后的参数：", theta)

    # 绘制损失函数曲线
    plt.plot(range(num_epochs), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Function Value")
    plt.title("Change in Loss Function Over Time")

    # 在Streamlit中显示图像
    st.pyplot(plt)
