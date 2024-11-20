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
st.title("逻辑回归")
st.markdown("""
   逻辑回归的核心是将线性回归的结果转化为概率。\n
  概率的范围是0到1，而线性回归的结果可能是任意实数，因此我们需要一个函数将线性回归的输出转换为0到1之间的概率值，这个函数就是sigmoid函数。
""")


st.header('示例代码')
st.code("""
     # 生成虚拟数据集
np.random.seed(42)
num_samples = 10000
X1 = np.random.normal(0, 1, num_samples)
X2 = np.random.normal(0, 1, num_samples)
# 假设两个特征之间有一定的相关性
X = np.column_stack((X1, X2))

# 定义真实的权重和偏置
true_weights = np.array([2, -3])
bias = 1.5
# 生成真实的标签 y，使用 sigmoid 函数
Z = np.dot(X, true_weights) + bias
y_true = (1 / (1 + np.exp(-Z)) > 0.5).astype(int)

# 逻辑回归模型（手动实现梯度下降）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, iterations=10000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    losses = []
    
    for i in range(iterations):
        # 计算模型预测
        Z = np.dot(X, weights) + bias
        y_pred = sigmoid(Z)
        
        # 计算损失（交叉熵损失）
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        losses.append(loss)
        
        # 计算梯度
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        
        # 更新参数
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias, losses

# 训练模型
weights, bias, losses = logistic_regression(X, y_true)

# 生成预测概率
Z_pred = np.dot(X, weights) + bias
y_pred = sigmoid(Z_pred)

# 图形1：交叉熵损失随迭代次数变化
plt.figure(figsize=(10, 6))
plt.plot(losses, color='red', linewidth=2)
plt.title('Loss Curve (Cross-Entropy Loss vs Iterations)', fontsize=16)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True)
plt.show()

# 图形2：散点图显示不同类别数据分布
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', s=50, alpha=0.7, edgecolors='k')
plt.title('Scatter Plot of Data (with True Labels)', fontsize=16)
plt.xlabel('Feature X1', fontsize=14)
plt.ylabel('Feature X2', fontsize=14)
plt.grid(True)
plt.show()

# 图形3：Sigmoid函数图形化显示概率输出
X_values = np.linspace(-6, 6, 100)
plt.figure(figsize=(10, 6))
plt.plot(X_values, sigmoid(X_values), color='blue', linewidth=2)
plt.title('Sigmoid Function', fontsize=16)
plt.xlabel('Input Value (z)', fontsize=14)
plt.ylabel('Probability (sigmoid(z))', fontsize=14)
plt.grid(True)
plt.show()

# 图形4：预测概率与实际标签对比的直方图
plt.figure(figsize=(10, 6))
sns.histplot(y_pred[y_true == 1], color='green', kde=True, label='True Positive', binwidth=0.05)
sns.histplot(y_pred[y_true == 0], color='orange', kde=True, label='True Negative', binwidth=0.05)
plt.title('Histogram of Predicted Probabilities', fontsize=16)
plt.xlabel('Predicted Probability', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
""")



# 训练模型
if st.button("开始训练"):
    # 设定随机种子
    np.random.seed(42)
    num_samples = 10000
    X1 = np.random.normal(0, 1, num_samples)
    X2 = np.random.normal(0, 1, num_samples)
    # 假设两个特征之间有一定的相关性
    X = np.column_stack((X1, X2))

    # 定义真实的权重和偏置
    true_weights = np.array([2, -3])
    bias = 1.5
    # 生成真实的标签 y，使用 sigmoid 函数
    Z = np.dot(X, true_weights) + bias
    y_true = (1 / (1 + np.exp(-Z)) > 0.5).astype(int)

    # 逻辑回归模型（手动实现梯度下降）
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def logistic_regression(X, y, learning_rate=0.01, iterations=10000):
        m, n = X.shape
        weights = np.zeros(n)
        bias = 0
        losses = []
        
        for i in range(iterations):
            # 计算模型预测
            Z = np.dot(X, weights) + bias
            y_pred = sigmoid(Z)
            
            # 计算损失（交叉熵损失）
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            losses.append(loss)
            
            # 计算梯度
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            # 更新参数
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        return weights, bias, losses

    # 训练模型
    weights, bias, losses = logistic_regression(X, y_true)

    # 生成预测概率
    Z_pred = np.dot(X, weights) + bias
    y_pred = sigmoid(Z_pred)

    # Streamlit界面
    st.title("Logistic Regression and Visualization")

    # 图形1：交叉熵损失随迭代次数变化
    st.subheader('1. Loss Curve (Cross-Entropy Loss vs Iterations)')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(losses, color='red', linewidth=2)
    ax1.set_title('Loss Curve (Cross-Entropy Loss vs Iterations)', fontsize=16)
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.grid(True)
    st.pyplot(fig1)

    # 图形2：散点图显示不同类别数据分布
    st.subheader('2. Scatter Plot of Data (with True Labels)')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', s=50, alpha=0.7, edgecolors='k')
    ax2.set_title('Scatter Plot of Data (with True Labels)', fontsize=16)
    ax2.set_xlabel('Feature X1', fontsize=14)
    ax2.set_ylabel('Feature X2', fontsize=14)
    ax2.grid(True)
    st.pyplot(fig2)

    # 图形3：Sigmoid函数图形化显示概率输出
    st.subheader('3. Sigmoid Function')
    X_values = np.linspace(-6, 6, 100)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(X_values, sigmoid(X_values), color='blue', linewidth=2)
    ax3.set_title('Sigmoid Function', fontsize=16)
    ax3.set_xlabel('Input Value (z)', fontsize=14)
    ax3.set_ylabel('Probability (sigmoid(z))', fontsize=14)
    ax3.grid(True)
    st.pyplot(fig3)

    # 图形4：预测概率与实际标签对比的直方图
    st.subheader('4. Histogram of Predicted Probabilities')
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.histplot(y_pred[y_true == 1], color='green', kde=True, label='True Positive', binwidth=0.05, ax=ax4)
    sns.histplot(y_pred[y_true == 0], color='orange', kde=True, label='True Negative', binwidth=0.05, ax=ax4)
    ax4.set_title('Histogram of Predicted Probabilities', fontsize=16)
    ax4.set_xlabel('Predicted Probability', fontsize=14)
    ax4.set_ylabel('Count', fontsize=14)
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)