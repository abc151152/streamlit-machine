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
st.title("Decision Tree")
st.markdown("""
   简单理解，决策树是一种分类和回归的机器学习算法，结构上像一棵倒挂的树，用于做出决策或预测。\n
  它通过一系列“是/否”问题把数据逐步划分成不同类别。每个分支都代表了一个决策步骤，直到最终分到一个叶节点，给出分类结果。
""")


st.header('示例代码')
st.code("""
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, 100),
        'Income': np.random.choice(['High', 'Medium', 'Low'], 100),
        'Area': np.random.choice(['Urban', 'Rural'], 100),
        'Purchased': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(data)

    # 计算熵的函数
    def entropy(y):
        values, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return -np.sum(prob * np.log2(prob))

    # 计算信息增益的函数
    def info_gain(df, feature, target):
        # 计算初始熵
        base_entropy = entropy(df[target])
        
        # 按照特征的每个值进行划分
        values = df[feature].unique()
        weighted_entropy = 0
        for value in values:
            subset = df[df[feature] == value]
            weight = len(subset) / len(df)
            weighted_entropy += weight * entropy(subset[target])
        
        # 信息增益 = 初始熵 - 加权熵
        gain = base_entropy - weighted_entropy
        return gain

    # 计算信息增益来选择最佳分割特征
    features = ['Age', 'Income', 'Area']
    target = 'Purchased'

    gains = {feature: info_gain(df, feature, target) for feature in features}
    best_feature = max(gains, key=gains.get)

    # 打印出最佳特征
    print(f"最佳分割特征是: {best_feature}")

    # 数据可视化

    # 1. 年龄和购买行为的关系
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Age', hue='Purchased', multiple='stack', palette='bright')
    plt.title('Age Distribution vs Purchase')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    # 2. 收入和购买行为的关系
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Income', hue='Purchased', palette='bright')
    plt.title('Income Level vs Purchase')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.show()

    # 3. 地区和购买行为的关系
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Area', hue='Purchased', palette='bright')
    plt.title('Area vs Purchase')
    plt.xlabel('Area')
    plt.ylabel('Count')
    plt.show()

    # 4. 选择的最佳特征的柱状图
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(gains.keys()), y=list(gains.values()), palette='bright')
    plt.title('Information Gain of Features')
    plt.xlabel('Feature')
    plt.ylabel('Information Gain')
    plt.show()
""")



# 训练模型
if st.button("开始训练"):
    # 生成虚拟数据集
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, 100),
        'Income': np.random.choice(['High', 'Medium', 'Low'], 100),
        'Area': np.random.choice(['Urban', 'Rural'], 100),
        'Purchased': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(data)

    # 计算熵的函数
    def entropy(y):
        values, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return -np.sum(prob * np.log2(prob))

    # 计算信息增益的函数
    def info_gain(df, feature, target):
        # 计算初始熵
        base_entropy = entropy(df[target])
        
        # 按照特征的每个值进行划分
        values = df[feature].unique()
        weighted_entropy = 0
        for value in values:
            subset = df[df[feature] == value]
            weight = len(subset) / len(df)
            weighted_entropy += weight * entropy(subset[target])
        
        # 信息增益 = 初始熵 - 加权熵
        gain = base_entropy - weighted_entropy
        return gain

    # 计算信息增益来选择最佳分割特征
    features = ['Age', 'Income', 'Area']
    target = 'Purchased'

    gains = {feature: info_gain(df, feature, target) for feature in features}
    best_feature = max(gains, key=gains.get)

    # 打印出最佳特征
    print(f"最佳分割特征是: {best_feature}")

    # 数据可视化
    # 1. 年龄和购买行为的关系
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Age', hue='Purchased', multiple='stack', palette='bright')
    plt.title('Age Distribution vs Purchase')
    plt.xlabel('Age')
    plt.ylabel('Count')
    st.pyplot(plt)

    # 2. 收入和购买行为的关系
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Income', hue='Purchased', palette='bright')
    plt.title('Income Level vs Purchase')
    plt.xlabel('Income')
    plt.ylabel('Count')
    st.pyplot(plt)

    # 3. 地区和购买行为的关系
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Area', hue='Purchased', palette='bright')
    plt.title('Area vs Purchase')
    plt.xlabel('Area')
    plt.ylabel('Count')
    st.pyplot(plt)

    # 4. 选择的最佳特征的柱状图
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(gains.keys()), y=list(gains.values()), palette='bright')
    plt.title('Information Gain of Features')
    plt.xlabel('Feature')
    plt.ylabel('Information Gain')
    st.pyplot(plt)