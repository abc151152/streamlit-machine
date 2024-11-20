import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Streamlit 标题和描述
# Streamlit 标题和描述
st.title("随机森林")
st.markdown("""
    随机森林的基础是决策树。
    决策树是一种树状结构，每个节点表示一个特征，每个叶子节点表示一个类别或一个数值。
    学习过程是递归的，根据选择的特征将数据划分成子集，直到达到停止条件。   
    随机性引入  •  随机抽样： 针对每个决策树的训练集，从原始数据集中进行随机抽样（有放回抽样），形成不同的训练子集。这使得每棵树的训练集都是略有不同的。  •  随机特征选择：在每次决策树的节点划分时，随机选择一个特征进行划分。这防止了某个特定特征对模型的过度依赖。     
    Bootstrap Aggregating (Bagging)  •  针对每个随机抽样得到的训练子集，训练一个独立的决策树。 
    •  预测时，对所有决策树的输出取平均（回归问题）或进行投票（分类问题）。     
    预测  •  对于回归问题，将所有决策树的预测结果取平均。 
    •  对于分类问题，进行投票，选择得票最多的类别作为最终预测。
""")

# 示例代码显示
st.header('示例代码')
st.code("""
    rng=np.random.RandomState(1)
    X=np.sort(200*rng.rand(600,1)-100,axis=0)
    y=np.pi*np.sin(X).ravel()+0.5*rng.rand(600)

    #创建随机森林模型
    n_trees=100
    max_depth=30
    regr_rf=RandomForestRegressor(n_estimators=n_trees,max_depth=max_depth,random_state=2)
    regr_rf.fit(X,y)

    #生成新数据进行预测
    X_tes=np.arange(-100,100,0.01)[:,np.newaxis]
    y_r=regr_rf.predict(X_tes)

    #绘制结果
    plt.figure(figsize=(10,6))
    plt.scatter(X,y,edgecolor="k",c="navy",s=20,marker="o",label="Data")
    plt.plot(X_tes, y_r,color="darkorange",label="Random Forest Prediction",linewidth=2)
    plt.xlabel("Input Feature")
    plt.ylabel("Target")
    plt.title("Random Forest Regression")
    plt.legend()
    plt.show()
""")

# 训练模型
if st.button("开始训练"):
    # 生成随机数据集
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
    y = np.pi * np.sin(X).ravel() + 0.5 * rng.rand(600)

    # 创建随机森林模型
    n_trees = 100
    max_depth = 30
    regr_rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=2)
    regr_rf.fit(X, y)

    # 生成新数据进行预测
    X_tes = np.arange(-100, 100, 0.01)[:, np.newaxis]
    y_r = regr_rf.predict(X_tes)  # 使用 X_tes 进行预测

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, edgecolor="k", c="navy", s=20, marker="o", label="Data")
    plt.plot(X_tes, y_r, color="darkorange", label="Random Forest Prediction", linewidth=2)
    plt.xlabel("Input Feature")
    plt.ylabel("Target")
    plt.title("Random Forest Regression")
    plt.legend()

    # 在Streamlit中显示图像
    st.pyplot(plt)

    # 清除绘图以防止影响后续图表
    plt.clf()