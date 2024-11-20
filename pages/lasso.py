import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Streamlit 标题和描述
st.title("Lasso")
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
    # 分割数据集
    X = data[['Square Footage', 'Bedrooms', 'Floor Height', 'Distance to City']]
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建岭回归模型
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    # 创建普通线性回归模型
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    # 预测房价
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    y_pred_linear = linear_model.predict(X_test_scaled)

    # 计算误差
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)
    linear_mse = mean_squared_error(y_test, y_pred_linear)

    print(f"岭回归均方误差: {ridge_mse}")
    print(f"普通线性回归均方误差: {linear_mse}")

    # 绘图
    plt.figure(figsize=(14, 10))

    # 图1：岭回归 vs 真实价格
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred_ridge, color='blue', label='Ridge Predictions', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Ridge Regression: Predicted vs True Prices')
    plt.legend()

    # 图2：线性回归 vs 真实价格
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred_linear, color='green', label='Linear Predictions', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression: Predicted vs True Prices')
    plt.legend()

    # 图3：岭回归系数
    plt.subplot(2, 2, 3)
    coef_ridge = ridge_model.coef_
    plt.bar(X.columns, coef_ridge, color='orange')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.title('Ridge Regression Coefficients')

    # 图4：线性回归系数
    plt.subplot(2, 2, 4)
    coef_linear = linear_model.coef_
    plt.bar(X.columns, coef_linear, color='purple')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.title('Linear Regression Coefficients')

    plt.tight_layout()
    plt.show()
""")



# 训练模型
if st.button("开始训练"):
    # 设置随机种子
    np.random.seed(42)

    # 生成虚拟数据集
    n_samples = 200
    square_footage = np.random.normal(1500, 300, n_samples)  # 房屋面积
    bedrooms = np.random.randint(1, 6, n_samples)  # 卧室数量
    floor_height = np.random.randint(1, 4, n_samples)  # 楼层高度
    distance_city = np.random.normal(10, 5, n_samples)  # 距离市中心的距离

    # 模拟价格：房价主要与房屋面积和卧室数量相关
    price = (square_footage * 300 + bedrooms * 50000 
            - distance_city * 1000 + floor_height * 20000 
            + np.random.normal(0, 50000, n_samples))  # 加入噪音

    # 创建数据框
    data = pd.DataFrame({
        'Square Footage': square_footage,
        'Bedrooms': bedrooms,
        'Floor Height': floor_height,
        'Distance to City': distance_city,
        'Price': price
    })

    # 分割数据集
    X = data[['Square Footage', 'Bedrooms', 'Floor Height', 'Distance to City']]
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建岭回归模型
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    # 创建普通线性回归模型
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    # 预测房价
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    y_pred_linear = linear_model.predict(X_test_scaled)

    # 计算误差
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)
    linear_mse = mean_squared_error(y_test, y_pred_linear)

    # 输出均方误差
    st.write(f"岭回归均方误差: {ridge_mse}")
    st.write(f"普通线性回归均方误差: {linear_mse}")

    # 绘图
    plt.figure(figsize=(14, 10))

    # 图1：岭回归 vs 真实价格
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred_ridge, color='blue', label='Ridge Predictions', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Ridge Regression: Predicted vs True Prices')
    plt.legend()

    # 图2：线性回归 vs 真实价格
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred_linear, color='green', label='Linear Predictions', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression: Predicted vs True Prices')
    plt.legend()

    # 图3：岭回归系数
    plt.subplot(2, 2, 3)
    coef_ridge = ridge_model.coef_
    plt.bar(X.columns, coef_ridge, color='orange')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.title('Ridge Regression Coefficients')

    # 图4：线性回归系数
    plt.subplot(2, 2, 4)
    coef_linear = linear_model.coef_
    plt.bar(X.columns, coef_linear, color='purple')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.title('Linear Regression Coefficients')

    plt.tight_layout()
    st.pyplot(plt)  # Replace plt.show() with st.pyplot() to display the plots in Streamlit