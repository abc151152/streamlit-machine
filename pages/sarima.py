import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Streamlit 标题和描述
st.title("SARIMA")
st.markdown("""
    SARIMA（季节性自回归积分滑动平均模型）适用于具有季节性和趋势的时间序列建模。它在ARIMA的基础上增加了季节性参数，能够有效处理季节性变化。
""")

# 示例代码部分
st.header('示例代码')
st.code("""
def SARIMA_test(series):
    try:
        # 设置SARIMA模型参数
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        result = model.fit()
        return result.summary()  # 返回模型的摘要信息
    except Exception as e:
        return f"SARIMA模型测试失败: {e}"
np.random.seed(42)
    n = 200
    phi_1, theta_1 = 0.8, 0.8
    epsilon = np.random.normal(0, 1, n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = phi_1 * X[t-1] + theta_1 * epsilon[t-1] + epsilon[t]

    start_date = '2002-02-21'
    dates = pd.date_range(start=start_date, periods=n)
    data = pd.DataFrame({'value': X}, index=dates)
    data.fillna(method="ffill", inplace=True)
    data = data.resample('D').interpolate('linear')

    # 显示数据的前几行
    st.write(data.head())

    # 白噪声检验
    lb = acorr_ljungbox(data['value'], lags=[1], return_df=True)
    p_value = lb['lb_pvalue'].values[0]
    st.write(f"白噪声检验 p值: {p_value}")

    # SARIMA检验
    st.subheader("SARIMA模型测试结果")
    st.text(SARIMA_test(data['value']))

    # ACF和PACF图
    st.subheader("ACF和PACF")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data['value'], lags=20, ax=ax1)
    plot_pacf(data['value'], lags=20, ax=ax2)
    st.pyplot(fig)

    # SARIMA模型拟合
    aic_values = []
    for p in range(0, 3):
        for q in range(0, 3):
            for P in range(0, 2):
                for Q in range(0, 2):
                    try:
                        model = SARIMAX(data['value'], order=(p, 1, q), seasonal_order=(P, 1, Q, 12))
                        result = model.fit()
                        aic_values.append([p, q, P, Q, result.aic])
                    except Exception:
                        continue

    # 找到最优的p、q、P和Q
    best_p, best_q, best_P, best_Q = min(aic_values, key=lambda x: x[4])[:4]
    st.write(f"最优SARIMA模型：AR({best_p}) MA({best_q}) 季节性AR({best_P}) 季节性MA({best_Q})")

    # 拟合最优SARIMA模型并进行预测
    best_model = SARIMAX(data['value'], order=(best_p, 1, best_q), seasonal_order=(best_P, 1, best_Q, 12))
    best_model_fit = best_model.fit()

    # 绘制拟合结果
    data['SARIMA'] = best_model_fit.predict(start=data.index[0], end=data.index[-1], typ='levels')
    st.subheader("SARIMA预测结果")
    fig, ax = plt.subplots(figsize=(12, 6))
    data['value'].plot(ax=ax, label='Original', color='blue')
    data['SARIMA'].plot(ax=ax, label='Predicted', color='red')
    ax.set_title(f'SARIMA Model ({best_p}, {best_q}, {best_P}, {best_Q}) 季节性周期=12')
    plt.legend()
    st.pyplot()
""")

# SARIMA测试函数
def SARIMA_test(series):
    try:
        # 设置SARIMA模型参数
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        result = model.fit()
        return result.summary()  # 返回模型的摘要信息
    except Exception as e:
        return f"SARIMA模型测试失败: {e}"

# 按钮触发代码
if st.button("运行本例"):
    # 生成模拟数据
    np.random.seed(42)
    n = 200
    phi_1, theta_1 = 0.8, 0.8
    epsilon = np.random.normal(0, 1, n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = phi_1 * X[t-1] + theta_1 * epsilon[t-1] + epsilon[t]

    start_date = '2002-02-21'
    dates = pd.date_range(start=start_date, periods=n)
    data = pd.DataFrame({'value': X}, index=dates)
    data.fillna(method="ffill", inplace=True)
    data = data.resample('D').interpolate('linear')

    # 显示数据的前几行
    st.write(data.head())

    # 白噪声检验
    lb = acorr_ljungbox(data['value'], lags=[1], return_df=True)
    p_value = lb['lb_pvalue'].values[0]
    st.write(f"白噪声检验 p值: {p_value}")

    # SARIMA检验
    st.subheader("SARIMA模型测试结果")
    st.text(SARIMA_test(data['value']))

    # ACF和PACF图
    st.subheader("ACF和PACF")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data['value'], lags=20, ax=ax1)
    plot_pacf(data['value'], lags=20, ax=ax2)
    st.pyplot(fig)

    # SARIMA模型拟合
    aic_values = []
    for p in range(0, 3):
        for q in range(0, 3):
            for P in range(0, 2):
                for Q in range(0, 2):
                    try:
                        model = SARIMAX(data['value'], order=(p, 1, q), seasonal_order=(P, 1, Q, 12))
                        result = model.fit()
                        aic_values.append([p, q, P, Q, result.aic])
                    except Exception:
                        continue

    # 找到最优的p、q、P和Q
    best_p, best_q, best_P, best_Q = min(aic_values, key=lambda x: x[4])[:4]
    st.write(f"最优SARIMA模型：AR({best_p}) MA({best_q}) 季节性AR({best_P}) 季节性MA({best_Q})")

    # 拟合最优SARIMA模型并进行预测
    best_model = SARIMAX(data['value'], order=(best_p, 1, best_q), seasonal_order=(best_P, 1, best_Q, 12))
    best_model_fit = best_model.fit()

    # 绘制拟合结果
    data['SARIMA'] = best_model_fit.predict(start=data.index[0], end=data.index[-1], typ='levels')
    st.subheader("SARIMA预测结果")
    fig, ax = plt.subplots(figsize=(12, 6))
    data['value'].plot(ax=ax, label='Original', color='blue')
    data['SARIMA'].plot(ax=ax, label='Predicted', color='red')
    ax.set_title(f'SARIMA Model ({best_p}, {best_q}, {best_P}, {best_Q}) 季节性周期=12')
    plt.legend()
    st.pyplot()
