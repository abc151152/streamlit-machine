import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

# Streamlit 标题和描述
st.title("ARIMA")
st.markdown("""
    ARIMA（自回归积分滑动平均模型）适用于非平稳时间序列的建模，通过差分使数据平稳，并结合自回归和滑动平均处理数据的时序特性。
""")

# 示例代码部分
st.header('示例代码')
st.code("""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

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

# ADF检验和ACF/PACF图
def ADF_test(series):
    adf = ADF(series)
    return f"序列{'平稳' if adf[1] <= 0.05 else '不平稳'}, p值={adf[1]}"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(data['value'], lags=20, ax=ax1)
plot_pacf(data['value'], lags=20, ax=ax2)
plt.show()
""")

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

    # ADF检验
    adf = ADF(data['value'])
    st.write(f"ADF检验结果：{'平稳' if adf[1] <= 0.05 else '不平稳'}, p值={adf[1]}")

    # ACF和PACF图
    st.subheader("ACF和PACF")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data['value'], lags=20, ax=ax1)
    plot_pacf(data['value'], lags=20, ax=ax2)
    st.pyplot(fig)

    # ARIMA模型拟合
    aic_values = []
    for p in range(0, 3):
        for q in range(0, 3):
            try:
                model = sm.tsa.ARIMA(data['value'], order=(p, 0, q))
                result = model.fit()
                aic_values.append([p, q, result.aic])
            except Exception:
                continue

    # 找到最优的p和q
    best_p, best_q = min(aic_values, key=lambda x: x[2])[:2]
    st.write(f"最优ARIMA模型：AR({best_p}) MA({best_q})")

    # 拟合最优ARIMA模型并进行预测
    best_model = sm.tsa.ARIMA(data['value'], order=(best_p, 0, best_q))
    best_model_fit = best_model.fit()

    # 绘制拟合结果
    data['ARIMA'] = best_model_fit.predict(start=data.index[0], end=data.index[-1], typ='levels')
    st.subheader("ARIMA预测结果")
    fig, ax = plt.subplots(figsize=(12, 6))
    data['value'].plot(ax=ax, label='Original', color='blue')
    data['ARIMA'].plot(ax=ax, label='Predicted', color='red')
    ax.set_title(f'ARIMA Model ({best_p}, {best_q})')
    plt.legend()
    st.pyplot()
