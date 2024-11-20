import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Streamlit 标题和描述
st.title("LSTM 时间序列预测")
st.markdown("""
    **长短期记忆网络（LSTM）** 是一种特殊的循环神经网络（RNN）结构，能够处理长序列数据，避免梯度消失和梯度爆炸问题。\n
    本示例展示如何使用 LSTM 模型对时间序列数据进行预测。
""")

# 数据生成和处理部分
st.header('示例代码')
st.code("""
    losses = []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(train_loader))
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        # 绘制损失曲线
        st.subheader("训练损失曲线")
        plt.figure(figsize=(10, 6))
        plt.plot(losses, color="red")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        st.pyplot(plt)

        # 模型评估
        model.eval()
        with torch.no_grad():
            predictions = model(X_train).cpu().numpy()

        # 可视化结果
        st.subheader("预测结果与真实值对比")
        plt.figure(figsize=(14, 8))
        plt.plot(df.index[seq_length:], y, label="True Temperature", color="blue")
        plt.plot(df.index[seq_length:], predictions.flatten(), label="Predicted Temperature", color="orange")
        plt.title("Prediction vs True Temperature")
        plt.xlabel("Date")
        plt.ylabel("Temperature")
        plt.legend()
        st.pyplot(plt)

        # 误差分布
        errors = y - predictions.flatten()
        st.subheader("预测误差分布")
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, color="purple", alpha=0.7)
        plt.title("Prediction Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        st.pyplot(plt)
""")


def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    temperature = np.sin(np.linspace(0, 2 * np.pi, len(dates))) * 10 + 20 + np.random.normal(0, 2, len(dates))
    humidity = np.cos(np.linspace(0, 2 * np.pi, len(dates))) * 20 + 60 + np.random.normal(0, 5, len(dates))
    wind_speed = np.abs(np.sin(np.linspace(0, 4 * np.pi, len(dates)))) * 5 + np.random.normal(0, 1, len(dates))
    
    df = pd.DataFrame({'Date': dates, 'Temperature': temperature, 'Humidity': humidity, 'WindSpeed': wind_speed})
    df.set_index('Date', inplace=True)
    return df

# 序列化函数
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length, 0])  # 预测温度
    return np.array(xs), np.array(ys)

# LSTM 模型定义
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# 用户配置
st.sidebar.header("模型配置")
seq_length = st.sidebar.slider("时间步长度 (seq_length)", 10, 60, 30)
hidden_size = st.sidebar.slider("隐藏层大小 (hidden_size)", 32, 128, 64)
num_layers = st.sidebar.slider("LSTM 层数 (num_layers)", 1, 3, 2)
learning_rate = st.sidebar.slider("学习率 (learning_rate)", 0.0001, 0.01, 0.001, step=0.0001)
epochs = st.sidebar.slider("训练轮次 (epochs)", 50, 300, 100)

# 数据准备
df = generate_data()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
X, y = create_sequences(data_scaled, seq_length)

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 模型初始化
model = LSTM(X_train.shape[2], hidden_size, num_layers, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
if st.button("开始训练"):
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 10 == 0:
            st.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 绘制损失曲线
    st.subheader("训练损失曲线")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color="red")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    st.pyplot(plt)

    # 模型评估
    model.eval()
    with torch.no_grad():
        predictions = model(X_train).cpu().numpy()

    # 可视化结果
    st.subheader("预测结果与真实值对比")
    plt.figure(figsize=(14, 8))
    plt.plot(df.index[seq_length:], y, label="True Temperature", color="blue")
    plt.plot(df.index[seq_length:], predictions.flatten(), label="Predicted Temperature", color="orange")
    plt.title("Prediction vs True Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    st.pyplot(plt)

    # 误差分布
    errors = y - predictions.flatten()
    st.subheader("预测误差分布")
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, color="purple", alpha=0.7)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    st.pyplot(plt)
