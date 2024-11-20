import streamlit as st
import streamlit_book as stb
from pathlib import Path

# Get the current path (ensure this script runs from a file context)
current_path = Path(__file__).parent.absolute()

# Set the configuration for the Streamlit book
stb.set_book_config(
    menu_title="时间序列数据算法",  # Menu title
    menu_icon="book",  # Example of setting a valid menu icon
    options=[
        "ARIMA",  # First page option
        "SARIMA",  # Second page option
        "LSTM",
        "XGBoost"
        "LDA",
        "logist",
        "lasso",
        "adam",
        "random",
        "decision",
        "SVM",
        "Spectral Clustering"

    ],
    paths=[
        current_path / "pages/arima.py",  # Path for the ARIMA page
        current_path / "pages/sarima.py",  # Path for the SARIMA page
        current_path / "pages/LSTM.py",
        current_path / "pages/XGBoost.py",
        current_path / "pages/LDA.py",
        current_path / "pages/logist.py",
        current_path / "pages/lasso.py",
        current_path / "pages/adam.py",
        current_path / "pages/random.py",
        current_path / "pages/decision.py",
        current_path / "pages/SVM.py",
        current_path / "pages/Spectral Clustering.py",      

    ],
    icons=[
        "brightness-high-fill",  # Icon for ARIMA
         ...,
         ...,
         ...,
         ...,
         ...,
         ...,
         ...,
         ...,
         ...,
         ...,
        "bar-chart" # Icon for SARIMA
    ],
    save_answers=False,  # Set to True if you want to save user answers
)
