import streamlit as st
import streamlit_book as stb
from pathlib import Path

# Get the current path (ensure this script runs from a file context)
current_path = Path(__file__).parent.absolute()

# Set the configuration for the Streamlit book
stb.set_book_config(
    menu_title="Machine Learning",  # Menu title
    menu_icon="book",  # Example of setting a valid menu icon
    options=[
        "ARIMA",  # First page option
        "SARIMA",  # Second page option
        "LSTM",
        "XGBoost",
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
        "graph-up",              # Icon for SARIMA
        "cpu",                   # Icon for LSTM
        "bar-chart",             # Icon for XGBoost
        "check-circle",          # Icon for LDA
        "exclamation-circle",    # Icon for logist
        "cursor",                # Icon for lasso
        "fingerprint",           # Icon for adam
        "shuffle",               # Icon for random
        "tree",                  # Icon for decision
        "gear",                  # Icon for SVM
        "cluster",               # Icon for Spectral Clustering
    ],
    save_answers=False,  # Set to True if you want to save user answers
)

# If you need to get query parameters, use this:
#query_params = st.experimental_get_query_params()
#st.write(query_params)  # Display the query parameters, if needed
query_params=st.experimental_get_query_params()
st.write(query_params) 

