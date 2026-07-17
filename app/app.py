import os
import sys
import streamlit as st


# ==========================================================
# Project Path Fix
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


# ==========================================================
# Import Dashboard Pages
# ==========================================================

from dashboard.home import show_home
from dashboard.dashboard import show_dashboard
from dashboard.prediction import show_prediction
from dashboard.batch_prediction import show_batch_prediction
from dashboard.model_performance import show_model_performance
from dashboard.about import show_about



# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Banking Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ==========================================================
# Custom CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

section[data-testid="stSidebar"]{
    background:#0E1117;
}

.sidebar-title{
    font-size:24px;
    font-weight:bold;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:10px;
    box-shadow:0px 2px 6px rgba(0,0,0,0.1);
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>

""", unsafe_allow_html=True)



# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.markdown(
    "<div class='sidebar-title'>🏦 Banking Fraud Detection</div>",
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔍 Single Prediction",
        "📁 Batch Prediction",
        "📈 Model Performance",
        "ℹ About"
    ]
)


st.sidebar.markdown("---")


st.sidebar.success(
    "AI Powered Fraud Detection"
)


st.sidebar.info(
"""
Model

Random Forest

Hyperparameter Tuning

GridSearchCV

Cross Validation

Feature Engineering
"""
)


st.sidebar.markdown("---")

st.sidebar.caption("Version 1.0")



# ==========================================================
# Page Routing
# ==========================================================


if page == "🏠 Home":

    show_home()


elif page == "📊 Dashboard":

    show_dashboard()


elif page == "🔍 Single Prediction":

    show_prediction()


elif page == "📁 Batch Prediction":

    show_batch_prediction()


elif page == "📈 Model Performance":

    show_model_performance()


elif page == "ℹ About":

    show_about()



# ==========================================================
# Footer
# ==========================================================

st.markdown("---")


st.markdown(
"""
<div class='footer'>

AI Banking Fraud Detection System

<br>

Built with ❤️ using

<br>

Python • Scikit-Learn • Streamlit • Plotly

</div>
""",
unsafe_allow_html=True
)