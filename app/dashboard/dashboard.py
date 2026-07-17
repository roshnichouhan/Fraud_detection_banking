import os
import pandas as pd
import streamlit as st
import plotly.express as px


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR,"../..")
)


DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "engineered_data.csv"
)


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)



def show_dashboard():


    st.title(
        "📊 Banking Fraud Dashboard"
    )


    df = load_data()


    filtered_df = df.copy()



    # ALL YOUR FILTERS


    # ALL YOUR CHARTS


    # ALL YOUR DOWNLOAD BUTTONS


    # ALL YOUR INSIGHTS