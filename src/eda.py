# src/eda.py

import streamlit as st
from ydata_profiling import ProfileReport
import pandas as pd
import tempfile
import os

def run_eda(df: pd.DataFrame):
    """Generate and render the EDA report inline in Streamlit without streamlit-pandas-profiling."""
    st.subheader("Exploratory Data Analysis (EDA)")
    
    with st.spinner("Generating EDA report..."):
        profile = ProfileReport(df, title="EDA Report", explorative=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            profile.to_file(tmp_file.name)
            tmp_path = tmp_file.name
        
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_report = f.read()
            st.components.v1.html(html_report, height=1000, scrolling=True)
        
        os.remove(tmp_path)  # Clean up temp file
