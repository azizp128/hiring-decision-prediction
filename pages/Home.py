import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def show():
    st.image("hiresight.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)

    st.title("Profile")
    st.text("A data-driven HR consulting firm that leverages machine learning to optimize recruitment. By utilizing historical candidate data, We support more accurate, objective, and efficient hiring decisions.")
    
    st.title("Project Overview")
    st.text("This project aims to support HR decision-making by predicting whether a candidate is suitable for pre-interview shortlisting. The prediction is based on several features:\n")