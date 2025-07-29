import streamlit as st

def show():
    st.image("hiresight.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)

    st.title("Profile")
    st.text("A data-driven HR consulting firm that leverages machine learning to optimize recruitment. By utilizing historical candidate data, We support more accurate, objective, and efficient hiring decisions.")
    
    st.title("Project Overview")
    st.markdown(
    """
    This project aims to enhance HR decision-making by predicting candidate suitability for shortlisting and final hiring using machine learning. Leveraging a simulated recruitment dataset of 1,500 candidates, the model analyzes features such as education, experience, test scores, and recruitment strategies.

    Two predictive models were developed:
    - **Pre-Interview Model (Decision Tree)**: Screens candidates early with an F1-score of 0.814.
    - **Post-Interview Model (XGBoost)**: Assists final decisions with an F1-score of 0.900.

    Key engineered features and SHAP-based explainability improve transparency and fairness. The solution reduces manual screening time, minimizes cost per hire, and increases hiring quality—streamlined through a deployed Streamlit app for real-time use by recruiters.
    """)