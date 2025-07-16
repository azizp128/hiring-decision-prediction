import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

def show():
    st.image("hiresight.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
    st.set_page_config(page_title='HR Recruitment Post-Interview Prediction', layout='centered')

    @st.cache_resource
    def load_model():
        model_path = os.path.join(os.path.dirname(__file__), 'model_1.pkl')
        return joblib.load(model_path)

    model = load_model()

    st.title(":orange[HR Recruitment Post-Interview Prediction]")

    col1, col2 = st.columns([1, 2])

    # === Left Column: About Study ===
    with col1:
        
        st.write("""
        This project aims to support HR decision-making by predicting whether a candidate 
        is suitable for post-interview evaluation. The prediction is based on several features:
        - Age
        - Gender
        - Recruitment Strategy
        - Education level
        - Years of Experience
        - Number of Previous Companies Worked At
        - Distance from company
        - Candidate's Interview Score
        - Candidate's Skill Assessment Score
        - Candidate's Personality Fit Score

        The model is trained using a machine learning pipeline that includes encoding, 
        resampling, and hyperparameter tuning.
        """)

    # === Right Column: Input ===
    with col2:
        st.header("Candidate Profile Input")

        Age = st.number_input("Age", min_value=18, max_value=65, step=1)
        Gender = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Male" if x == 0 else "Female"
        )
        RecruitmentStrategy = st.selectbox(
            "Recruitment Strategy",
            options=[3, 2, 1],
            format_func=lambda x: {
                3: 'Walk-in or offline',
                2: 'Online platform (LinkedIn, Job Portal)',
                1: 'Headhunter/Agency'
            }[x]
        )
        education_order = ['High School', 'Diploma/Bachelor', 'Master', 'Post Graduate']
        EducationLevel = st.selectbox("Education Level", options=education_order)
        ExperienceYears = st.number_input("Years of Experience", min_value=0, max_value=15, step=1)
        PreviousCompanies = st.number_input("Number of Previous Companies Worked At", min_value=1, max_value=5, step=1)
        DistanceFromCompany = st.number_input("Distance from Company (in km)", min_value=0, max_value=55, step=1)
        InterviewScore = st.slider("Interview Score (1–100)", min_value=0, max_value=100, step=1)
        SkillScore = st.slider("Skill Assessment Score (1–100)", min_value=0, max_value=100, step=1)
        PersonalityScore = st.slider("Personality Fit Score (1–100)", min_value=0, max_value=100, step=1)

        columns = [
            'Age', 'Gender', 'RecruitmentStrategy', 'EducationLevel', 'ExperienceYears',
            'PreviousCompanies', 'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore'
        ]

        def preprocess_input(
            Age, Gender, RecruitmentStrategy, EducationLevel, ExperienceYears,
            PreviousCompanies, DistanceFromCompany, InterviewScore, SkillScore, PersonalityScore
        ):
            row = pd.DataFrame([{
                    'Age': Age,
                    'ExperienceYears': ExperienceYears,
                    'PreviousCompanies': PreviousCompanies,
                    'DistanceFromCompany': DistanceFromCompany,
                    'Gender': "Male" if Gender == 0 else "Female",
                    'RecruitmentStrategy': {
                        1: 'Headhunter/Agency',
                        2: 'Online platform (LinkedIn, Job Portal)',
                        3: 'Walk-in or offline'
                    }[RecruitmentStrategy],
                    'EducationLevel': EducationLevel,
                    'InterviewScore': InterviewScore,
                    'SkillScore': SkillScore,
                    'PersonalityScore': PersonalityScore
                }])
            return pd.DataFrame(row, columns=columns)

        def predict():
            X = preprocess_input(
                Age, Gender, RecruitmentStrategy, EducationLevel, ExperienceYears,
                PreviousCompanies, DistanceFromCompany, InterviewScore, SkillScore, PersonalityScore
            )
            with st.spinner("Evaluating candidate..."):
                prediction = model.predict(X)[0]
                decision = "Hired ✅" if prediction == 1 else "Not Hired ❌"
            st.success(f"Prediction: {decision}")

        if st.button("Predict"):
            predict()