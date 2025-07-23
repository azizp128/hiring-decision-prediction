import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

def show_post_pred():
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
        st.header("📘 About the Study")
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
        st.header("👤 Candidate Profile Input")

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

    # === Bulk Upload ===
    st.header("📂 Bulk Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload Candidate Data (.csv)", type=["csv"], key="post_file_upload")

    if uploaded_file is not None:
        try:
            df_bulk = pd.read_csv(uploaded_file)

            st.subheader("Preview of Uploaded Data")
            st.dataframe(df_bulk.head())

            # --- Bulk Preprocessing ---
            def bulk_preprocess(df):
                df_processed = df.copy()

                # Required columns
                required_columns = [
                    'Age', 'Gender', 'RecruitmentStrategy', 'EducationLevel',
                    'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany',
                    'InterviewScore', 'SkillScore', 'PersonalityScore'
                ]
                
                # Check for missing columns
                missing_cols = set(required_columns) - set(df_processed.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns in uploaded CSV: {missing_cols}")

                # Keep only required columns
                df_processed = df_processed[required_columns]

                # Normalize Gender
                df_processed['Gender'] = df_processed['Gender'].apply(
                    lambda x: 'Male' if str(x).strip().lower() == 'male' else 'Female'
                )

                # Normalize Recruitment Strategy
                recruitment_map = {
                    'Headhunter/Agency': 'Headhunter/Agency',
                    'Online platform (LinkedIn, Job Portal)': 'Online platform (LinkedIn, Job Portal)',
                    'Walk-in or offline': 'Walk-in or offline'
                }
                df_processed['RecruitmentStrategy'] = df_processed['RecruitmentStrategy'].map(recruitment_map)

                # Filter valid education levels
                valid_education_levels = ['High School', 'Diploma/Bachelor', 'Master', 'Post Graduate']
                df_processed = df_processed[df_processed['EducationLevel'].isin(valid_education_levels)]

                # Optional: drop rows with any NaN values
                df_processed = df_processed.dropna()

                return df_processed


            df_input = bulk_preprocess(df_bulk)

            # --- Prediction ---
            with st.spinner("Processing and predicting..."):
                predictions = model.predict(df_input)
                df_bulk['Prediction'] = ["Hired" if pred == 1 else "Not Hired" for pred in predictions]

                # Store results in session state
                st.session_state['post_predictions'] = df_bulk

            # --- Show and Download ---
            st.success("Bulk Post-Interview Prediction Completed ✅")
            st.subheader("📊 Prediction Results")
            st.dataframe(df_bulk)

            # Convert to CSV
            csv = df_bulk.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='bulk_post_inteview_predictions.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

    elif "post_predictions" in st.session_state:
        st.subheader("📊 Last Bulk Prediction")
        st.dataframe(st.session_state["post_predictions"])