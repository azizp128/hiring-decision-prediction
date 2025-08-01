import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap

def beautify_feature_name(feature):
    """
    Convert preprocessor feature names to readable format
    Examples:
    - 'cat__Gender_Male' -> 'Gender: Male'
    - 'cat__RecruitmentStrategy_Online platform (LinkedIn, Job Portal)' -> 'Recruitment Strategy: Online platform (LinkedIn, Job Portal)'
    - 'num__DistanceFromCompany' -> 'Distance From Company'
    """
    # Handle categorical features (format: cat__ColumnName_Value)
    if feature.startswith('cat__'):
        # Remove 'cat__' prefix
        feature_clean = feature[5:]
        
        # Split on the first underscore to separate column name and value
        if '_' in feature_clean:
            parts = feature_clean.split('_', 1)
            column_name = parts[0]
            value = parts[1]
            
            # Clean up column name (convert camelCase to spaced)
            column_name = ''.join([' ' + c.lower() if c.isupper() else c for c in column_name]).strip()
            column_name = column_name.title()
            
            # Return in format "Column Name: Value"
            return f"{column_name}: {value}"
        else:
            # Fallback if no underscore found
            feature_clean = ''.join([' ' + c.lower() if c.isupper() else c for c in feature_clean]).strip()
            return feature_clean.title()
    
    # Handle numerical features (format: num__ColumnName)
    elif feature.startswith('num__'):
        # Remove 'num__' prefix
        feature_clean = feature[5:]
        
        # Convert camelCase to spaced format
        feature_clean = ''.join([' ' + c.lower() if c.isupper() else c for c in feature_clean]).strip()
        return feature_clean.title()
    
    # Handle other prefixes or fallback
    else:
        # Remove common prefixes and clean up
        for prefix in ['remainder__', 'passthrough__', 'oe__']:
            if feature.startswith(prefix):
                feature = feature[len(prefix):]
                break
        
        # Convert camelCase/snake_case to readable format
        feature = feature.replace('_', ' ')
        feature = ''.join([' ' + c.lower() if c.isupper() else c for c in feature]).strip()
        return feature.title()

def show_pre_pred():
    st.image("hiresight.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
    st.set_page_config(page_title='HR Recruitment Pre-Interview Prediction', layout='centered')

    @st.cache_resource
    def load_model():
        model_path = os.path.join(os.path.dirname(__file__), 'shortlisting_model.pkl')
        return joblib.load(model_path)

    model = load_model()['pipeline']

    st.title(":orange[HR Recruitment Pre-Interview Prediction]")

    # === Input User ===
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
    PreviousCompanies = st.number_input("Number of Previous Companies Worked At", min_value=0, max_value=5, step=1)
    DistanceFromCompany = st.number_input("Distance from Company (in km)", min_value=0, max_value=55, step=1)

    columns = [
        'Age', 'Gender', 'RecruitmentStrategy', 'EducationLevel', 'ExperienceYears',
        'PreviousCompanies', 'DistanceFromCompany'
    ]

    def preprocess_input(
        Age, Gender, RecruitmentStrategy, EducationLevel, ExperienceYears,
        PreviousCompanies, DistanceFromCompany
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
                'EducationLevel': EducationLevel
            }])
        return pd.DataFrame(row, columns=columns)

    def predict():
        # Get user input
        X_input = preprocess_input(
            Age, Gender, RecruitmentStrategy, EducationLevel, ExperienceYears,
            PreviousCompanies, DistanceFromCompany
        )

        with st.spinner("Evaluating candidate..."):
            # Predict using the full pipeline
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]
            
            # Determine confidence level and styling
            if proba >= 0.8:
                confidence_level = "HIGH CONFIDENCE"
                confidence_color = "green"
                confidence_emoji = "🎯"
            elif proba >= 0.6:
                confidence_level = "MODERATE CONFIDENCE"
                confidence_color = "orange"
                confidence_emoji = "⚖️"
            else:
                confidence_level = "LOW CONFIDENCE"
                confidence_color = "red"
                confidence_emoji = "⚠️"

            # Create descriptive message based on prediction
            if prediction == 1:
                # Shortlisted
                decision_text = "RECOMMENDED FOR SHORTLISTING"
                decision_emoji = "✅"
                decision_color = "green"
                description = f"Candidate has a <strong>{proba * 100:.1f}%</strong> probability of being suitable for pre-interview screening"
                action_text = "Proceed to next stage"
            else:
                # Not shortlisted
                decision_text = "NOT RECOMMENDED"
                decision_emoji = "❌"
                decision_color = "red"
                description = f"Candidate has a <strong>{(1-proba) * 100:.1f}%</strong> probability of being unsuitable for pre-interview screening"
                action_text = "Consider alternative candidates"

            # Display the enhanced prediction
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            # Main decision with styling
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(128,128,128,0.1); margin: 10px 0;">
                <h2 style="color: {decision_color}; margin: 0;">{decision_emoji} {decision_text}</h2>
                <p style="font-size: 18px; margin: 10px 0;">{description}</p>
                <p style="color: {confidence_color}; font-weight: bold; margin: 5px 0;">{confidence_emoji} {confidence_level}</p>
                <p style="font-style: italic; color: gray; margin: 5px 0;">Recommendation: {action_text}</p>
            </div>
            """, unsafe_allow_html=True)

                # Add interpretation guide
            with st.expander("📖 How to Interpret This Result"):
                st.markdown("""
                **Confidence Levels:**
                - 🎯 **High Confidence (80%+)**: Strong prediction, safe to follow recommendation
                - ⚖️ **Moderate Confidence (60-79%)**: Good prediction, consider additional factors
                - ⚠️ **Low Confidence (<60%)**: Uncertain prediction, manual review recommended
                
                **Recommendation Actions:**
                - ✅ **Shortlisted**: Move to next screening stage
                - ❌ **Not Recommended**: Consider other candidates or review requirements
                """)
            
            st.markdown("---")
            # Get transformed input for SHAP (only)
            try:
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']

                X_transformed = preprocessor.transform(X_input)

                # Run SHAP
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_transformed)

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # For binary classification, usually returns [class_0_shap, class_1_shap]
                    if len(shap_values) == 2:
                        shap_values_for_class_1 = shap_values[1][0]  # Get first (and only) sample
                    else:
                        st.error(f"Expected 2 classes, got {len(shap_values)}")
                        return
                elif isinstance(shap_values, np.ndarray):
                    # Handle different array shapes
                    if len(shap_values.shape) == 3:  # Shape: (n_samples, n_features, n_classes)
                        shap_values_for_class_1 = shap_values[0, :, 1]  # First sample, all features, class 1
                    elif len(shap_values.shape) == 2:  # Shape: (n_samples, n_features) - single class
                        shap_values_for_class_1 = shap_values[0, :]  # First sample, all features
                    else:
                        st.error(f"Unexpected SHAP array shape: {shap_values.shape}")
                        return
                else:
                    st.error(f"Unexpected SHAP output type: {type(shap_values)}")
                    return

                # Get correct feature names
                feature_names = preprocessor.get_feature_names_out()

                # Safely create SHAP DataFrame
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_values_for_class_1
                })

                # Display top features
                st.markdown("### 🔍 Feature Contributions")

                # Get the actual input values for filtering
                input_values = {
                    'Gender': "Male" if Gender == 0 else "Female",
                    'RecruitmentStrategy': {
                        1: 'Headhunter/Agency',
                        2: 'Online platform (LinkedIn, Job Portal)',
                        3: 'Walk-in or offline'
                    }[RecruitmentStrategy],
                    'EducationLevel': EducationLevel
                }
                
                # Filter SHAP values to show only relevant features
                filtered_shap_data = []

                for _, row in shap_df.iterrows():
                    feature_name = row['feature']
                    shap_val = row['shap_value']
                    
                    # For categorical features, only show if it's the active category
                    if feature_name.startswith('cat__'):
                        # Extract the category value from the feature name
                        feature_parts = feature_name[5:].split('_', 1)
                        if len(feature_parts) == 2:
                            column_name, category_value = feature_parts
                            
                            # Check if this is the active category
                            if column_name in ['Gender', 'RecruitmentStrategy', 'EducationLevel']:
                                if input_values.get(column_name) == category_value:
                                    filtered_shap_data.append({
                                        'feature': feature_name,
                                        'shap_value': shap_val,
                                        'display_name': beautify_feature_name(feature_name),
                                        'abs_shap': abs(shap_val)
                                    })
                            else:
                                # For other categorical features, include all
                                filtered_shap_data.append({
                                    'feature': feature_name,
                                    'shap_value': shap_val,
                                    'display_name': beautify_feature_name(feature_name),
                                    'abs_shap': abs(shap_val)
                                })
                    else:
                        # For numerical features, always include
                        filtered_shap_data.append({
                            'feature': feature_name,
                            'shap_value': shap_val,
                            'display_name': beautify_feature_name(feature_name),
                            'abs_shap': abs(shap_val)
                        })

                # Sort by absolute impact
                filtered_shap_df = pd.DataFrame(filtered_shap_data)
                filtered_shap_df = filtered_shap_df.sort_values('abs_shap', ascending=False)

                # Show all features
                top_features = filtered_shap_df

                for _, row in top_features.iterrows():
                    feature_name = row['display_name']
                    shap_val = row['shap_value']
                    
                    # Determine contribution type
                    if shap_val > 0:
                        contribution_type = "Increases"
                        emoji = "📈"
                    else:
                        contribution_type = "Decreases"
                        emoji = "📉"
                    
                    st.markdown(f"{emoji} **{feature_name}**: {contribution_type} likelihood by `{shap_val:.3f}`")

                st.markdown("---")
                #  Show a summary of the candidate's profile
                st.markdown("### 👤 Candidate Profile Summary")
                st.markdown(f"- **Age**: {Age} years old")
                st.markdown(f"- **Gender**: {input_values['Gender']}")
                st.markdown(f"- **Recruitment Strategy**: {input_values['RecruitmentStrategy']}")
                st.markdown(f"- **Education Level**: {input_values['EducationLevel']}")
                st.markdown(f"- **Experience**: {ExperienceYears} years")
                st.markdown(f"- **Previous Companies**: {PreviousCompanies} companies")
                st.markdown(f"- **Distance**: {DistanceFromCompany} km")
                
            except KeyError:
                st.error("❌ Could not find 'encoder' or 'classifier' in pipeline steps.")
            except Exception as e:
                st.error(f"❌ SHAP explanation failed: {e}")

    if st.button("Predict"):
        predict()
    
    # === Bulk Upload ===
    st.header("📂 Bulk Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload Candidate Data (.csv)", type=["csv"], key="pre_file_upload")

    # Sample data for preview
    st.subheader("📄 Sample Data Format")
    df_sample_pre = pd.read_csv(os.path.join(os.path.dirname(__file__), '../sample-data/sample_bulk_pre_interview.csv'))
    st.dataframe(df_sample_pre.head(5))  # Show only first 5 rows for preview

    # Download button for sample data
    csv = df_sample_pre.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample Data",
        data=csv,
        file_name='sample_bulk_pre_interview.csv',
        mime='text/csv',
    )

    if uploaded_file is not None:
        try:
            df_bulk = pd.read_csv(uploaded_file)

            st.subheader("Preview of Uploaded Data")
            st.dataframe(df_bulk.head())

            # --- Bulk Preprocessing ---
            def bulk_preprocess(df):
                df_processed = df.copy()

                # Ensure columns are correct
                required_columns = [
                    'Age', 'Gender', 'RecruitmentStrategy', 'EducationLevel',
                    'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany'
                ]

                unexpected_columns = set(df_processed.columns) - set(required_columns)
                if unexpected_columns:
                    raise ValueError(f"Unexpected columns found in CSV: {unexpected_columns}")

                df_processed = df_processed[required_columns]

                # Normalize Gender
                df_processed['Gender'] = df_processed['Gender'].apply(
                    lambda x: 'Male' if str(x).strip().lower() == 'male' else 'Female'
                )

                # Normalize RecruitmentStrategy
                recruitment_map = {
                    'Headhunter/Agency': 'Headhunter/Agency',
                    'Online platform (LinkedIn, Job Portal)': 'Online platform (LinkedIn, Job Portal)',
                    'Walk-in or offline': 'Walk-in or offline'
                }
                df_processed['RecruitmentStrategy'] = df_processed['RecruitmentStrategy'].map(recruitment_map)

                # Education Level (as is)
                valid_education_levels = ['High School', 'Diploma/Bachelor', 'Master', 'Post Graduate']
                df_processed = df_processed[df_processed['EducationLevel'].isin(valid_education_levels)]

                # Final column order
                df_processed = df_processed[[
                    'Age', 'Gender', 'RecruitmentStrategy', 'EducationLevel',
                    'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany'
                ]]

                return df_processed

            df_input = bulk_preprocess(df_bulk)

            # --- Prediction ---
            with st.spinner("Processing and predicting..."):
                predictions = model.predict(df_input)
                df_bulk['Prediction'] = ["Shortlisted" if pred == 1 else "Not Shortlisted" for pred in predictions]

                # Store results in session state
                st.session_state['pre_predictions'] = df_bulk

            # --- Show and Download ---
            st.success("Bulk Pre-Interview Prediction Completed ✅")
            st.subheader("📊 Prediction Results")
            st.dataframe(df_bulk)

            # Convert to CSV
            csv = df_bulk.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='bulk_pre_interview_predictions.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    elif "pre_predictions" in st.session_state:
        st.subheader("📊 Last Bulk Prediction")
        st.dataframe(st.session_state["pre_predictions"])