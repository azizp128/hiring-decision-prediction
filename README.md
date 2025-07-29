# 🔍 Problem Statement
The recruitment process faces significant cost and time inefficiencies, with 69% of applicants ultimately identified as low-fit, despite undergoing structured assessments and interviews. This highlights gaps in both early-stage screening and final hiring decisions. To address this, we propose predictive models that improve the quality of candidate shortlisting and hiring outcomes—ensuring that suitable candidates are prioritized from the start and retained through to successful hires.

# 🎯 Solution
We built two predictive models:
- **Pre-Interview Model (Decision Tree)**: Screens candidates early with an F1-score of 0.814.
- **Post-Interview Model (XGBoost)**: Assists final decisions with an F1-score of 0.900.

Both models were enhanced with feature engineering, SHAP explainability, and class balancing (RandomOverSampler). The tool predicts hiring outcomes based on candidate attributes, helping HR teams prioritize the right applicants.

# 📂 Dataset
- 1,500 rows × 13 columns
- Features: Demographics, Education, Experience, Scores, Distance, etc.
- Pre-Interview Target: Shortlisted (1 = Shortlisted, 0 = Not Shortlisted)
- Post-Interview Target: HiringDecision (1 = Hired, 0 = Not Hired)

# 🛠 Tools
- Languages & Libraries: Python, Pandas, Scikit-learn, XGBoost, Optuna, SHAP, Streamlit
- Deployment: Interactive web app using Streamlit
- Visualization: Matplotlib, Seaborn

# 📎 Demo
- Streamlit App: [Hiresight Hiring Decision Prediction](hiresight-hiring-decision-prediction.streamlit.app)
