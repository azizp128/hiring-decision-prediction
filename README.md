# 🔍 Problem Statement
The recruitment process faces significant cost and time inefficiencies, with 69% of applicants ultimately identified as low-fit, despite undergoing structured assessments and interviews. This highlights gaps in both early-stage screening and final hiring decisions. To address this, we propose predictive models that improve the quality of candidate shortlisting and hiring outcomes—ensuring that suitable candidates are prioritized from the start and retained through to successful hires.

# 🎯 Solution
We built two predictive models:
- **Pre-Interview Model (Decision Tree)**: Screens candidates early with an F1-score of 0.814.
- **Post-Interview Model (XGBoost)**: Assists final decisions with an F1-score of 0.900.

Both models were enhanced with feature engineering, SHAP explainability, and class balancing (RandomOverSampler). The tool predicts hiring outcomes based on candidate attributes, helping HR teams prioritize the right applicants.

# 📂 Dataset
- 1,500 rows × 11 columns
- Features: Demographics, Education, Experience, Scores, Distance, etc.
- Pre-Interview Target: ShortlistingDecision (1 = Shortlisted, 0 = Not Shortlisted)
- Post-Interview Target: HiringDecision (1 = Hired, 0 = Not Hired)

# 🛠 Tools
- Languages & Libraries: Python, Pandas, Scikit-learn, XGBoost, Optuna, SHAP, Streamlit
- Deployment: Interactive web app using Streamlit
- Visualization: Matplotlib, Seaborn

# 📎 Demo
- Streamlit App: [Hiresight Hiring Decision Prediction](https://hiresight-hiring-decision-prediction.streamlit.app/)

# 🚀 Improvement
Features used in the **Pre-Interview** model training process should ideally not be included in the feature set for the **Post-Interview** model training. Why? Because these features are only required for the initial candidate screening process (Pre-Interview). The Post-Interview model should be trained using features that are more relevant to assessing the candidate’s quality after the interview stage, such as **PersonalityScore, InterviewScore, SkillScore, CommunicationScore, ProblemSolvingScore, CulturalFitScore, LeadershipPotential,** or **TechnicalAssessmentResult**.

However, in this project, the dataset used is a dummy dataset, and we were not able to obtain additional data from stakeholders. Therefore, all features from the Pre-Interview stage were still included in the Post-Interview model training process. This was done to help the model capture patterns in the dataset more effectively, compared to training it using only features such as PersonalityScore, InterviewScore, and SkillScore.

This recommendation can serve as an improvement for similar projects in the future, ensuring that features for the Pre-Interview and Post-Interview stages are separated in accordance with best practices.
