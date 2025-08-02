import streamlit as st

def show():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8bc0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .profile-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .overview-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .model-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border-left: 5px solid #ffffff;
    }
    
    .model-card-alt {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border-left: 5px solid #ffffff;
    }
    
    .feature-list {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .metric-container {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffd700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .highlight-text {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem;
        border-radius: 5px;
        display: inline-block;
        margin: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    st.image("hiresight.png", use_container_width=True)
    st.markdown("---")
    
    # Profile Section
    st.markdown("""
    <div class="profile-container">
        <h2>🏢 About Us</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            A data-driven HR consulting firm that leverages machine learning to optimize recruitment. By utilizing historical candidate data, We support more accurate, objective, and efficient hiring decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview Section
    st.markdown("""
    <div class="overview-container">
        <h2>📊 Project Overview</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            This innovative project transforms HR decision-making by predicting candidate suitability 
            for both shortlisting and final hiring stages using state-of-the-art machine learning techniques. 
            Our comprehensive analysis of a simulated recruitment dataset containing <strong>1,500 candidates</strong> 
            evaluates multiple dimensions including education, experience, assessment scores, and recruitment strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Performance Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">0.814</div>
            <div class="metric-label">Pre-Interview F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">0.900</div>
            <div class="metric-label">Post-Interview F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">1,500</div>
            <div class="metric-label">Candidates Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Two Model Approach
    st.markdown("## 🤖 Our Dual-Model Approach")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white; 
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37); border-left: 5px solid #ffffff;">
            <h3>🔍 Pre-Interview Screening Model</h3>
            <p><strong>Algorithm:</strong> Decision Tree</p>
            <p><strong>Purpose:</strong> Early candidate screening and shortlisting</p>
        </div>
        """, unsafe_allow_html=True)

        # Streamlit button that updates session state
        if st.button("🚀 Try Pre-Interview Model", key="pre_interview_btn", use_container_width=True):
            # Update session state to change the page
            st.session_state["navigation_target"] = "Pre-Interview Model"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white; 
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37); border-left: 5px solid #ffffff;">
            <h3>✅ Post-Interview Decision Model</h3>
            <p><strong>Algorithm:</strong> XGBoost</p>
            <p><strong>Purpose:</strong> Final candidate hiring decision support</p>
        </div>
        """, unsafe_allow_html=True)
        # Streamlit button that updates session state
        if st.button("✨ Try Post-Interview Model", key="post_interview_btn", use_container_width=True):
            # Update session state to change the page
            st.session_state["navigation_target"] = "Post-Interview Model"
            st.rerun()
    
    # Key Benefits Section
    st.markdown("## 🚀 Key Benefits & Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>⏱️ Time Efficiency</h4>
            <p>Dramatically reduces manual screening time through automated candidate evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card-alt">
            <h4>💰 Cost Reduction</h4>
            <p>Minimizes cost per hire by optimizing resource allocation and reducing bad hires</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-card">
            <h4>📈 Quality Enhancement</h4>
            <p>Increases hiring quality through data-driven insights and objective evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Features
    st.markdown("## 🔧 Technical Excellence")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        <div class="feature-list" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px;">
            <h4>🎛️ Advanced Features</h4>
            <ul>
                <li>🔀 <strong>Feature Engineering:</strong> Sophisticated data transformation</li>
                <li>🎯 <strong>SHAP Explainability:</strong> Transparent AI decisions</li>
                <li>⚖️ <strong>Bias Mitigation:</strong> Fair and ethical hiring practices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col2:
        st.markdown("""
        <div class="feature-list" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1.5rem; border-radius: 10px;">
            <h4>🚀 Deployment Ready</h4>
            <ul>
                <li>📱 <strong>Streamlit App:</strong> User-friendly interface</li>
                <li>⚡ <strong>Real-time Predictions:</strong> Instant candidate evaluation</li>
                <li>📊 <strong>Interactive Dashboards:</strong> Comprehensive analytics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;">
        <h3>Ready to Transform Your Hiring Process? 🚀</h3>
        <p style="font-size: 1.1rem;">Navigate to our prediction models using the sidebar to experience the power of AI-driven recruitment!</p>
    </div>
    """, unsafe_allow_html=True)