import streamlit as st
from pages import home, hiring_model, shortlisting_model

# CSS to hide the default sidebar (replace with the actual class or ID)
st.markdown("""
<style>
.st-emotion-cache-79elbk {
    # visibility: hidden; 
    display: none;
}
# #stSitebarNav{
#     # visibility: hidden;
#     display:none;
# }
# # #stSitebarUserContent{
# #     position: top;
# }
</style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Hiresight Hiring Decision Prediction",
    layout="centered",
    initial_sidebar_state="auto"
)

# Set page configuration to remove default sidebar
def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Pre-Interview Model", "Post-Interview Model"]
    
    # Initialize session state for current page
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Check if navigation was triggered by button click
    if "navigation_target" in st.session_state:
        # Update current page from button click
        st.session_state["current_page"] = st.session_state["navigation_target"]
        # Remove the target after using it
        del st.session_state["navigation_target"]

    # Use the current page to set the selectbox index
    current_index = pages.index(st.session_state["current_page"])
    
    # Sidebar selectbox with synchronized selection
    page = st.sidebar.selectbox(
        "Go to", 
        pages, 
        index=current_index,
        key="page_selector"
    )
    
    # Update current page if selectbox changed
    if page != st.session_state["current_page"]:
        st.session_state["current_page"] = page

    # Clear results when switching pages
    if "last_page" not in st.session_state:
        st.session_state["last_page"] = None

    if st.session_state["last_page"] != st.session_state["current_page"]:
        # Clear unrelated session state
        st.session_state.pop("pre_predictions", None)
        st.session_state.pop("post_predictions", None)

    # Update the last page tracker
    st.session_state["last_page"] = st.session_state["current_page"]

    # Navigation logic using current_page
    if st.session_state["current_page"] == "Home":
        home.show()
    elif st.session_state["current_page"] == "Pre-Interview Model":
        shortlisting_model.show_pre_pred()
    elif st.session_state["current_page"] == "Post-Interview Model":
        hiring_model.show_post_pred()

if __name__ == "__main__":
    main()