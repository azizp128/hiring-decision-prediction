import streamlit as st
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

from pages import home, hiring_model, shortlisting_model

# Set page configuration to remove default sidebar
def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Pre-Interview Model", "Post-Interview Model"]
    page = st.sidebar.selectbox("Go to", pages)

    # Clear results when switching pages
    if "last_page" not in st.session_state:
        st.session_state["last_page"] = None

    if st.session_state["last_page"] != page:
        # Clear unrelated session state
        st.session_state.pop("pre_predictions", None)
        st.session_state.pop("post_predictions", None)

    # Update the current page
    st.session_state["last_page"] = page

    if page == "Home":
        home.show()
    elif page == "Pre-Interview Model":
        shortlisting_model.show_pre_pred()
    elif page == "Post-Interview Model":
        hiring_model.show_post_pred()

if __name__ == "__main__":
    main()