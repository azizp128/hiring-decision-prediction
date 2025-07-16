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

from pages import Home, Model1, Model2

# Set page configuration to remove default sidebar


def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Pre-Interview Model", "Post-Interview Model"]
    page = st.sidebar.selectbox("Go to", pages)

    if page == "Home":
        Home.show()
    elif page == "Pre-Interview Model":
        Model1.show()
    elif page == "Post-Interview Model":
        Model2.show()

if __name__ == "__main__":
    main()
















