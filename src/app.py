import streamlit as st

st.set_page_config(initial_sidebar_state="expanded")

pg = st.navigation(
    [
        st.Page("pages/0_✨_Handwritten_digit_recognizer.py"),
        st.Page("pages/1_ℹ️_About_the_app.py"),
    ]
)
pg.run()
