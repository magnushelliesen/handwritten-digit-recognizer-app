import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/0_🤖_Handwritten_digit_recognizer.py"),
        st.Page("pages/1_🤷‍♂️_About_the_app.py")
    ]
)
pg.run()
