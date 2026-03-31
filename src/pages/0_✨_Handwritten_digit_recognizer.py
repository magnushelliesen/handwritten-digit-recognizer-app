import streamlit as st
from streamlit_drawable_canvas import st_canvas  # pyright: ignore

from backend.backend import return_neural_network
from backend.backend_0 import (
    recognize_on_click,
    print_guess,
    print_steps,
    reset_on_click,
)

nn = return_neural_network()

st.session_state.setdefault("is_guessing", False)

with st.container(width=400):
    if not st.session_state.is_guessing:
        st.subheader("Write a digit 🖋️")
        st.write("Such as 1, 2, ... (It doesn't know letters.)")

        _, col, _ = st.columns([1, 6, 1])

        # Accept drawing as user input
        with col:
            drawing = st_canvas(
                stroke_width=30,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=300,
                height=300,
                drawing_mode="freedraw",
                key="canvas",
            )

        if st.button(
            "Recognize digit",
            icon="👀",
            on_click=recognize_on_click,
            args=[drawing, nn],
            width="stretch",
        ):
            pass
    else:
        st.subheader("Best guess 💡")
        print_guess()

        col_1, col_2 = st.columns(2)

        if col_1.button("Go back", icon="↩️", on_click=reset_on_click, width="stretch"):
            pass

        if col_2.button("Show details", icon="🧮", width="stretch"):
            st.subheader("Steps in calculation 🚀")
            print_steps(nn)
