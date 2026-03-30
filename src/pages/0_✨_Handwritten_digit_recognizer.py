import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type: ignore

from backend.backend import return_neural_network
from backend._0_backend import (
    recognize_on_click,
    print_guess,
    print_steps,
    reset_on_click,
)

nn = return_neural_network()

st.session_state.setdefault("is_guessing", False)


if not st.session_state.is_guessing:
    st.subheader("Write a digit (1, 2, ...) 🖋️")

    # Accept drawing as user input
    drawing = st_canvas(
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=250,
        height=250,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button(
        "Recognize digit", icon="👀", on_click=recognize_on_click, args=[drawing, nn]
    ):
        pass
else:
    st.subheader("Best guess 💡")
    print_guess()

    if st.button("Reset", icon="↩️", on_click=reset_on_click):
        pass

    if st.button("Show nitty gritty details", icon="🧮"):
        st.subheader("Steps in calculation 🚀")
        print_steps(nn)
