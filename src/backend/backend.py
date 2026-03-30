import streamlit as st
from functions import get_neural_network

from neural_network import NeuralNetwork


def return_neural_network() -> NeuralNetwork:
    # Get NeuralNetwork-instance
    if "nn" in st.session_state:
        nn = st.session_state.nn
    else:
        with st.spinner(
            ":rainbow[Fetching neural network, hang on...]", show_time=True
        ):
            nn = get_neural_network()
        st.session_state.nn = nn

    return nn
