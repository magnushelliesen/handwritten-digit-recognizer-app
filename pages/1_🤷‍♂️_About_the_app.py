import streamlit as st
from functions import get_neural_network

#Get NeuralNetwork-instance
if 'nn' in st.session_state:
    nn = st.session_state.nn
else:
    nn = get_neural_network()
    st.session_state.nn = nn

st.markdown(
    f"""
    # About the app
    This app is written by [Magnus Kvåle Helliesen](https://github.com/magnushelliesen).
    It runs on Google Cloud Run, and reads a pre-trained neural network from Google Cloud Storage.
    
    The code for the app can be found in [this GitHub-repo](https://github.com/magnushelliesen/handwritten-digit-recognizer-app),
    and the code for the neural network-package can be found in [this GitHub-repo](https://github.com/magnushelliesen/neural-network).

    ## About the neural network
    The neural network has {nn.n_hidden: ,.0f} hidden layers, with {nn.dim_hidden: ,.0f} nodes each.
    The neural network has been trained showing it {nn.training: ,.0f} random digits from the
    [MNIST dataset](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/).
    """
    )