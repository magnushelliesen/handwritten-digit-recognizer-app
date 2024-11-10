import streamlit as st

st.markdown(
    """
    # About the app
    This app is written by [Magnus Kvåle Helliesen](https://github.com/magnushelliesen).
    It runs on Google Cloud Run, and reads a pre-trained neural network from Google Cloud Storage.
    
    The code for the app can be found in [this GitHub-repo](https://github.com/magnushelliesen/handwritten-digit-recognizer-app),
    and the code for the neural network-package can be found in [this GitHub-repo](https://github.com/magnushelliesen/neural-network).
    """
    )