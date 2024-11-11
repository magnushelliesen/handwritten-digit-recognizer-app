import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matrix_mapper.matrix_mapper import matrix_mapper
from neural_network.neural_network import NeuralNetwork
from google.cloud import storage
import pickle
from pathlib import Path
import os
from time import sleep
from functions import get_neural_network

st.set_page_config(
    initial_sidebar_state="collapsed"
)

#Get NeuralNetwork-instance
if 'nn' in st.session_state:
    nn = st.session_state.nn
else:
    nn = get_neural_network()
    st.session_state.nn = nn

# Show welcome message once
if not 'initialized' in st.session_state:
    welcome = st.empty()
    for i in range(0, 40, 1):
        welcome.progress(i, "Welcome to this handwritten digit-recognizer 👋",)
        sleep(0.05)
    for i in range(40, 100, 1):
        welcome.progress(i, "For best result, use the whole white square 👍")
        sleep(0.05)
    welcome.empty()
    st.session_state['initialized'] = True

st.header('Write a digit 🖋️')

# Accept drawing as user input 
drawing = st_canvas(
    stroke_width=30,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=250,
    height=250,
    drawing_mode="freedraw",
    key="canvas"
)

# Use neural net to recognize user input
if st.button("Recognize digit 👀"):
    X = np.mean(np.array(drawing.image_data)[:, :, :3], axis=2)

    # Padding the input with white in the edges

    # Pad in y direction
    height = X.shape[0]
    width = X.shape[1]

    X = np.vstack((
        np.full((int(height*0.2), width), 255),
        X,
        np.full((int(height*0.2), width), 255)
    ))
    
    # Pad in x direction
    height = X.shape[0]
    width = X.shape[1]

    X = np.hstack((
        np.full((height, int(width*0.2)), 255),
        X,
        np.full((height, int(width*0.2)), 255)
    ))

    # Resizing input to 28 x 28
    x = matrix_mapper(X, 28, 28)

    # Reshaping to a vector
    digit = (255-x).reshape(784)

    # Normalizing the input
    digit_norm = (digit-digit.mean())/digit.std()

    # Use neural net to make prediction
    try:
        prediction = nn.predict(digit_norm)
    except ValueError:
        st.error("I'm sorry, I can't make that out 😭")
        st.stop()

    # Find the highest probability and return as guess
    max_p = 0
    for n, p in enumerate(prediction):
        if p > max_p:
            max_p = p
            guess = n

    # Return guess with varying level of confidence
    if max_p > 0.8:
        st.header(f"I'm pretty sure it's a {guess} 😁")
    elif max_p > 0.4:
        st.header(f"It kinda looks like a {guess} 🙂")
    elif max_p > 0.2:
        st.header(f"Could it be a {guess}? 🤔")
    else:
        st.header(f"My best guess is a {guess} 🫣")