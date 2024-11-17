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
from functions import get_neural_network, center_input
import matplotlib.pyplot as plt

st.set_page_config(
    initial_sidebar_state="collapsed"
)

#Get NeuralNetwork-instance
if 'nn' in st.session_state:
    nn = st.session_state.nn
else:
    nn = get_neural_network()
    st.session_state.nn = nn

st.header('Write a digit 🖋️')

# Accept drawing as user input 
drawing = st_canvas(
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=250,
    height=250,
    drawing_mode="freedraw",
    key="canvas"
)

calculate = st.button("Recognize digit 👀")
show_details = st.toggle("Show nitty gritty details 🧮", True)

# Use neural net to recognize user input
if calculate:
    X = np.mean(np.array(drawing.image_data)[:, :, :3], axis=2)

    X = center_input(X)

    # Padding the input with white in the edges

    # Pad in y direction
    height = X.shape[0]
    width = X.shape[1]

    X = np.vstack((
        np.full((int(height*0.25), width), 255),
        X,
        np.full((int(height*0.15), width), 255)
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
        st.error("I'm terribly sorry, I can't make that out 😭")
        st.stop()

    # Return guess with varying level of confidence
    guess = sorted(zip(prediction, range(10)), reverse=True)

    st.header("Best guess:")
    if guess[0][0] > 0.8:
        st.write(f"I'm pretty sure it's a {guess[0][1]} 😁")
    elif guess[0][0] > 0.4:
        st.write(f"It kinda looks like a {guess[0][1]} 🙂")
    elif guess[0][0] > 0.2:
        st.write(f"It could be a {guess[0][1]} 🤔")
    else:
        st.write(f"My best guess is a {guess[0][1]} 🫣")
    if guess[1][0] > 0.2:
        st.write(f"... but it could also be a {guess[1][1]} 😵‍💫")

    if show_details:
        st.header("Steps in calculation:")

        st.write('The digit is first pre-preprocessed, \
                 that is: cropped, centered and turned into $28 \\times 28$ pixles \
                 (which is the same format the MNIST dataset operates with). \
                 After pre-processing, the digit looks like this:')
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.imshow(x, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax.axis('off')
        st.pyplot(fig)

        st.write('Next, the digit is made into a $28 \\times 28 = 784$ element long vector.\
                 Finally, this is fed to the forward propagation-method of the pre-trained neural network instance, \
                 and out pops the following probability distribution $P$ over digits $i$:')
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(height=prediction, x=[f'{i}' for i in range(10)], color='#f63366')
        plt.xlabel('$i$')
        plt.ylabel('$P(i)$')
        plt.yticks(ticks=np.linspace(0, 1, 6))
        st.pyplot(fig)