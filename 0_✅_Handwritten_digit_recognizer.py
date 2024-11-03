import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matrix_mapper.matrix_mapper import matrix_mapper
from neural_network.neural_network import NeuralNetwork
from google.cloud import storage
import pickle
import os

# Set the path to your service account key file if running locally
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "neural-network-app-440619-e35407f6e90c.json"

# Function to run once and get neural net from pickle stored in bucket
@st.cache_data
def get_neural_network():
    # Create a storage client
    client = storage.Client()

    # Specify the bucket name
    bucket_name = 'neural-network-pre-trained'
    bucket = client.get_bucket(bucket_name)

    # Get pickled neural network
    blob = bucket.blob('nn.pickle')

    pickle_data = blob.download_as_bytes()

    # Load the pickle data
    return pickle.loads(pickle_data)

nn = get_neural_network()

st.header('Handwritten digit recognizer')
st.write('By Magnus Kvåle Helliesen')

# Accept drawing as user input 
drawing = st_canvas(
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=500,
    height=500,
    drawing_mode="freedraw",
    key="canvas"
)

# Use neural net to recognize user input
if st.button('Recognize'):
    X = np.mean(np.array(drawing.image_data)[:, :, :3], axis=2)
    x = matrix_mapper(X, 28, 28)
    
    #x[x<200] = 0
    digit = (255-x).reshape(784)

    # Normalize the input
    digit_norm = (digit-digit.mean())/digit.std()

    # Use neural net to make prediction
    prediction = nn.predict(digit_norm)

    # Find the highest probability and return as guess
    max_p = 0
    for n, p in enumerate(prediction):
        if p > max_p:
            max_p = p
            guess = n

    # Return guess with varying level of confidence
    if max_p > 0.8:
        st.header(f"I'm pretty sure it's a {guess}")
    elif max_p > 0.4:
        st.header(f"It kinda looks like a {guess}")
    elif max_p > 0.2:
        st.header(f"Could it be a {guess}?")
    else:
        st.header(f"My best guess is a {guess}?")
