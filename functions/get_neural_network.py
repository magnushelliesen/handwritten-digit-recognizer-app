import streamlit as st
from google.cloud import storage
import pickle
from pathlib import Path
import os

# Function to run once and get neural net from pickle stored in bucket
@st.cache_data
def get_neural_network():
    # Set the path to your service account key file if running locally
    if Path("neural-network-app-440619-e35407f6e90c.json").exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "neural-network-app-440619-e35407f6e90c.json"

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