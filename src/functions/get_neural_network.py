import streamlit as st
from google.cloud import storage  # pyright: ignore
import pickle
from pathlib import Path
import os


from typing import Any


# Function to run once and get neural net from pickle stored in bucket
@st.cache_data(show_spinner=False)  # type: ignore
def get_neural_network() -> Any:
    # Set the path to your service account key file if running locally
    if Path("neural-network-app-440619-e35407f6e90c.json").exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            "neural-network-app-440619-e35407f6e90c.json"
        )

    # Create a storage client
    client = storage.Client()

    # Specify the bucket name
    bucket_name = "neural-network-pre-trained"
    bucket = client.get_bucket(bucket_name)  # pyright: ignore

    # Get pickled neural network
    blob = bucket.blob("nn.pickle")  # pyright: ignore

    pickle_data = blob.download_as_bytes()  # pyright: ignore

    # Load the pickle data
    return pickle.loads(pickle_data)
