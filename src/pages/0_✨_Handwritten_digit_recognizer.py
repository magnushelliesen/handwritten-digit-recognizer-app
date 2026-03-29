import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type: ignore
import numpy as np
from matrix_mapper.matrix_mapper import matrix_mapper  # type: ignore
from functions import get_neural_network, center_input
import matplotlib.pyplot as plt

from numpy.typing import NDArray

# Get NeuralNetwork-instance
if "nn" in st.session_state:
    nn = st.session_state.nn
else:
    with st.spinner("Fetching neural network, hang on...", show_time=True):
        nn = get_neural_network()
    st.session_state.nn = nn

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

calculate = st.button("Recognize digit 👀")
show_details = st.toggle("Show nitty gritty details 🧮", True)

# Use neural net to recognize user input
if calculate:
    drawing_array: NDArray[np.float128]
    drawing_array = np.mean(np.array(drawing.image_data)[:, :, :3], axis=2)  # type: ignore

    drawing_array = center_input(drawing_array)

    # Padding the input with white in the edges

    # Pad in y direction
    height = drawing_array.shape[0]
    width = drawing_array.shape[1]

    drawing_array = np.vstack(
        (
            np.full((int(height * 0.25), width), 255),
            drawing_array,
            np.full((int(height * 0.15), width), 255),
        )
    )

    # Pad in x direction
    height = drawing_array.shape[0]
    width = drawing_array.shape[1]

    drawing_array = np.hstack(
        (
            np.full((height, int(width * 0.2)), 255),
            drawing_array,
            np.full((height, int(width * 0.2)), 255),
        )
    )

    # Resizing input to 28 x 28
    drawing_array_resized = matrix_mapper(drawing_array, 28, 28)

    # Reshaping to a vector
    digit = (255 - drawing_array_resized).reshape(784)

    # Normalizing the input
    normalized_digit = (digit - digit.mean()) / digit.std()

    # Use neural net to make prediction
    try:
        prediction = nn.predict(normalized_digit)
    except ValueError:
        st.error("I'm terribly sorry, I can't make that out 😭")
        st.stop()

    # Return guess with varying level of confidence
    guess = sorted(zip(prediction, range(10)), reverse=True)

    st.subheader("Best guess:")
    if guess[0][0] > 0.8:
        st.write(f"I'm _pretty_ sure thats's a {guess[0][1]} 😁")
    elif guess[0][0] > 0.4:
        st.write(f"It kinda looks like a {guess[0][1]} 🙂")
    elif guess[0][0] > 0.2:
        st.write(f"It _could_ be a {guess[0][1]} 🤔")
    else:
        st.write(f"My best guess is a {guess[0][1]} 🫣")
    if guess[1][0] > 0.2:
        st.write(f"... but it could also be a {guess[1][1]} 😵‍💫")

    if show_details:
        st.subheader("Steps in calculation:")

        # Input layer
        st.write(
            """
            The digit is first pre-preprocessed, \
            that is: cropped, centered and turned into $28 \\times 28$ pixles \
            (which is the same format the MNIST dataset operates with). \
            After pre-processing, the digit looks like this:
            """
        )
        fig, ax = plt.subplots(figsize=(4, 4), frameon=False)  # type: ignore
        plt.imshow(drawing_array_resized, cmap="plasma")  # type: ignore
        plt.xticks([])  # type: ignore
        plt.yticks([])  # type: ignore
        st.pyplot(fig)

        # Hidden layer(s)
        st.write(
            """
            Next, the digit is made into a $784$ ($=28 \\times 28$) element-long vector, \
            which is fed to the predict-method (i.e. _forward propagation_-method) of the pre-trained neural network instance. \
            This is what the activations through the hidden layers look like \
            (the vectors have been made into square matrices for visual purposes):
            """
        )

        fig, ax = plt.subplots(nrows=1, ncols=nn.n_hidden, figsize=(nn.n_hidden * 2, 2), frameon=False)  # type: ignore
        for i, activation in enumerate(nn.last_activations[:-1]):
            ax[i].imshow(activation.reshape(15, 15), cmap="plasma")
            ax[i].set_title(f"Hidden layer {i+1}:", color="#f63366")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        st.pyplot(fig)

        # Output layer
        st.write(
            "Finally, out pops the following probability distribution $P$ over digits $i$:"
        )
        fig, ax = plt.subplots(figsize=(4, 2), frameon=False)  # type: ignore
        ax.bar(height=prediction, x=[f"{i}" for i in range(10)], color="#f63366")
        ax.tick_params(axis="x", colors="#f63366")
        ax.tick_params(axis="y", colors="#f63366")
        ax.set_xlabel("$i$", color="#f63366")
        ax.set_ylabel("$P(i)$", color="#f63366")
        plt.yticks(ticks=np.linspace(0, 1, 6))  # type: ignore
        st.pyplot(fig)

        st.write("Cool, huh? 😎")
