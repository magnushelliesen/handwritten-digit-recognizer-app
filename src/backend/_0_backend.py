import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type: ignore
import numpy as np
from matrix_mapper.matrix_mapper import matrix_mapper  # type: ignore
from functions import center_input
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from streamlit_drawable_canvas import CanvasResult  # type: ignore
from neural_network import NeuralNetwork


def recognize_on_click(drawing: CanvasResult, nn: NeuralNetwork) -> None:
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
    st.session_state.drawing_array_resized = matrix_mapper(drawing_array, 28, 28)

    # Reshaping to a vector
    digit = (255 - st.session_state.drawing_array_resized).reshape(784)

    # Normalizing the input
    normalized_digit = (digit - digit.mean()) / digit.std()

    # Use neural net to make prediction
    try:
        st.session_state.prediction = nn.predict(normalized_digit)
    except ValueError:
        st.error("I'm terribly sorry, I can't make that out 😭")
        st.stop()

    # Return guess with varying level of confidence
    st.session_state.guess = sorted(
        zip(st.session_state.prediction, range(10)), reverse=True
    )
    st.session_state.is_guessing = True


def print_guess() -> None:
    st.subheader("Best guess:")
    if st.session_state.guess[0][0] > 0.8:
        st.write(
            f":green[I'm _pretty_ sure thats's a {st.session_state.guess[0][1]}] 😁"
        )
    elif st.session_state.guess[0][0] > 0.4:
        st.write(f":orange[It kinda looks like a {st.session_state.guess[0][1]}] 🙂")
    elif st.session_state.guess[0][0] > 0.2:
        st.write(f":orange[It _could_ be a {st.session_state.guess[0][1]}] 🤔")
    else:
        st.write(f":red[My best guess is a {st.session_state.guess[0][1]}] 🫣")
    if st.session_state.guess[1][0] > 0.2:
        st.write(
            f":red[... but it could also be a {st.session_state.guess[1][1]}] 😵‍💫"
        )


def print_steps(nn: NeuralNetwork) -> None:
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
    plt.imshow(st.session_state.drawing_array_resized, cmap="plasma")  # type: ignore
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
    for i, activation in enumerate(nn.last_activations[:-1]):  # type: ignore
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
    ax.bar(
        height=st.session_state.prediction,
        x=[f"{i}" for i in range(10)],
        color="#f63366",
    )
    ax.tick_params(axis="x", colors="#f63366")
    ax.tick_params(axis="y", colors="#f63366")
    ax.set_xlabel("$i$", color="#f63366")
    ax.set_ylabel("$P(i)$", color="#f63366")
    plt.yticks(ticks=np.linspace(0, 1, 6))  # type: ignore
    st.pyplot(fig)

    st.write("Cool, huh? 😎")


def reset_on_click() -> None:
    st.session_state.drawing_array_resized = None
    st.session_state.prediction = None
    st.session_state.guess = None
    st.session_state.is_guessing = False
