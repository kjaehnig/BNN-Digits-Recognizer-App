import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/60000.

def process_image(image_data):
    # Preprocess the canvas image for prediction
    if image_data.shape[2] == 4:
        image_data = image_data[:, :, :3]
    gryimg = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    invimg = gryimg
    img = cv2.resize(invimg, (28, 28),
               interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    # img = np.expand_dims(img, axis=-1) / 255.0
    img = img[np.newaxis, :]
    return img

def plot_prediction_probs(probs):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(range(10), probs.squeeze(), tick_label=range(10))
    ax.set_title("BNN Predictions")
    plt.xlabel('Probability')
    plt.ylabel('Digit')
    return fig

def plot_preprocessed_image(img):
    fig, imgax = plt.subplots(figsize=(1.,1.))
    imgax.imshow(img.reshape(28,28, 1), cmap='gray')
    imgax.set_title('What model sees', fontsize=4)
    imgax.tick_params(left=False,
                      bottom=False,
                      labelleft=False,
                      labelbottom=False)
    return fig
# Load the saved Bayesian model
model = load_model('mnist_bnn',
                   compile=False,)
                   # custom_objects={'neg_loglike':neg_loglike,
                   #                 'divergence':divergence})


# Initialize session state variables if they don't already exist
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0
if 'incorrect_predictions' not in st.session_state:
    st.session_state.incorrect_predictions = 0

# st.write(f"Correct Predictions: {st.session_state.correct_predictions}")
# st.write(f"Incorrect Predictions: {st.session_state.incorrect_predictions}")


if "yes_checkbox_val" not in st.session_state:
    st.session_state["yes_checkbox_val"] = False
if 'no_checkbox_val' not in st.session_state:
    st.session_state['no_checkbox_val'] = False

st.title('MNIST Digit Classifier')

# Streamlit canvas for drawing digits
# canvas_result = st_canvas(stroke_width=10, stroke_color='#ffffff',
#                           background_color='#000000', height=200, width=200,
#                           drawing_mode='freedraw',key='canvas')

# def predict_digit_from_canvas(canvas_data):
#     if canvas_data is not None:
#         # Preprocessing
#         img = process_image(canvas_data.astype('float32'))
#         # st.pyplot(plot_preprocessed_image(img))
#         # Prediction
#         pred = model(img)
#         # st.write(pred.numpy().shape)
#         pred = np.percentile(pred.numpy(), 50, axis=0)
#         # st.write(pred.T)
#         pred_digit = np.argmax(pred)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.pyplot(plot_preprocessed_image(img))
#         with col2:
#             st.pyplot(plot_prediction_probs(pred))
#         # st.pyplot(plot_prediction_probs(pred))
#         # return f'Predicted Digit: {pred_digit}'
#         return img, pred, pred_digit
#     return "No digit drawn or image not processed correctly."
#
# # Button to submit the drawing for prediction
# if st.button('Submit'):
#     img, pred, pred_digit = predict_digit_from_canvas(canvas_result.image_data)
#     st.write(f'Predicted digit: {pred_digit}')
# Side-by-side canvas and results
def predict_digit_from_canvas(canvas_data, num_samples):
    if canvas_data is not None:
        # Preprocessing
        img = process_image(canvas_data.astype('float32'))

        # Prediction
        # pred = model.predict(img, batch_size=num_samples)  # Assume model.predict handles BNN sampling
        pred = np.array([model(img).numpy().squeeze() for ii in range(num_samples)])
        # st.write(pred)
        # pred = np.percentile(pred, 50, axis=0)  # Median over samples
        pred = np.sum(pred, axis=0) / num_samples
        pred_digit = np.argmax(pred)

        return img, pred, pred_digit
    return "No digit drawn or image not processed correctly."


def clear_selection():
    for key in st.session_state.keys():
        if key.startswith("User_input_on_prediction"):
            st.session_state[key] = "False"

col1, col2 = st.columns(2)
with col1:
    with st.container():
        # Streamlit canvas for drawing digits
        canvas_result = st_canvas(
            stroke_width=10, 
            stroke_color='#ffffff',
            background_color='#000000', 
            height=300, 
            width=300,
            drawing_mode='freedraw', 
            key='canvas',
            update_streamlit=True)


with st.sidebar:
    st.header("Control Panel")
    # Sampling number input
    N = st.slider('N (Number of Samplings)', min_value=0, max_value=50, value=1)
    if N > 10:
        st.warning("Setting N above 10 may slow down the predictions.")

pred_digit = None
if pred_digit is None:
    st.session_state.disabled=True

img = None
    # Button to submit the drawing for prediction
if st.button('Submit'):
    img, pred, pred_digit = predict_digit_from_canvas(canvas_result.image_data, N)
    st.write(f"Predicted digit: {pred_digit}")

with col2:
    if img is not None:
        with st.container():
            st.write("What model sees")
            st.image(img.reshape(28,28,1), 
                clamp=True,
                use_column_width='always')

            st.write("Probabilities")
            st.bar_chart(x=[0,1,2,3,4,5,6,7,8,9], 
                y=pred.squeeze())


def register_prediction_checkbox():
    if st.session_state.yes_checkbox_val:
        st.session_state.correct_predictions += 1
        with st.sidebar:
            st.write("Thanks for responding!")
    elif st.session_state.no_checkbox_val:
        st.session_state.incorrect_predictions += 1
        with st.sidebar:
            st.write("Whoops! Let's try again!")

with st.sidebar:
    st.header("Is the model correct?")
    feedback = st.form(
        "Is the model correct?", 
        clear_on_submit=True,
        )

    feedback.checkbox('Yes', value=False, key='yes_checkbox_val')
    feedback.checkbox('No', value=False, key='no_checkbox_val')

    feedback.form_submit_button("Submit", 
        on_click=register_prediction_checkbox,
        disabled=True if img is None else False)


    st.write(f"Correct Predictions: {st.session_state.correct_predictions}")
    st.write(f"Incorrect Predictions: {st.session_state.incorrect_predictions}")

