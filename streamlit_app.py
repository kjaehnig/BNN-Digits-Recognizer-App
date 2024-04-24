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
    fig, ax = plt.subplots(figsize=(3,6))
    ax.bar(probs.squeeze(), range(10), tick_label=range(10))
    ax.set_title("BNN Predictions")
    plt.xlabel('Probability')
    plt.ylabel('Digit')
    return fig

def plot_preprocessed_image(img):
    fig, imgax = plt.subplots(figsize=(1.,1.))
    imgax.imshow(img.reshape(28,28, 1), cmap='gray')
    imgax.set_title('What model sees', fontsize=8)
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

# Initialize or get the counter from session state
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0

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
        st.write(pred)
        # pred = np.percentile(pred, 50, axis=0)  # Median over samples
        pred = np.sum(pred, axis=0) / num_samples
        pred_digit = np.argmax(pred)

        return img, pred, pred_digit
    return "No digit drawn or image not processed correctly."

col1, col2 = st.columns(2)

with col1:
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

    # Sampling number input
    N = st.slider('N (Number of Samplings)', min_value=0, max_value=50, value=1)
    if N > 10:
        st.warning("Setting N above 10 may slow down the predictions.")

    # Button to submit the drawing for prediction
img = None

if st.button('Submit'):
    img, pred, pred_digit = predict_digit_from_canvas(canvas_result.image_data, N)
    st.write(pred_digit)

if img is not None:
    feedback = st.radio("Is the prediction correct?", ('Yes', 'No', 'Submit Response'), index=2)
    if feedback in ('Yes', 'No'):
        if 'last_input' not in st.session_state or st.session_state.last_input != feedback:
            st.session_state.last_input = feedback
            if feedback == 'Yes':
                st.session_state.correct_predictions += 1
                st.success("Thank you for your feedback!")
            else:
                st.error("Sorry, let's try another one!")

    st.write(f"Correct Predictions: {st.session_state.correct_predictions}")



with col2:
    if img is not None:
        st.pyplot(plot_preprocessed_image(img))
        st.pyplot(plot_prediction_probs(pred))