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
    img = cv2.resize(image_data.astype('uint8'), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def plot_prediction_probs(probs):
    fig, ax = plt.subplots()
    ax.bar(range(10), probs.squeeze(), tick_label=range(10))
    plt.xlabel('Digits')
    plt.ylabel('Probability')
    return fig


# Load the saved Bayesian model
model = load_model('mnist_bnn',
                   compile=True,
                   custom_objects={'neg_loglike':neg_loglike,
                                   'divergence':divergence})

st.title('MNIST Digit Classifier')

# Streamlit canvas for drawing digits
canvas_result = st_canvas(stroke_width=10, stroke_color='#ffffff',
                          background_color='#000000', height=150, width=150,
                          drawing_mode='freedraw',key='canvas')

def predict_digit_from_canvas(canvas_data):
    if canvas_data is not None:
        # Preprocessing
        img = process_image(canvas_data)
        # Prediction
        pred = model(img)
        print(pred)
        pred_digit = np.argmax(pred)
        st.pyplot(plot_prediction_probs(pred))
        return f'Predicted Digit: {pred_digit}'
    return "No digit drawn or image not processed correctly."

# Button to submit the drawing for prediction
if st.button('Submit'):
    prediction = predict_digit_from_canvas(canvas_result.image_data)
    st.write(prediction)
