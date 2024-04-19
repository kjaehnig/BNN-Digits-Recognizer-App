import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the saved Bayesian model
model = load_model('mnist_bnn', compile=False)

st.title('MNIST Digit Classifier')

# Streamlit canvas for drawing digits
canvas_result = st.canvas(stroke_width=10, stroke_color='#ffffff',
                          background_color='#000000', height=150, width=150,
                          drawing_mode='freedraw')

if canvas_result.image_data is not None:
    # Preprocess the canvas image for prediction
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict the digit
    pred = model.predict(img)
    st.write(f'Predicted Digit: {np.argmax(pred)}')

if st.button('Clear'):
    st.experimental_rerun()