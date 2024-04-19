import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# Load the saved Bayesian model
model = load_model('mnist_bnn', compile=False)

st.title('MNIST Digit Classifier')

# Streamlit canvas for drawing digits
canvas_result = st_canvas(stroke_width=10, stroke_color='#ffffff',
                          background_color='#000000', height=150, width=150,
                          drawing_mode='freedraw',key='canvas')

def predict_digit_from_canvas(canvas_data):
    if canvas_data is not None:
        # Preprocessing
        img = cv2.resize(canvas_data.astype('uint8'), (28, 28))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)  # Reshape for the model

        # Prediction
        pred = model.predict(img)
        pred_digit = np.argmax(pred)
        return f'Predicted Digit: {pred_digit}'
    return "No digit drawn or image not processed correctly."

# Button to submit the drawing for prediction
if st.button('Submit'):
    prediction = predict_digit_from_canvas(canvas_result.image_data)
    st.write(prediction)

# Button to clear the canvas
if st.button('Clear Canvas'):
    # This will clear the canvas and the prediction display
    st.session_state[canvas_result.key] = None
    st.rerun()