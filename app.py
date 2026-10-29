import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

@st.cache_resource
def get_model():
 return tf.keras.models.load_model('gesture_model.h5')

model = get_model()
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

st.title("Sign Language Classifier 🤟")
st.write("Upload a hand gesture image to predict the letter.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption='Uploaded Image', width=150)
    
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0])) * 100
    
    st.success(f"Prediction: **{CLASSES[pred_idx]}** (Confidence: **{confidence:.2f}%**)")
