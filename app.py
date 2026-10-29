import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Sign Language Classifier",
    page_icon="🤟",
    layout="centered"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1E1E1E;
        border: 2px solid #FF4B4B;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-title">Sign Language Gesture Classifier 🤟</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Based Hand Gesture Recognition System for ASL</div>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def get_model():
    return tf.keras.models.load_model('gesture_model.h5')

with st.spinner('Model load ho raha hai, bara-e-karam intezar karein...'):
    model = get_model()

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Sidebar info
st.sidebar.title("ℹ️ Project Info")
st.sidebar.info(
    "Yeh application Sign Language MNIST dataset par trained custom Convolutional Neural Network (CNN) model ka istemal karti hai "
    "taaki hand gestures ko accurately classify kiya ja sake."
)

# File Uploader Section
uploaded_file = st.file_uploader("🖼️ Yahan hand gesture image upload karein...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert('L').resize((28, 28))
        # Display enlarged preview for better UX
        st.image(Image.open(uploaded_file), caption='Original Image', use_column_width=True)
    
    with col2:
        st.subheader("Model Prediction")
        with st.spinner('Predicting...'):
            img_array = np.array(image) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
        
        st.markdown(f"""
            <div class="prediction-box">
                <h2 style='color: #4CAF50; margin:0;'>Letter: {CLASSES[pred_idx]}</h2>
                <p style='font-size: 16px; margin-top: 10px;'>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
