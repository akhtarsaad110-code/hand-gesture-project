import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="ASL Gesture Recognition",
    page_icon="🤟",
    layout="wide"
)

# Professional & Responsive Styling (CSS)
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<p class="main-header">🤟 ASL Sign Language Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Production-Grade Deep Learning Hand Gesture Recognition System</p>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gesture_model.h5')

with st.spinner("Loading AI model... Please wait."):
    model = load_model()

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Sidebar Navigation & Info
with st.sidebar:
    st.title("📊 Project Dashboard")
    st.info("This application utilizes a custom Convolutional Neural Network (CNN) trained on the Sign Language MNIST dataset for real-time gesture classification.")
    st.markdown("---")
    st.markdown("### 🔤 Supported Classes")
    st.write(", ".join(CLASSES))

# Main Responsive Two-Column Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a hand gesture image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Hand Gesture", use_column_width=True)

with col2:
    st.markdown("### 🔍 Prediction & Analytics")
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            processed_img = image.convert('L').resize((28, 28))
            img_array = np.array(processed_img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
        
        # Display Results in Professional Cards/Metrics
        st.success(f"### Predicted Letter: **{CLASSES[pred_idx]}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
        
        st.markdown("---")
        st.info("💡 **Tip:** For best results, upload a clear, well-lit hand gesture image.")
    else:
        st.info("👈 Please upload an image on the left panel to see model predictions.")
