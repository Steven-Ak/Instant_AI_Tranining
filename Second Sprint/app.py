import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource
def load_brain_model(path):
    model = load_model(path)
    return model

model = load_brain_model(r"C:\Instant_AI_Training\Second Sprint\brain_tumor_classifier_improved_v2.h5")

# Define class names (change if needed)
class_names = ['no', 'yes']

# Streamlit UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to classify the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Change if your model uses a different size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.markdown(f"### 🧬 Prediction: `{class_names[predicted_class]}`")
    st.markdown(f"### 🔍 Confidence: `{confidence * 100:.2f}%`")
