import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# App Title
st.set_page_config(page_title="Brain Tumor AI Diagnosis", layout="centered")
st.title("🧠 Brain Tumor AI Diagnosis")
st.write("Upload an MRI image and choose the model type:")

# Model Selection
model_choice = st.radio("Choose a model:", ["Classification", "Segmentation"])

# Load Models
@st.cache_resource
def load_classifier():
    path = r"C:\Instant_AI_Training\Second Sprint\brain_tumor_classifier_improved_v4.keras"
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_segmenter():
    path = r"C:\Instant_AI_Training\Second Sprint\brain_tumor_segmentation_unet.keras"
    return tf.keras.models.load_model(path, compile=False)

classifier_model = load_classifier()
segmenter_model = load_segmenter()

# File Upload
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# Classification Prediction
def classify(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    prediction = classifier_model.predict(image_array)[0][0]
    return prediction

# Segmentation Prediction
def segment(image: Image.Image):
    image = image.convert("L")                      # Convert to grayscale
    image = image.resize((128, 128))                # Match UNet input size
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Shape: (1, 128, 128, 1)
    mask = segmenter_model.predict(image_array)[0, :, :, 0]  # Shape: (128, 128)
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

# Optional Overlay Function
def overlay_mask(original_image: Image.Image, mask):
    original = original_image.resize((128, 128)).convert("RGB")
    original_array = np.array(original)
    red_mask = np.zeros_like(original_array)
    red_mask[:, :, 0] = mask  # Apply mask to red channel
    overlay = cv2.addWeighted(original_array, 0.7, red_mask, 0.3, 0)
    return overlay

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            if model_choice == "Classification":
                prediction = classify(image)
                if prediction >= 0.5:
                    st.error("🧠 Tumor Detected")
                else:
                    st.success("✅ No Tumor Detected")
                st.write(f"**Confidence:** {prediction:.2f}")
            else:
                mask = segment(image)
                overlay = overlay_mask(image, mask)
                st.success("✅ Tumor Segmentation Complete")
                st.image(overlay, caption="🧠 Tumor Overlay", use_column_width=True)
                st.image(mask, caption="🩻 Tumor Mask", use_column_width=True)