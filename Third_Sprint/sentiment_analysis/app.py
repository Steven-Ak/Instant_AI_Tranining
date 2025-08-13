import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Hugging Face repo ID
HF_REPO_ID = "stevenakram/sentiment_bert_model"

# Download and load tokenizer & model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model = TFAutoModelForSequenceClassification.from_pretrained(HF_REPO_ID)

# Class names (update according to your model config)
class_names = ['Irrelevant', 'Negative', 'Neutral', 'Positive']

# Streamlit UI
st.title("Real-Time Sentiment Analysis")
st.write("Enter your text below and get the sentiment prediction!")

user_input = st.text_area("Your Text", "")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        # Tokenize input
        inputs = tokenizer(
            user_input,
            return_tensors="tf",
            truncation=True,
            padding='max_length',
            max_length=128
        )

        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        predicted_class_idx = int(tf.argmax(logits, axis=1).numpy())
        predicted_class = class_names[predicted_class_idx]

        # Display results
        st.success(f"Predicted Sentiment: **{predicted_class}**")
        st.write("### Class Probabilities:")
        for cls, prob in zip(class_names, probs):
            st.write(f"{cls}: {prob:.2%}")  # nicer percentage format