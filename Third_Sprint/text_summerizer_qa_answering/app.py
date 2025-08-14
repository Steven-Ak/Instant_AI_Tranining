import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
import torch

# Load both models once and keep cached in memory
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn", 
        device=device
    )
    qa_model = pipeline(
        "question-answering", 
        model="bert-large-uncased-whole-word-masking-finetuned-squad",
        device=device
    )
    return summarizer, qa_model

summarizer, qa_model = load_models()

st.title("📄 Text Summarizer & Question Answering")

# File Upload
uploaded_file = st.file_uploader("Upload PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])

text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    if file_type == "pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
    
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    st.success(f"Loaded {file_type.upper()} file successfully!")

else:
    # Manual paste
    text = st.text_area("Or paste your article or document here:", height=300)

# Summarization
if text.strip():
    if st.button("Summarize Text"):
        with torch.no_grad():  # Speed up inference
            with st.spinner("Summarizing..."):
                summary = summarizer(
                    text, 
                    max_length=150, 
                    min_length=50, 
                    do_sample=False
                )[0]['summary_text']
        st.subheader("Summary")
        st.write(summary)

    # Question Answering
    st.subheader("Ask a Question about the Text")
    question = st.text_input("Your question:")
    if question:
        with torch.no_grad():
            with st.spinner("Finding answer..."):
                answer = qa_model(question=question, context=text)
        st.write(f"**Answer:** {answer['answer']} (score: {answer['score']:.2f})")
