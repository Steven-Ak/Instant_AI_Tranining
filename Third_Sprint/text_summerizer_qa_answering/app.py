import streamlit as st
from transformers import pipeline
import PyPDF2
import docx

# Cached model loaders
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def get_qa_model():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

st.title("📄 Text Summarizer & Question Answering")

# File Upload
uploaded_file = st.file_uploader("Upload PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])
text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    try:
        if file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
        elif file_type == "txt":
            text = uploaded_file.read().decode("utf-8")
        
        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        
        st.success(f"Loaded {file_type.upper()} file successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    text = st.text_area("Or paste your article or document here:", height=300)

# Summarization
if text.strip():
    if st.button("Summarize Text"):
        with st.spinner("Loading summarizer and summarizing..."):
            try:
                summarizer = get_summarizer()
                # Avoid very long input errors by truncating
                summary = summarizer(text[:3000], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error during summarization: {e}")

    # Question Answering
    st.subheader("Ask a Question about the Text")
    question = st.text_input("Your question:")
    if question:
        with st.spinner("Loading QA model and finding answer..."):
            try:
                qa_model = get_qa_model()
                answer = qa_model(question=question, context=text[:3000])
                st.write(f"**Answer:** {answer['answer']} (score: {answer['score']:.2f})")
            except Exception as e:
                st.error(f"Error during question answering: {e}")
