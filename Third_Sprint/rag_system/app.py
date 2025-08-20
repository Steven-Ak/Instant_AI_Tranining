import streamlit as st
import tempfile
from rag import load_documents, create_vector_db, build_qa_chain

st.title("📘 RAG Q&A System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.write("✅ PDF uploaded and being processed...")

    docs = load_documents(pdf_path)
    db = create_vector_db(docs)
    qa = build_qa_chain(db)

    st.write("✅ PDF processed")

    query = st.text_input("Ask a question about the PDF")
    if query:
        answer = qa.run(query)
        st.write("**Answer:** ", answer)
