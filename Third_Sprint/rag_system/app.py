import streamlit as st
from rag import load_documents, create_vector_db, build_qa_chain

st.title("📘 RAG Q&A System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = "data/" + uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("✅ PDF uploaded and being processed")

    docs = load_documents(pdf_path)
    db = create_vector_db(docs)
    qa = build_qa_chain(db)

    st.write("✅ PDF processed")

    query = st.text_input("Ask a question about the PDF")
    if query:
        answer = qa.run(query)
        st.write("**Answer:** ", answer)