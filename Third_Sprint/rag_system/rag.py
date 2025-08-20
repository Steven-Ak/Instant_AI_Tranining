from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline


# Load & Split PDF
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


# Create Vector DB
def create_vector_db(chunks, persist_path="db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db


# Setup Generator Model
def get_llm():
    gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0                      
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


# Build RAG chain
def build_qa_chain(db):
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff"
    )
    return qa