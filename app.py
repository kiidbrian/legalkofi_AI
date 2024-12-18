import os
import pinecone
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name) # delete the temp file

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", ".", "?", "!", " ", ""])
    return text_splitter.split_documents(docs)

def get_or_create_vector_collection() -> pinecone.Index:
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Get or create index
    index_name = "legalkofi"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)

def add_to_vector_store(documents: list[Document], file_name: str):
    # Generate embeddings for the documents
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Extract text content and metadata from documents
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"{file_name}_{i}" for i in range(len(documents))]  # Generate unique IDs

    # Generate embeddings for all texts
    embeddings = embedding_model.encode(texts)
    
    collection = get_or_create_vector_collection()
    
    # Batch upsert in chunks of 1000
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadata = metadatas[i:i + batch_size]
        # Upsert to Pinecone
        vectors_to_upsert = zip(batch_ids, batch_embeddings, batch_metadata)
        collection.upsert(vectors=[(id, emb.tolist(), meta) 
                            for id, emb, meta in vectors_to_upsert])
    
    st.success("Data added to the vector database successfully!")

if __name__ == "__main__":
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="LegalKofi")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "⚡️ Process",
        )

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_store(all_splits, normalize_uploaded_file_name)
            st.write(all_splits)

    # Question and Answer Area
    st.header("⚖️ LegalKofi")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

    if ask and prompt:
        pass