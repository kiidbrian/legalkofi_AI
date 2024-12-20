import os
import pinecone
import tempfile
import ollama
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

system_prompt = """
You are an AI assistant tasked with answering questions solely about a legal document given context from the document. Your goal is to analyze the document and provide a detailed and accurate answer to the question.

context will be passed as "Context:"
question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

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

def query_vector_store(prompt: str, top_k: int = 5) -> list[Document]:
    collection = get_or_create_vector_collection()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode(prompt).tolist()
    results = collection.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    response = ollama.chat(
        model="llama3.2",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

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
        results = query_vector_store(prompt)
        context = results["matches"][0]
        response = call_llm(context, prompt)
        st.write_stream(response)
