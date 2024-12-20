import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, Index
from config import LLMConfig
from openai import OpenAI
from ollama import Client
# from services import query_llama3
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

def get_or_create_vector_collection() -> Index:
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
    metadatas = [{
        'text': doc.page_content,
        **doc.metadata
    } for doc in documents]
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

    # Convert results to Documents
    documents = []
    for match in results['matches']:
        # Create Document from metadata
        doc = Document(
            page_content=match['metadata']['text'],
            metadata={
                k: v for k, v in match['metadata'].items() if k != 'text'
            }
        )
        documents.append(doc)
    
    return documents

def call_llm(context: str, prompt: str, model_provider: str = "ollama", model_name: str = "llama3.2"):
    """Calls the specified language model with context and prompt to generate a response.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question
        model_provider: String indicating the provider ("ollama" or "openai")
        model_name: String specifying the model to use (e.g., "llama2", "gpt-4")

    Yields:
        String chunks of the generated response as they become available

    Raises:
        Exception: If there are issues communicating with the model APIs
    """
    if not LLMConfig.validate_model(model_provider, model_name):
        raise ValueError(f"Invalid model: {model_name} for provider: {model_provider}")

    def get_messages(formatted_prompt):
        return [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]

    def handle_ollama_response(response):
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break

    def handle_openai_response(response): 
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    formatted_prompt = f"Context: {context}\nQuestion: {prompt}"
    messages = get_messages(formatted_prompt)

    if model_provider == LLMConfig.OLLAMA:
        # response = query_llama3(messages)
        client = Client()
        response = client.chat(
            model=model_name,
            stream=True,
            messages=messages
        )
        yield from handle_ollama_response(response)

    elif model_provider == LLMConfig.OPENAI:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            stream=True, 
            messages=messages
        )
        yield from handle_openai_response(response)

    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

if __name__ == "__main__":
    # Document Upload Area
    model_provider, model_name = None, None
    with st.sidebar:
        st.sidebar.title("Configure your LLM")

        model_provider = st.selectbox(
            "**🔧 Model Provider**",
            options=LLMConfig.PROVIDERS.keys(),
            index=0
        )
        model_name = st.selectbox(
            "**🤖 Model Name**",
            options=LLMConfig.PROVIDERS[model_provider]["available_models"],
            index=0
        )

        st.divider()

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

        st.sidebar.divider()
        st.sidebar.markdown(
            """
            <div style="text-align: center;color: grey;">
                Made with ♡ by <a href="https://paapa.dev" style="color: grey;">kiidbrian</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Question and Answer Area
    st.header("⚖️ LegalKofi")
    st.markdown("###### Powered by OpenAI GPT-4o and Llama 3.3 from Hugging Face 🦙")
    prompt = st.chat_input(" What can I help you with?")

    if prompt:
        results = query_vector_store(prompt)
        context = results[0].page_content
        response = call_llm(context, prompt, model_provider, model_name)
        st.write_stream(response)
