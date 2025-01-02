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
You are an AI assistant designed to assist legal professionals in Ghana by answering questions about legal documents using provided context. Your role is to analyze the given context carefully and provide accurate, detailed, and well-organized answers that directly address the questions posed.

Input Structure

The relevant document information will be provided as "Context:"
The question will be provided as "Question:"

Guidelines for answering questions:
1. Thorough Analysis: Carefully examine the context, identifying and extracting all relevant details needed to answer the question accurately.
2. Logical Organization: Plan your response to ensure the information flows logically and comprehensively.
3. Contextual Relevance: Base your entire answer solely on the information provided in the context. Avoid adding external knowledge, personal opinions, or assumptions.
4. Clarity and Detail: Provide a detailed and precise response, ensuring it is both comprehensive and easy to understand. Address all aspects of the question if the context permits.
5. Acknowledge Gaps: If the context lacks sufficient information to answer the question fully, clearly state this and explain the limitation.

Formatting Guidelines:

1. Clear Language: Use straightforward, professional language appropriate for legal professionals.
2. Structured Responses: Organize your answer into paragraphs for clarity.
3. Lists for Complex Information: Use bullet points or numbered lists to simplify complex details or multi-step explanations.
4. Headings/Subheadings: Include headings or subheadings if they enhance the structure of your response.
5. Professional Presentation: Maintain proper grammar, punctuation, and spelling throughout your answer.

Important:
Your analysis and response must strictly adhere to the provided context. Any unsupported information, external references, or speculation is strictly prohibited.
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
            options=[LLMConfig.OPENAI],
            index=0
        )
        model_name = st.selectbox(
            "**🤖 Model Name**",
            options=LLMConfig.PROVIDERS[model_provider]["available_models"],
            index=0
        )

        # st.divider()

        # uploaded_file = st.file_uploader(
        #     "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        # )

        # process = st.button(
        #     "⚡️ Process",
        # )

        # if uploaded_file and process:
        #     normalize_uploaded_file_name = uploaded_file.name.translate(
        #         str.maketrans({"-": "_", ".": "_", " ": "_"})
        #     )
        #     all_splits = process_document(uploaded_file)
        #     add_to_vector_store(all_splits, normalize_uploaded_file_name)
        #     st.write(all_splits)

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
    st.header("⚖️🇬🇭 LegalKofi")
    st.markdown("###### 🤖 ✨ Your AI-powered assistant for smarter legal work.")
    prompt = st.chat_input(" What can I help you with?")

    if prompt:
        results = query_vector_store(prompt)
        context = results[0].page_content
        response = call_llm(context, prompt, model_provider, model_name)
        st.write_stream(response)
