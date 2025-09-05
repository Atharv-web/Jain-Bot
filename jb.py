import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone

# Load environment variables once
load_dotenv()

# --- Cache heavy models/clients globally ---
_embedding_model = None
_llm_model = None
_index = None
_namespace = "jain-vidya-3"


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Load once, reuse
        _embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
    return _embedding_model


def get_llm_model():
    global _llm_model
    if _llm_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        _llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
    return _llm_model


def get_pinecone_index():
    global _index
    if _index is None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        _index = pinecone_client.Index("jain-bot")
    return _index


# --- Main Chatbot Function ---
def chatbot(user_input: str) -> str:
    embedding_model = get_embedding_model()
    vectorized_query = embedding_model.encode(
        f"query: {user_input}", normalize_embeddings=True
    ).tolist()

    index = get_pinecone_index()
    search_result = index.query(
        namespace=_namespace,
        vector=vectorized_query,
        top_k=3,
        include_metadata=True
    )

    retrieved_docs = [res["metadata"]["text"] for res in search_result["matches"]]
    db_context = "\n\n---\n\n".join(retrieved_docs)

    rag_prompt = f"""You are an assistant for question-answering tasks in Hindi.
    Use the below given context to answer question. Read and understand the context carefully.

    {db_context}

    Now, carefully read the question:

    {user_input}

    Provide a correct answer to this question by using the above context.
    Answer:"""

    llm = get_llm_model()
    model_response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=rag_prompt)
    ])

    return model_response.content