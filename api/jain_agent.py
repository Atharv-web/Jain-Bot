import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage
from pinecone import Pinecone

embedding_model = None 
llm_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
    return embedding_model

def get_llm_model():
    global llm_model
    if llm_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash" ,api_key=api_key)    
    return llm_model

# pinecone setup
index_name = "jain-bot"
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key = pinecone_api_key)
index= pinecone_client.Index(index_name)
namespace = "jain-vidya-3"


async def chatbot(user_input: str) -> str:
    "Async Chatbot"
    embedding = get_embedding_model()
    vectorized_query = await asyncio.to_thread(lambda : embedding.encode(f"query: {user_input}", normalize_embeddings = True).tolist())
    
    search_result = index.query(
        namespace=namespace,
        vector=vectorized_query,
        top_k=3,
        include_metadata=True
    )

    retrieved_docs = [result['metadata']['text'] for result in search_result['matches']]

    db_context = "\n\n---\n\n".join(retrieved_docs)

    rag_prompt = """You are an assistant for question-answering tasks in Hindi.
    Use the below given context to answer question. Read and understand the context carefully.

    {context}

    Now, carefully read the question:

    {question}

    Provide a correct answer to this question by using the above context.
    Answer:"""

    formatted_prompt = rag_prompt.format(context=db_context,question=user_input)
    llm = get_llm_model()
    model_response = llm.invoke([HumanMessage(content=formatted_prompt)])
    return model_response.content