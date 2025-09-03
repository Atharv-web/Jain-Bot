import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage
from pinecone import Pinecone


def chatbot(user_input):
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
    vectorized_query = embedding_model.encode(f"query: {user_input}", normalize_embeddings = True).tolist()
    
    index_name = "jain-bot"
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_client = Pinecone(api_key = pinecone_api_key)
    index= pinecone_client.Index(index_name)
    namespace = "jain-vidya-3"
    
    search_result = index.query(
        namespace=namespace,
        vector=vectorized_query,
        top_k=3,
        include_metadata=True
    )

    retrieved_docs = [result['metadata']['text'] for result in search_result['matches']]

    db_context = "\n\n---\n\n".join(retrieved_docs)
# formatted_prompt = rag_prompt.format(context=context, question=user_input)
    rag_prompt = """You are an assistant for question-answering tasks in Hindi.
    Use the below given context to answer question. Read and understand the context carefully.

    {context}

    Now, carefully read the question:

    {question}

    Provide a correct answer to this question by using the above context.
    Answer:"""

    formatted_prompt = rag_prompt.format(context=db_context,question=user_input)

    api_key = os.getenv("GEMINI_API_KEY")
    llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash" ,api_key=api_key)
    model_response = llm_model.invoke([SystemMessage(content=formatted_prompt)])
    return model_response.content
