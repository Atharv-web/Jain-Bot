# import os
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage
# from dotenv import load_dotenv
# load_dotenv()

# # port = int(os.getenv("PORT", 8501))
# # st.set_option("server.port", port)
# # st.set_option("server.address", "0.0.0.0")

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("intfloat/multilingual-e5-large")

# @st.cache_resource
# def load_pinecone_index():
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     return pc.Index("jain-bot")

# # Load once
# EMBED_MODEL = load_embedding_model()
# INDEX = load_pinecone_index()
# NAMESPACE = "jain-vidya-3"


# def get_answer(user_input):
#     # Embed user query
#     query_vec = EMBED_MODEL.encode(f"query: {user_input}", normalize_embeddings=True).tolist()

#     # Pinecone query
#     res = INDEX.query(namespace=NAMESPACE, vector=query_vec, top_k=3, include_metadata=True)
#     retrieved_docs = [m['metadata']['text'] for m in res['matches']]
#     context = "\n\n---\n\n".join(retrieved_docs)

#     # Minimal prompt to reduce LLM processing time
#     prompt = f"""You are an assistant for answering questions in Hindi. The questions will be in hindi, so read the question and the context carefully to answer it.

#     Context:
#     {context}

#     Question:
#     {user_input}

#     Answer in Hindi:"""

#     llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
#     response = llm_model.invoke([HumanMessage(content=prompt)])
#     return response.content

# st.set_page_config(page_title="Jain-Bot")
# st.title("Ask your Doubts..")

# user_input = st.text_area("Enter..:", height=100)

# if st.button("Get Answer"):
#     if user_input.strip():
#         with st.spinner("Generating answer..."):
#             answer = get_answer(user_input)
#         st.subheader("Answer:")
#         st.write(answer)
#     else:
#         st.warning("Please type a question!")