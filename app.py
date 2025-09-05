from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.jain_agent import chatbot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_headers = ["*"],
    allow_methods = ["*"],
)

class Query(BaseModel):
    query : str

@app.post("/chat")
async def chat_endpoint(req: Query):
    answer = await chatbot(req.query)
    return {"answer": answer}