from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.jain_agent import chatbot
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_headers = ["*"],
    allow_methods = ["*"],
)

app.mount("/", StaticFiles(directory="public", html=True), name="static")

class Query(BaseModel):
    query : str

@app.post("/chat")
async def chat_endpoint(req: Query):
    answer = await chatbot(req.query)
    return {"answer": answer}