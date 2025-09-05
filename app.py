from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.jain_agent import chatbot
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_headers = ["*"],
    allow_methods = ["*"],
)

@app.get("/")
async def read_index():
    return FileResponse(os.path.join("public", "index.html"))

@app.post("/chat")
async def chat_endpoint(req:Request):
    body = await req.json()
    user_query = body.get("query","")
    answer = await chatbot(user_query)
    return {"answer": answer}