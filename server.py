import json
from typing import List, Optional
# from typing import list
from fastapi.params import Depends
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import (
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
import os

import socketio

# Load environment variables from .env file
load_dotenv()

# Access the OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()
app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins=[], async_mode="asgi")

# Serve the chat.html file at the root URL
app.mount("/", StaticFiles(directory=".", html=True), name="static")
app.mount("/socket/", socketio.ASGIApp(sio))

class Message(BaseModel):
    id: str
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[Message]
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"messages": [{"id": "1", "role": "user", "content": "Hello"}]}
            ]
        }
    }

async def send_completion_events(messages, chat: ChatOpenAI):
    chat_messages = []
    for message in messages:
        if message["role"] == "user":
            chat_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            chat_messages.append(AIMessage(content=message["content"]))

    async for patch in chat.astream_log(messages):
        for op in patch.ops:
            if op["op"] == "add" and op["path"] == "/streamed_output/-":
                content = (
                    op["value"] if isinstance(op["value"], str) else op["value"].content
                )
                yield content
        # yield f"data: {json.dumps(patch.to_dict())}\n\n"
    # for message in messages:
    #     completion = chat.send_message(message["content"])
    #     yield f"data: {json.dumps({'id': message['id'], 'role': 'bot', 'content': completion})}\n\n"    

@app.post("/api/completion")
async def stream(request: Request, payload: ChatPayload):
    messages = [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content
        }
        for message in payload.messages
    ]
    chat = ChatOpenAI()
    return StreamingResponse(
        send_completion_events(messages, chat=chat),
        media_type="text/plain",
    )

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def chat_message(sid, data):
    print(f"Received message from {sid}: {data}")
    messages = data.get("messages", [])
    chat = ChatOpenAI()
    async for chunk in send_completion_events(messages, chat=chat):
        await sio.emit("chat_chunk", {"sid": sid, "chunk": chunk})
    await sio.emit("chat_end", {"sid": sid})



class HelloWorldParams(BaseModel):
    content: Optional[str] = "Default parameter"

@app.get("/hello")
async def hello(params: HelloWorldParams = Depends()):
    return {"params": params.model_dump()}

# uvicorn server:app --reload