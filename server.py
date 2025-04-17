import json
from typing import List, Optional
# from typing import list
from fastapi.params import Depends
from langchain_openai import ChatOpenAI
from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import (
    StreamingResponse,
)

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()
app = FastAPI()

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
    async for patch in chat.astream_log(messages):
        for op in patch.ops:
            if op["op"] == "add" and op["path"] == "/streamed_output/-":
                content = (
                    op["value"] if isinstance(op["value"], str) else op["value"].content
                )
                json_dict = {"type": "llm_chunk", "content": content}
                json_str = json.dumps(json_dict)
        
                yield f"data: {json_str}\n\n"
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
        media_type="text/event-stream",
    )

class HelloWorldParams(BaseModel):
    content: Optional[str] = "Default parameter"

@app.get("/hello")
async def hello(params: HelloWorldParams = Depends()):
    return {"params": params.model_dump()}

# uvicorn server:app --reload