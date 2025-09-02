import json
from typing import List, Optional
# from typing import list
from fastapi.params import Depends
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


import redis
app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")

# Serve the chat.html file at the root URL
app.mount("/socket.io/", socketio.ASGIApp(sio))
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Configure Redis connection
# Make sure Redis is running and accessible at this address
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
r = redis.asyncio.Redis(host=redis_host, port=redis_port, decode_responses=True)

# Configure Socket.IO Redis adapter
sio.pubsub_manager = socketio.AsyncRedisManager(f"redis://{redis_host}:{redis_port}")

# Local dictionary to store sid to username mapping for this instance
local_users = {}

@sio.event
async def connect(sid, environ):
    print('connect ', sid)
    local_users[sid] = 'Anonymous' # Initialize locally

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)
    await update_user_list()

async def update_user_list():
    current_users = await r.hgetall('users')
    await sio.emit('user_list', {'users': list(current_users.values())})

@sio.on("join")
async def join(sid, data):
    print(f"User with sid {sid} set username to {data}")
    local_users[sid] = data # Update locally
    await r.hset('users', sid, data)
    await update_user_list()

@sio.event
async def message(sid, data):
    sender_username = local_users.get(sid, 'Anonymous') # Get username from local dictionary
    current_users = await r.hgetall('users')

    if message_text.startswith('/msg '):
        parts = message_text.split(' ', 2)
        if len(parts) >= 3:
            recipient_username = parts[1]
            private_message = parts[2]
            for user_sid, username in current_users.items():
                if username == recipient_username:
                    await sio.emit('message', {'user': sender_username, 'message': f'(Private) {private_message}'}, room=user_sid)
                    await sio.emit('message', {'user': sender_username, 'message': f'(To {recipient_username}) {private_message}'}, room=sid)
                    return
            await sio.emit('message', {'user': 'System', 'message': f'User {recipient_username} not found.'}, room=sid)
        else:
            await sio.emit('message', {'user': 'System', 'message': 'Invalid /msg command. Use /msg <username> <message>'}, room=sid)
    else:
        await sio.emit('message', {'user': sender_username, 'message': data})

# uv pip install .
# uvicorn server:app --reload --port 8090