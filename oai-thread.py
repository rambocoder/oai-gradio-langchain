import base64
import gradio as gr
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Access the OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize public and shadow chat histories
public_history = []
shadow_history = []

def get_image_base64(file):
    with open(file, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}"
            }
        }

def guard_message(message):
    """Check if the message contains the word 'duck'."""
    if "duck" in message.get("text", "").lower():
        return True
    return False

def process_message(message, history):
    global public_history, shadow_history

    # Check for the guard condition
    if guard_message(message):
        public_history.append({"role": "user", "content": [{"type": "text", "text": message["text"]}]})
        public_history.append({"role": "assistant", "content": [{"type": "text", "text": "not up in here"}]})
        return "not up in here"

    # Update shadow history
    shadow_history = history if history else []
    shadow_messages = [
        {"role": msg['role'], 
         "content": [get_image_base64(msg['content'][0])] if isinstance(msg['content'], tuple) 
         else [{"type": "text", "text": msg['content']}]
        } 
        for msg in shadow_history
    ] if shadow_history else []

    # Extract text and images from the message
    text = message.get("text", "")
    files = message.get("files", [])

    # Initialize content list for the message
    content = []

    # Add text input if available
    if text:
        content.append({
            "type": "text",
            "text": text
        })

    # Add image inputs if available
    for file in files:
        value = get_image_base64(file)
        content.append(value)

    # Add the complete message to shadow history
    if content:
        shadow_messages.append({
            "role": "user",
            "content": content
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # vision capabilities
            messages=shadow_messages
        )
        # Extract the assistant's response
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {str(e)}"

    # Update public history
    public_history.append({"role": "user", "content": [{"type": "text", "text": text}]})
    public_history.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})

    
    return reply

demo = gr.ChatInterface(
    fn=process_message, 
    type="messages", 
    examples=[
        {"text": "2+2=?", "files": []},
        {"text": "duck you", "files": []}
    ], 
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload", "microphone"])
)

demo.launch()