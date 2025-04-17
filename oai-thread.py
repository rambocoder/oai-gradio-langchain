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

def get_image_base64(file):
    with open(file, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            }

def process_message(message, history):
    # Initialize the messages list with system message if needed, or start directly with history
    # messages = [] if not history else history
    messages = [
        {"role": msg['role'], 
         "content": [get_image_base64(msg['content'][0])] if isinstance(msg['content'], tuple) 
         else [{"type": "text", "text": msg['content']}]
        } 
        for msg in history
    ] if history else []


    # Extract text and images from the message
    text = message.get("text", "")
    files = message.get("files", [])

    # Add text input if available
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

    # Add the complete message to inputs
    if content:
        messages.append({
            "role": "user",
            "content": content
        })

    # Call OpenAI API (GPT-4 with vision, if available)
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",  # Use GPT-4 with vision capabilities
        messages=messages)
        # Extract the assistant's response
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {str(e)}"

    # Return the assistant's response and image count summary
    return reply





demo = gr.ChatInterface(
    fn=process_message, 
    type="messages", 
    examples=[
        {"text": "2+2=?", "files": []}
    ], 
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload", "microphone"])
)

demo.launch()