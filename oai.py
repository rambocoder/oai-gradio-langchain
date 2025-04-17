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

def count_images(message, history):
    num_images = len(message["files"])
    total_images = 0
    for message in history:
        if isinstance(message["content"], tuple):
            total_images += 1
    return f"You just uploaded {num_images} images, total uploaded: {total_images+num_images}"

def process_message(message, history):
    # Extract text and images from the message
    text = message.get("text", "")
    files = message.get("files", [])

    # Prepare the input for OpenAI API
    inputs = []

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
        with open(file, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            })

    # Add the complete message to inputs
    if content:
        inputs.append({
            "role": "user",
            "content": content
        })

    # Call OpenAI API (GPT-4 with vision, if available)
    try:
        response = client.chat.completions.create(model="gpt-4o",  # Use GPT-4 with vision capabilities
        messages=inputs)
        # Extract the assistant's response
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {str(e)}"

    # Count the number of images in the current message
    num_images = len(files)

    # Count the total number of images in the history
    total_images = sum(len(msg.get("files", [])) for msg in history)

    # Return the assistant's response and image count summary
    return f"{reply}\n\nYou just uploaded {num_images} images, total uploaded: {total_images + num_images}"



demo = gr.ChatInterface(
    # fn=count_images, 
    fn=process_message, 
    type="messages", 
    examples=[
        {"text": "No files", "files": []}
    ], 
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload", "microphone"])
)

demo.launch()