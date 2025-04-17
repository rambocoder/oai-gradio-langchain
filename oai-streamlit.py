# streamlit run oai-streamlit.py
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from openai import OpenAI
import os

# Set up OpenAI API key (You'll need to have this in your environment variables)
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)  # Or set it directly if you prefer, but not recommended for security

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to get response from OpenAI API
def get_openai_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        message = completion.choices[0].message.content
        return message
    except Exception as e:
        return f"Error: {e}"  # Handle errors gracefully


# Function to handle example button clicks
def handle_example_click(example_text):
    st.session_state.messages.append({"role": "user", "content": example_text})  # Add user message
    response = get_openai_response(st.session_state.messages)  # Get AI response
    st.session_state.messages.append({"role": "assistant", "content": response})  # Store AI response
    st.rerun()

# Sidebar with example questions
with st.sidebar:
    st.header("Examples")
    if st.button("2+2=?", key="example1"):
        handle_example_click("2+2=?")
    if st.button("What is the capital of France?", key="example2"):
        handle_example_click("What is the capital of France?")
    if st.button("Write a short poem about cats.", key="example3"):
        handle_example_click("Write a short poem about cats.")


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
prompt = st.chat_input("Ask me anything...")


# Process the user's input
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})  # Store user message

    with st.chat_message("user"):
        st.markdown(prompt)  # Display user message

    # Get the AI's response
    response = get_openai_response(st.session_state.messages)

    # Store the AI's response
    st.session_state.messages.append({"role": "assistant", "content": response})  # Store assistant message

    with st.chat_message("assistant"):
        st.markdown(response) #Display assistant message
