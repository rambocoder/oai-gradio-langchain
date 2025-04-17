import warnings
import os
import json

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template("Tell me about a {adjective} {subject}.")
prompt = prompt_template.format(adjective="funny", subject="elephants")
print(prompt)
# https://huggingface.co/blog/langchain
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model="mixedbread-ai/mxbai-embed-large-v1",
    task="feature-extraction",
)
texts = ["Hello, world!", "How are you?"]
results = hf_embeddings.embed_documents(texts)
print(results)

llm = HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# llm = HuggingFaceEndpoint(
#     endpoint_url="https://router.huggingface.co/hf-inference/v1",
    
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )



chat = ChatHuggingFace(llm=llm, verbose=False)

messages = [
                ("system", "You are a helpful translator. Translate the user sentence to Ukrainian."),
                ("human", "I love programming."),
            ]

result = chat.invoke(messages)
print(result.content)





