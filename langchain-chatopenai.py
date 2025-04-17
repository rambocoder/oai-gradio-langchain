import warnings
import os
import json

from dotenv import load_dotenv
load_dotenv()

from openai.types.chat import ChatCompletion
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import langchain_openai

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)

# Replace with your actual OpenAI API key or set the environment variable OPENAI_API_KEY
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the ChatOpenAI model
chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.0, model="gpt-3.5-turbo")

original_generate = ChatOpenAI._generate  # Store the original method
# https://github.com/huggingface/text-generation-inference/issues/2136#issuecomment-2198372112
def normalize_chat_complition(response: ChatCompletion) -> ChatCompletion:
    choices = []
    for choice in response.choices:
        if choice.message.tool_calls :
            tool_calls = []
            for tool_call in choice.message.tool_calls:
                tool_call.function.arguments = json.dumps(tool_call.function.arguments)
                tool_calls.append(tool_call)
            choice.message.tool_calls = tool_calls
        if not choice.message.content:
            choice.message.content = " "
        if choice.message.content:
            choice.message.content = "Overwritted!"
        choices.append(choice)
    return response
    # return ChatCompletion(
    #     id=response.id,
    #     choices=choices,
    #     created=response.created,
    #     model=response.model,
    #     object="chat.completion",
    #     system_fingerprint=response.system_fingerprint,
    #     usage=response.usage
    #     )

def patched_generate(self,
        messages,
        stop= None,
        run_manager = None,
        **kwargs):
    print("patched_generate")
    if self.streaming:
        stream_iter = self._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return generate_from_stream(stream_iter)
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    generation_info = None
    if "response_format" in payload:
        if self.include_response_headers:
            warnings.warn(
                "Cannot currently include response headers when response_format is "
                "specified."
            )
        payload.pop("stream")
        response = self.root_client.beta.chat.completions.parse(**payload)
    elif self.include_response_headers:
        raw_response = self.client.with_raw_response.create(**payload)
        response = raw_response.parse()
        generation_info = {"headers": dict(raw_response.headers)}
    else:
        response = normalize_chat_complition(self.client.create(**payload))
    return self._create_chat_result(response, generation_info)
    # print(f"Intercepted call to create with prompt: {prompt}")
    # # Modify the prompt or arguments as needed
    # modified_prompt = f"Modified: {prompt}"
    # kwargs['prompt'] = modified_prompt  # Update the prompt in kwargs

    # Call the original method with the potentially modified arguments
    # return original_generate(*args, **kwargs)

# Replace the original method with the patched version
ChatOpenAI._generate = patched_generate

# def custom_generate(self, *args, **kwargs):
#     print("Custom generate method called")
#     # You can call the original method if you need to
#     original_generate = getattr(self.__class__, '_generate')
#     return original_generate(*args, **kwargs)
# class_to_patch = getattr(langchain_openai, 'ChatOpenAI')
# setattr(class_to_patch, '_generate', custom_generate)
# Create the messages
msgs = [
    SystemMessage(content="Respond in Spanish."),
    HumanMessage(content="Hi!")
]

# Get the response
response = chat(msgs)

# Print the assistant's message
print(response.content)
