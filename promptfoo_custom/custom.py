# my_script.py
import json
import time
from typing import Any, Dict
from typing import Any, Dict, List, Optional, Union


class ProviderOptions:
    id: Optional[str]
    config: Optional[Dict[str, Any]]


class CallApiContextParams:
    vars: Dict[str, str]


class TokenUsage:
    total: int
    prompt: int
    completion: int


class ProviderResponse:
    output: Optional[Union[str, Dict[str, Any]]]
    error: Optional[str]
    tokenUsage: Optional[TokenUsage]
    cost: Optional[float]
    cached: Optional[bool]
    logProbs: Optional[List[float]]


class ProviderEmbeddingResponse:
    embedding: List[float]
    tokenUsage: Optional[TokenUsage]
    cached: Optional[bool]


class ProviderClassificationResponse:
    classification: Dict[str, Any]
    tokenUsage: Optional[TokenUsage]
    cached: Optional[bool]


def generate_response(prompt_text: str, additional_option: str) -> str:
    """
    Simulates an LLM response based on simple keyword matching in the prompt.
    In a real scenario, this function would call an actual LLM API or model.
    """
    prompt_lower = prompt_text.lower()

    # Simulate processing time
    time.sleep(0.1)

    # if config and config.get("simulate_error"):
    #     raise Exception("Simulated error based on config")
    # Simulate a simple keyword-based response
    if "hello" in prompt_lower:
        return "Hello! How can I assist you today?"
    elif "what is your name" in prompt_lower:
        return "I am a simulated LLM."

    if "first us president" in prompt_lower:
        return "George Washington"
    elif "world war ii end" in prompt_lower:
        return "1945"
    elif "capital of france" in prompt_lower:
        return "Paris"
    elif "what is 2+2?" in prompt_lower and additional_option == "UUUUUUUUUUUUUUUUUUU":
        return "2"
    else:
        return "I don't have information on that specific question."


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> ProviderResponse:
    # Note: The prompt may be in JSON format, so you might need to parse it.
    # For example, if the prompt is a JSON string representing a conversation:
    # prompt = '[{"role": "user", "content": "Hello, world!"}]'
    # You would parse it like this:
    # prompt = json.loads(prompt)
    # print(f"Prompt after parsing: {prompt}")

    # The 'options' parameter contains additional configuration for the API call.
    config = options.get("config", None)
    additional_option = config.get("additionalOption", None)
    print(f"Additional option: {additional_option}")

    # The 'context' parameter provides info about which vars were used to create the final prompt.
    user_variable = context["vars"].get("userVariable", None)
    print(f"User variable: {user_variable}")

    # The prompt is the final prompt string after the variables have been processed.
    # Custom logic to process the prompt goes here.
    # For instance, you might call an external API or run some computations.
    # TODO: Replace with actual LLM API implementation.
    def call_llm(prompt):
        return f"Stub response for prompt: {prompt}"

    output = generate_response(prompt, additional_option)

    # The result should be a dictionary with at least an 'output' field.
    result = {
        "output": output,
    }

    # if some_error_condition:
    #     result['error'] = "An error occurred during processing"

    # if token_usage_calculated:
    #     # If you want to report token usage, you can set the 'tokenUsage' field.
    #     result['tokenUsage'] = {"total": token_count, "prompt": prompt_token_count, "completion": completion_token_count}

    # if failed_guardrails:
    #     # If guardrails triggered, you can set the 'guardrails' field.
    #     result['guardrails'] = {"flagged": True}

    return result


def call_embedding_api(prompt: str) -> ProviderEmbeddingResponse:
    # Returns ProviderEmbeddingResponse
    pass


def call_classification_api(prompt: str) -> ProviderClassificationResponse:
    # Returns ProviderClassificationResponse
    pass
