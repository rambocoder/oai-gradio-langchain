from datetime import datetime
from typing import Optional
from langgraph.graph import MessagesState
from langchain.schema import SystemMessage


class MyMessagesState(MessagesState):
    config: dict
    LastUpdatedDateTime: Optional[str]


class InitSystemPromptNode:
    """LangGraph node for initializing system prompt"""

    def __init__(self, system_prompt: str = "return only numbers"):
        self.system_prompt = system_prompt

    async def __call__(self, state: MyMessagesState, config=None) -> dict:
        """Add system prompt if this is the first invocation"""
        messages = state.get("messages", [])
        if not messages:
            messages = []
        messages.insert(0, SystemMessage(content=self.system_prompt))

        return {
            "messages": messages,
            "config": {"initialized": True, "system_prompt_added": True},
        }
