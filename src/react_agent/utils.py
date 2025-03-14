"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import os


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    # Check if using OpenRouter (detected via OPENAI_API_BASE)
    if os.environ.get("OPENAI_API_BASE", "").startswith("https://openrouter.ai"):
        # For Claude models on OpenRouter, we need to use the ChatOpenAI class
        if provider == "anthropic" and "claude" in model:
            return ChatOpenAI(
                model=f"anthropic/{model}",
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
    
    # Default behavior for all other cases
    return init_chat_model(model, model_provider=provider)
