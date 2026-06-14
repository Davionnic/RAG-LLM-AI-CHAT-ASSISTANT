"""Factories for the Groq chat model and local Hugging Face embeddings."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

_chat_model: BaseChatModel | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def get_chat_model() -> BaseChatModel:
    global _chat_model
    if _chat_model is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        _chat_model = ChatGroq(
            model=config.CHAT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0,
            max_tokens=512,
        )
    return _chat_model


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    return _embeddings
