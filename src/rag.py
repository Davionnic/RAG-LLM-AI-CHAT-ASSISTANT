"""Conversational RAG chain: history-aware retrieval + grounded answering."""

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.ingest import get_vector_store
from src.llm import get_chat_model

REPHRASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question as a standalone question that can be understood without "
            "the history. Do NOT answer it; just rephrase it (or return it "
            "unchanged if it is already standalone).",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions strictly using "
            "the provided context from the user's documents.\n"
            "- Answer only from the context below. Do not use outside knowledge.\n"
            "- If the context does not contain the answer, say you could not "
            "find it in the documents.\n"
            "- Be concise and cite the source file name(s) you used.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


_chain = None


def build_chain():
    """Build (and cache) the RAG chain."""
    global _chain
    if _chain is None:
        llm = get_chat_model()
        retriever = get_vector_store().as_retriever(search_kwargs={"k": 4})
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, REPHRASE_PROMPT
        )
        answer_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
        _chain = create_retrieval_chain(history_aware_retriever, answer_chain)
    return _chain


def to_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    """Convert [(role, content), ...] pairs into LangChain messages."""
    messages: list[BaseMessage] = []
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def answer(
    question: str, chat_history: list[tuple[str, str]] | None = None
) -> tuple[str, list[Document]]:
    """Answer a question. Returns (answer_text, source_documents)."""
    chain = build_chain()
    result = chain.invoke(
        {"input": question, "chat_history": to_messages(chat_history or [])}
    )
    return result["answer"], result.get("context", [])
