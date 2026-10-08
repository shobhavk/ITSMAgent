"""Q&A (Agent) tab: chat index and chat handlers."""
import pandas as pd

from app.services import rag
from ui.chat_format import clean_answer


def _build_chat_index(full_df: pd.DataFrame) -> "rag.TicketIndex":
    """Rebuilds the chat/RAG index from the latest analysis dataframe.
    Cheap and synchronous - embeddings are computed lazily on first
    question (see rag.TicketIndex._ensure_vectors), not here. Also the
    index the "Detect Similar Recurring Issues" button clusters over -
    whichever of chat or that button runs first pays the one embedding
    call; the other reuses the same cached vectors for free."""
    return rag.build_index_from_dataframe(full_df)


def _history_to_messages(history: list[tuple[str, str]]) -> list[dict]:
    """Converts the internal (question, answer) tuple history - the shape
    rag.answer_question expects for prompt context - into the role/content
    message dicts gr.Chatbot renders in this Gradio version."""
    messages = []
    for question, answer in history:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


async def _chat_respond(message: str, history: list, full_df, stats: dict):
    message = (message or "").strip()
    if not message:
        return _history_to_messages(history), history, ""

    if full_df is None or len(full_df) == 0:
        answer = "Run an analysis on the Dashboard tab first - then come back and ask away."
    else:
        # The model (or the no-LLM fallback) sometimes answers with a JSON blob; show readable text.
        answer = clean_answer(await rag.answer_question(message, full_df, stats or {}, history))

    history = history + [(message, answer)]
    return _history_to_messages(history), history, ""


def _chat_clear():
    return [], []
