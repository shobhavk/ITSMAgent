"""
Standalone diagnostic for the chat/RAG tool-calling path. Run this directly
against your real .env config to isolate a bind_tools/ainvoke failure from
the rest of the app:

    python debug_chat_tools.py

If this fails, the traceback IS the real cause of "I couldn't reach the
assistant model" - paste it back for a targeted fix.
"""
import asyncio
import traceback

from app.services.llm_client import get_chat_model
from app.services.rag import _make_tools, TicketIndex


async def main():
    print("1) Checking get_chat_model()...")
    model = get_chat_model()
    if model is None:
        print("   -> get_chat_model() returned None. Check your .env / LLM_PROVIDER "
              "config and llm_client.py's factory - the chat model never initialized, "
              "so the app is silently in rule_based mode. This is the most common cause.")
        return
    print(f"   -> got a model: {type(model)}")

    print("2) Checking bind_tools()...")
    index = TicketIndex([{"ticket_id": "INC1", "category": "Network issues", "priority": "P1",
                           "status": "Closed", "host": "SW01", "assignment_group": "Network Team",
                           "worklog_score": 80, "text": "sample ticket text"}])
    tools = _make_tools(index)
    try:
        model_with_tools = model.bind_tools(tools)
        print("   -> bind_tools() succeeded")
    except Exception:
        print("   -> bind_tools() FAILED - this is your root cause:")
        traceback.print_exc()
        return

    print("3) Checking a real .ainvoke() call with tools bound...")
    try:
        from langchain_core.messages import HumanMessage
        response = await model_with_tools.ainvoke([HumanMessage(content="How many Network issues tickets are there?")])
        print("   -> ainvoke() succeeded")
        print("   -> tool_calls:", getattr(response, "tool_calls", None))
        print("   -> content:", response.content)
    except Exception:
        print("   -> ainvoke() FAILED - this is your root cause:")
        traceback.print_exc()
        return

    print("\nAll checks passed - the tool-calling path works in isolation.")
    print("If chat still fails end-to-end, the issue is likely in the tool execution")
    print("loop itself (check the logger.exception output from answer_question).")


asyncio.run(main())
