import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# -------------------------
# 1. Initialize LLM
# -------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# -------------------------
# 2. Define State
# -------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# -------------------------
# 3. Define Agent Node
# -------------------------
def agent(state: State):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# -------------------------
# 4. Build Graph
# -------------------------
graph = StateGraph(State)

graph.add_node("agent", agent)

graph.add_edge(START, "agent")


# -------------------------
# 5. Add Memory
# -------------------------
checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)


# -------------------------
# 6. Create Thread
# -------------------------
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}


# -------------------------
# 7. First Message
# -------------------------
response = app.invoke(
    {
        "messages": [
            ("user", "My name is Mansi.")
        ]
    },
    config
)

print(response["messages"][-1].content)


# -------------------------
# 8. Second Message
# -------------------------
response = app.invoke(
    {
        "messages": [
            ("user", "What's my name?")
        ]
    },
    config
)

print(response["messages"][-1].content)


# -------------------------
# 9. Third Message
# -------------------------
response = app.invoke(
    {
        "messages": [
            ("user", "What am I learning?")
        ]
    },
    config
)

print(response["messages"][-1].content)


# -------------------------
# 10. Different Thread
# -------------------------
config_user2 = {
    "configurable": {
        "thread_id": "user-2"
    }
}

response = app.invoke(
    {
        "messages": [
            ("user", "What's my name?")
        ]
    },
    config_user2
)

print(response["messages"][-1].content)