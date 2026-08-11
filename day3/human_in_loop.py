import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    approval: str


@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    print(f"EMAIL SENT")
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")

    return "Email sent successfully."

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key = os.getenv("GROQ_API_KEY")
)

llm_with_tools = llm.bind_tools([send_email])

def agent(state:State):
    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }

def human_approval(state: State):
    last_message = state["messages"][-1]

    approval = interrupt({
        "message": "The agent wants to send an email. Approve?",
        "tool_call": last_message.tool_calls
    })

    return {
        "approval": approval
    }

tool_node = ToolNode([send_email])
graph = StateGraph(State)
graph.add_node("agent",agent)
graph.add_node("tools",tool_node)

graph.add_edge(START,"agent")
graph.add_conditional_edges("agent",lambda state: "approval" if state["messages"][-1].tool_calls else END)
graph.add_node("approval", human_approval)

graph.add_edge("approval", "tools")
graph.add_edge("tools", "agent")

# app = graph.compile()
checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)
config = {
    "configurable": {
        "thread_id": "email-approval-1"
    }
}
# response = app.invoke({
#     "messages":[
#         ("user","send an email to test@example.com saying hello.")
#     ]
# })
# print(response["messages"][-1])
response = app.invoke(
    {
        "messages": [
            (
                "user",
                "Send an email to test@example.com with subject 'Meeting' "
                "and body 'Let's meet tomorrow.'"
            )
        ]
    },
    config
)
response = app.invoke(
    Command(resume="approved"),
    config
)

print(response)
# print(response)


