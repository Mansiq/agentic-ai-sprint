import ast
import operator

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_tavily import TavilySearch

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------
# 1. SAFE CALCULATOR
# --------------------------------------------------

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_calculate(expression):

    def evaluate(node):

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numbers are allowed")

        if isinstance(node, ast.BinOp):

            left = evaluate(node.left)
            right = evaluate(node.right)

            op_type = type(node.op)

            if op_type not in allowed_operators:
                raise ValueError("Operator not allowed")

            return allowed_operators[op_type](left, right)

        if isinstance(node, ast.UnaryOp):

            operand = evaluate(node.operand)

            op_type = type(node.op)

            if op_type not in allowed_operators:
                raise ValueError("Operator not allowed")

            return allowed_operators[op_type](operand)

        raise ValueError("Invalid expression")

    tree = ast.parse(expression, mode="eval")

    return evaluate(tree.body)


# --------------------------------------------------
# 2. TOOLS
# --------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression."""

    try:
        return str(safe_calculate(expression))

    except Exception as e:
        return f"Calculation error: {e}"


tavily_search = TavilySearch(
    max_results=5
)
# @tool
# def search(query: str) -> str:
#     """Search for factual information. Currently a stub."""

#     return (
#         f"Search result for '{query}': "
#         "The Eiffel Tower was completed in 1889."
#     )


tools = [
    calculator,
    tavily_search
]


# --------------------------------------------------
# 3. LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# 4. SYSTEM PROMPT
# --------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful reasoning agent.

You have two tools:

calculator(expression):
Use this tool for ALL arithmetic calculations.
The calculator accepts standard Python-style arithmetic expressions.

For percentages, NEVER use the % symbol to mean percentage.
For example:
- 18% of 245 -> 18 * 245 / 100
- 20% of 500 -> 20 * 500 / 100

tavily_search(query):
Use this tool when you need factual information from the web.

For multi-step questions:
1. Break the problem into smaller steps.
2. Identify which tool is needed for each step.
3. Execute the steps sequentially when one step depends on the result of another.
4. Wait for the result of a tool before using that result in another tool call.
5. If a calculation requires information from a web search, perform the web search FIRST.
6. Then use the returned information in the calculator.
7. Perform ALL arithmetic using the calculator tool.
8. Never calculate arithmetic mentally when the calculator can be used.
9. Never invent tool results.
10. If a tool returns an error, correct the input and call the tool again.
11. Do not make dependent tool calls in parallel.

When finished, answer the user's complete question, not just one part of it.
"""
# --------------------------------------------------
# 5. STATE
# --------------------------------------------------

class State(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# --------------------------------------------------
# 6. LLM NODE
# --------------------------------------------------

def call_model(state: State):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# --------------------------------------------------
# 7. TOOL NODE
# --------------------------------------------------

tool_node = ToolNode(tools)


# --------------------------------------------------
# 8. ROUTING
# --------------------------------------------------

def should_continue(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


# --------------------------------------------------
# 9. GRAPH
# --------------------------------------------------

graph = StateGraph(State)

graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

# Generate graph visualization
png_data = app.get_graph().draw_mermaid_png()

with open("langgraph_agent.png", "wb") as f:
    f.write(png_data)

print("Graph saved as langgraph_agent.png")


# --------------------------------------------------
# 10. TEST
# --------------------------------------------------

question = """
What is 18% of 245, then add the year
the Eiffel Tower was completed?
"""

result = app.invoke({
    "messages": [
        ("user", question)
    ]
})


for message in result["messages"]:

    print("\n====================")

    print(type(message).__name__)

    print(message.content)

    if hasattr(message, "tool_calls"):
        print("Tool calls:", message.tool_calls)