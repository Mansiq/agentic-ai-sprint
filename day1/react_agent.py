import os
import ast
import operator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------
# Calculator Tool
# -----------------------

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def safe_eval(node):

    if isinstance(node, ast.Constant):
        return node.value

    elif isinstance(node, ast.BinOp):
        return OPS[type(node.op)](
            safe_eval(node.left),
            safe_eval(node.right),
        )

    elif isinstance(node, ast.UnaryOp):
        return OPS[type(node.op)](
            safe_eval(node.operand)
        )

    else:
        raise ValueError("Invalid Expression")


def calculator(expression):

    tree = ast.parse(expression, mode="eval")

    return safe_eval(tree.body)


# -----------------------
# Search Tool
# -----------------------

def search(query):

    database = {
        "eiffel tower": "1889",
        "capital of france": "Paris",
        "largest ocean": "Pacific Ocean",
    }

    query = query.lower()

    for key in database:

        if key in query:
            return database[key]

    return "No information found."


# -----------------------
# System Prompt
# -----------------------

system_prompt = """
You are a ReAct Agent.

You have two tools.

calculator(expression)

search(query)

Rules:

Think step by step.

If you need calculation respond ONLY as:

ACTION: calculator
INPUT: 18*245/100

If you need information respond ONLY as:

ACTION: search
INPUT: Eiffel Tower completion year

When you know the answer respond ONLY as:

FINAL: answer

Only ONE action per response.

Wait for Observation before next action.
"""


# -----------------------
# User Question
# -----------------------

question = "What is 18% of 245 then add the year the Eiffel Tower was completed?"

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": question
    }
]

# -----------------------
# ReAct Loop
# -----------------------

while True:

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0
    )

    reply = response.choices[0].message.content.strip()

    print("\nAssistant:")
    print(reply)

    messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )

    # -----------------------
    # Final Answer
    # -----------------------

    if reply.startswith("FINAL:"):

        print("\nAnswer:")
        print(reply.replace("FINAL:", "").strip())

        break

    # -----------------------
    # Calculator
    # -----------------------

    elif reply.startswith("ACTION: calculator"):

        expression = reply.split("INPUT:")[1].strip()

        result = calculator(expression)

        print("Calculator:", result)

        messages.append(
            {
                "role": "user",
                "content": f"Observation: {result}"
            }
        )

    # -----------------------
    # Search
    # -----------------------

    elif reply.startswith("ACTION: search"):

        query = reply.split("INPUT:")[1].strip()

        result = search(query)

        print("Search:", result)

        messages.append(
            {
                "role": "user",
                "content": f"Observation: {result}"
            }
        )

    else:

        print("Unexpected response.")
        break