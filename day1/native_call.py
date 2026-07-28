# import os
# import ast
# import operator
# import json
# from pathlib import Path

# from dotenv import load_dotenv
# from groq import Groq

# BASE_DIR = Path(__file__).resolve().parent.parent

# load_dotenv(BASE_DIR / ".env")

# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY")
# )

# # -----------------------
# # Calculator Tool
# # -----------------------

# OPS = {
#     ast.Add: operator.add,
#     ast.Sub: operator.sub,
#     ast.Mult: operator.mul,
#     ast.Div: operator.truediv,
#     ast.USub: operator.neg,
# }


# def safe_eval(node):

#     if isinstance(node, ast.Constant):
#         return node.value

#     elif isinstance(node, ast.BinOp):
#         return OPS[type(node.op)](
#             safe_eval(node.left),
#             safe_eval(node.right),
#         )

#     elif isinstance(node, ast.UnaryOp):
#         return OPS[type(node.op)](
#             safe_eval(node.operand)
#         )

#     else:
#         raise ValueError("Invalid Expression")


# def calculator(expression):

#     tree = ast.parse(expression, mode="eval")

#     return safe_eval(tree.body)


# # -----------------------
# # Search Tool
# # -----------------------

# def search(query):

#     database = {
#         "eiffel tower": "1889",
#         "capital of france": "Paris",
#         "largest ocean": "Pacific Ocean",
#     }

#     query = query.lower()

#     for key in database:

#         if key in query:
#             return database[key]

#     return "No information found."


# calculator_tool = {
#     "type": "function",
#     "function": {
#         "name": "calculator",
#         "description": "Evaluate a mathematical expression safely.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "expression": {
#                     "type": "string",
#                     "description": "A mathematical expression to evaluate."
#                 }
#             },
#             "required": ["expression"]
#         }
#     }
# }

# search_tool = {
#     "type":"function",
#     "function":{
#         "name":"search",
#         "description":"Search for factual information.",
#         "parameters":{
#             "type":"object",
#             "properties":{
#                 "query":{
#                     "type":"string",
#                     "description":"The search query"
#                 }
#             },
#             "required":["query"]
#         }
#     }
# }

# question = "What is 18% of 245 then add the year the Eiffel Tower was completed?"

# system_prompt = """
# You are a helpful AI assistant.

# Use tools whenever needed.

# Rules:

# 1. For calculator, ALWAYS generate valid Python arithmetic expressions.

# Good examples:
# 18*245/100
# 0.18*245
# (18*245)/100 + 1889

# Bad examples:
# 18% of 245
# twenty plus five
# 18 percent of 245

# 2. For search, provide only the search query.

# Do not invent facts.
# """

# messages = [
#     {
#         "role": "system",
#         "content": system_prompt
#     },
#     {
#         "role": "user",
#         "content": question
#     }
# ]

# response = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     messages=messages,
#     tools=[calculator_tool, search_tool],
#     tool_choice="auto",
#     temperature=0,
# )

# tool_calls = response.choices[0].message.tool_calls

# print(tool_calls)
# print(tool_calls[0])
# print(tool_calls[0].function.name)
# print(tool_calls[0].function.arguments)
# # print(response.choices[0].message.content)

# # -----------------------
# # Extract Tool Information
# # -----------------------

# # -----------------------
# # Execute All Tool Calls
# # -----------------------

# for tool_call in tool_calls:

#     tool_name = tool_call.function.name

#     arguments = json.loads(tool_call.function.arguments)

#     print("\nTool Name:", tool_name)
#     print("Arguments:", arguments)

#     # Execute the correct tool
#     if tool_name == "calculator":

#         result = calculator(arguments["expression"])

#     elif tool_name == "search":

#         result = search(arguments["query"])

#     else:

#         result = "Unknown Tool"

#     print("Tool Output:", result)

#     # Append assistant's tool request only once
#     if response.choices[0].message not in messages:
#         messages.append(response.choices[0].message)

#     # Append this tool's result
#     messages.append(
#         {
#             "role": "tool",
#             "tool_call_id": tool_call.id,
#             "content": str(result),
#         }
#     )


# # -----------------------
# # Ask Groq Again
# # -----------------------


# final_response = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     messages=messages,
#     tools=[calculator_tool, search_tool],
#     temperature=0,
# )
# print("\nFinal Answer:")
# print(final_response.choices[0].message.content)

import os
import ast
import operator
import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------
# Load API Key
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------
# Safe Calculator
# ---------------------------------------------------

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.BinOp):
        return OPS[type(node.op)](
            safe_eval(node.left),
            safe_eval(node.right),
        )

    if isinstance(node, ast.UnaryOp):
        return OPS[type(node.op)](
            safe_eval(node.operand)
        )

    raise ValueError("Unsupported expression")


def calculator(expression):
    tree = ast.parse(expression, mode="eval")
    return safe_eval(tree.body)


# ---------------------------------------------------
# Search Tool (Stub)
# ---------------------------------------------------

DATABASE = {
    "capital of france": "Paris",
    "largest ocean": "Pacific Ocean",
    "eiffel tower completed": "1889",
    "president of india": "Droupadi Murmu",
}


def search(query):

    query = query.lower()

    for key in DATABASE:
        if key in query:
            return DATABASE[key]

    return "No information found."


# ---------------------------------------------------
# Python Tool Map
# ---------------------------------------------------

tool_functions = {
    "calculator": calculator,
    "search": search,
}

# ---------------------------------------------------
# Tool Schemas
# ---------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate arithmetic expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression."
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for factual information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------
# Agent
# ---------------------------------------------------

def run_agent(question):

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    while True:

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            # model = "llama-3.1-8b-instant",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # ------------------------
        # Final Answer
        # ------------------------

        if not message.tool_calls:

            return message.content

        messages.append(message)

        # ------------------------
        # Execute each tool
        # ------------------------

        for tool_call in message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(tool_call.function.arguments)

            print(f"\nCalling Tool : {function_name}")
            print("Arguments    :", arguments)
            result = tool_functions[function_name](**arguments)

            # if function_name == "calculator":

            #     result = tool_functions[function_name](
            #         arguments["expression"]
            #     )

            # elif function_name == "search":

            #     result = tool_functions[function_name](
            #         arguments["query"]
            #     )

            # else:

            #     result = "Unknown Tool"

            print("Observation  :", result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    questions = [

        "What is 18% of 245, then add the year the Eiffel Tower was completed?",

        "What is 200*15?",

        "What is the capital of France?",

        "What is the largest ocean?",
    ]

    for q in questions:

        print("\n")
        print("=" * 80)
        print("QUESTION:", q)

        answer = run_agent(q)

        print("\nFINAL ANSWER")
        print(answer)