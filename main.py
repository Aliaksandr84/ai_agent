import os
import dotenv
import json

from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-proxy.lab.epam.com"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-oss-120b"
os.environ["AZURE_OPENAI_APIVERSION"] = "2023-07-01-preview"

#print("api_key = ", os.environ["API_KEY"].strip())

client = AzureOpenAI(
    api_key=os.environ["API_KEY"].strip(),
    api_version=os.environ["AZURE_OPENAI_APIVERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"]
)

#print("client = ", client.api_key)

messages = [
               {"role": "system", "content": "You are a helpful assistant. You believe in AI."},
               {"role": "user", "content": "Tell me about the members of the Beatles."}
           ]
print("----------------------------------")

#response = client.chat.completions.create(
#    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
#    messages=messages
#)
#print("----------------------------------")
#print(response.choices[0].message.content)

print("----------------------------------")
#messages.append({"role": "system", "content": response.choices[0].message.content})

messages.append({"role": "user", "content": "What is the best rock band in 2050?"})

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=messages
)

print(response.choices[0].message.content)

def music_prediction(question: str) -> str:
    return "The Beatles will be resurrected by AI and will be the best rock band in 2050."

tools = [
    {
        "type": "function",
        "function": {
            "name": "music_prediction",
            "description": "Retrieve prediction of the rock music",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The query string to get prediction",
                    },
                },
                "required": ["question"],
            },
        },
    }
]

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=messages,
    tools=tools
)

print("Response content:")
print(response.choices[0].message.content)

print("Full response:")
print(response)

# If llm designed to use tools, we can extract the tool calls from the response
if response.choices[0].message.tool_calls:
        print("Full Function Response:")
        print(response.choices[0].message.tool_calls[0])
        print()
        print("Function Name:")
        print(response.choices[0].message.tool_calls[0].function.name)
        print("Function Arguments:")
        print(response.choices[0].message.tool_calls[0].function.arguments)

arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
question = arguments.get('question')
        
#tool_call_result = music_prediction(question)

#print("Function Response:")
#print(tool_call_result)

#messages.append(response.choices[0].message)
#messages.append({"role": "tool", "content": json.dumps({"result": tool_call_result}), "tool_call_id": #response.choices[0].message.tool_calls[0].id})

#response = client.chat.completions.create(
#    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
#    messages=messages
#)

#print(response.choices[0].message.content)

model = AzureChatOpenAI(
    api_key=os.environ["API_KEY"].strip(),
    api_version=os.environ["AZURE_OPENAI_APIVERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"]
)

response = model.invoke([HumanMessage(content="What is the best rock band in 2050?")])
print("^^^^^^^^^ = ", response.content)
