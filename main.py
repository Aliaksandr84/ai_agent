import os
import dotenv

from openai import AzureOpenAI

dotenv.load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-proxy.lab.epam.com"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o-2024-05-13"
os.environ["AZURE_OPENAI_APIVERSION"] = "2023-07-01-preview"

print("----------------------------------")

client = AzureOpenAI(
    api_key=os.environ["OPENAI_API_KEY"].strip(),
    api_version=os.environ["AZURE_OPENAI_APIVERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"]
)

messages = [
               {"role": "system", "content": "You are a helpful assistant. You believe in AI."},
               {"role": "user", "content": "Tell me about the members of the Beatles."}
           ]

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=messages
)
print("----------------------------------")
print(response.choices[0].message.content)

print("----------------------------------")
messages.append({"role": "system", "content": response.choices[0].message.content})
