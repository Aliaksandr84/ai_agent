from langchain_openai import AzureChatOpenAI
from pprint import pprint
import os

from openai import AzureOpenAI

deployment_name = "deepseek-r1" # select anyone from the list above

azure_llm = AzureChatOpenAI(
		deployment_name=deployment_name,
        max_tokens=800,
        #top_p=0.95,
        #frequency_penalty=0,
        #presence_penalty=0,
        stop=None,
	api_key = os.environ["API_KEY"].strip(),  # Put your DIAL API Key here
	api_version = "2025-11-13",
	azure_endpoint = "https://ai-proxy.lab.epam.com"
	)

response = azure_llm.invoke("this is a try to test the model")
content = response.content
response_metadata = response.response_metadata
pprint(content)
pprint(response_metadata)

def perform_prompt(input_prompt: str):
    output_response = azure_llm.invoke(input_prompt)
    output_content = output_response.content
    pprint(output_content)