# Simple example using Hugging Face Inference API (no local download needed)
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-8B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=128,
    temperature=0.7,
    huggingfacehub_api_token=api_token,
)

print("Testing Hugging Face API Model...")
print("-" * 50)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a technology expert."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{query}"),])
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
print(chat_history)
prompt = chat_template.invoke({'chat_history': chat_history, 'query': "Who am i?"   })
print(prompt)
response = llm.invoke(prompt)
print("Response from Hugging Face API Model:")
print(response)