from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import os

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-405B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task = "text-generation",
    max_new_tokens=1050,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

#1st Prompt - Detailed report
template1 = PromptTemplate(
    template = "Write a report on the following topic: {topic}",
    input_variables=["topic"]
)

#2nd Prompt- Summary
template2 = PromptTemplate(
    template = "Summarize the following text in 5 sentences: {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "Artificial General Intelligence"})
result = llm.invoke(prompt1)

prompt2 = template2.invoke({"text": result})
result2 = llm.invoke(prompt2)
print(result2)