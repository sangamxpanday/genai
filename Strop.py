from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

result =chain.invoke({"topic": "Artificial General Intelligence"})

print(result)