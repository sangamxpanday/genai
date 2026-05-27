from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

api_token = os.getenv("HUGGINGFACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a joke about the following topic: {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "Explain the joke: {joke}",
    input_variables=["joke"]
)

parser = StrOutputParser()

chain = RunnableSequence(template1 ,model ,parser, template2, model, parser)

print(chain.invoke({"topic": "programming"}))