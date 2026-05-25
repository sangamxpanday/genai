from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-405B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task = "text-generation",
    max_new_tokens=1050,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = "Write a report on the following topic: {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "Summarize the following text: {text}",
    input_variables=["text"]
)

chain = template1 | llm | template2 | llm | parser
result = chain.invoke({"topic": "Artificial General Intelligence"})
print(result)

chain.get_graph().print_ascii()