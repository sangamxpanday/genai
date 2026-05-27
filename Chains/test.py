from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-405B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task = "text-generation",
    max_new_tokens=256,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = "what is capital city of {country}?",
    input_variables=["country"]
)

chain = template1 | llm | parser

result = chain.invoke({"country": "Nepal"})
print(result)
chain.get_graph().print_ascii()