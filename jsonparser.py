from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel
import os

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-405B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task = "text-generation",
    max_new_tokens=1050,
    temperature=0.7,
    huggingfacehub_api_token=api_token,
)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template=(
        "Give me the name, age and city of a fictional person.\n"
        "{format_instructions}\n"
        "json response:"
    ),
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template1 | llm | parser
result = chain.invoke({})

print(result)