from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-405B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task = "text-generation",
    max_new_tokens=1050,
    temperature=0.7,
    huggingfacehub_api_token=api_token,
)
print("Using structured output parsing")

class Fact(BaseModel):
    fact1: str = Field(description="First interesting fact about the topic")
    fact2: str = Field(description="Second interesting fact about the topic")
    fact3: str = Field(description="Third interesting fact about the topic")

parser = PydanticOutputParser(pydantic_object=Fact)

template = PromptTemplate(
    template=(
        "Provide three interesting facts about the following topic: {topic}\n"
        "{format_instructions}\n"
        "Structured response:"
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = template | llm | parser
result = chain.invoke({"topic": "Artificial General Intelligence"})
print(result)