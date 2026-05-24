import os
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-8B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)
print("Using normal model")
review_text = "The hardware is great but software is bad. There are too many preinstalled apps you can't uninstall. I want to return this product"
result = llm.invoke(review_text)
print(result)
print("\nUsing structured output parsing")

class Review(BaseModel):
    summary: str
    sentiment: str

parser = PydanticOutputParser(pydantic_object=Review)

prompt = PromptTemplate(
    template="""Respond with ONLY valid JSON matching this format, no other text.

{format_instructions}

Example response:
{{"summary": "Brief summary here", "sentiment": "positive or negative"}}

Review to analyze: {review}

JSON response:""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

try:
    result = chain.invoke({"review": review_text})
    print(result)
except Exception as e:
    print(f"Error: {e}")
    raw_chain = prompt | llm
    raw_result = raw_chain.invoke({"review": review_text})
    print(f"Raw LLM output: {raw_result}")