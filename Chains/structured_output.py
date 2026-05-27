import os
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Annotated, Optional, Literal

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-8B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

review_text = "The hardware is great but software is bad. There are too many preinstalled apps you can't uninstall. I want to return this product"

"""
print("Using normal model")
result = llm.invoke(review_text)
print(result)
"""
print("\nUsing structured output parsing")

class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes mentioned in the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment of the review")
    summary: str = Field(description="A concise summary of the review")
    pros: Optional[list[str]] = Field(default=None, description="List of pros mentioned in the review")
    cons: Optional[list[str]] = Field(default=None, description="List of cons mentioned in the review")

parser = PydanticOutputParser(pydantic_object=Review)

prompt = PromptTemplate(
    template="""Respond with ONLY valid JSON matching this format, no other text.

{format_instructions}

Review to analyze: {review}

JSON response:""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

result = chain.invoke({"review": review_text})
print("Result: ", result)
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Key Themes:", result.key_themes)
print("Pros:", result.pros)
print("Cons:", result.cons)

#There is also a json schema which is used for cross language compatibility, but the pydantic output parser is more convenient to use in Python. 
# The json schema can be used with any language that can parse json, 
# and the format instructions are included in the prompt to ensure the model outputs valid json.