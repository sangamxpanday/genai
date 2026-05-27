from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

api_token = os.getenv("HUGGINGFACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

chat_model = ChatHuggingFace(llm=llm)

# ------------------------
# Classification Schema
# ------------------------

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

parser = PydanticOutputParser(pydantic_object=Review)

# ------------------------
# Classification Prompt
# ------------------------

template1 = PromptTemplate(
    template="""
Classify the sentiment of the following text.

Text:
{text}

Respond ONLY with valid JSON.

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

classifier_chain = template1 | chat_model | parser

text1 = """
The $699 price tag is incredible for what you get.
The AMOLED display is buttery smooth,
and the battery easily lasts two days.
"""

result = classifier_chain.invoke({"text": text1})
print(result)

# ------------------------
# Response Prompts
# ------------------------

template2 = PromptTemplate(
    template="""
Write an appropriate one line response to this positive review:

{review}
""",
    input_variables=["review"]
)

template3 = PromptTemplate(
    template="""
Write an appropriate one line response to this negative review:

{review}
""",
    input_variables=["review"]
)

text_parser = StrOutputParser()

positive_chain = template2 | chat_model | text_parser
negative_chain = template3 | chat_model | text_parser

# ------------------------
# Branch Logic
# ------------------------

branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        RunnableLambda(lambda x: {"review": x}) | positive_chain
    ),
    (
        lambda x: x.sentiment == "negative",
        RunnableLambda(lambda x: {"review": x}) | negative_chain
    ),
    RunnableLambda(
        lambda x: "No response needed for neutral reviews"
    )
)

# ------------------------
# Combine Chains
# ------------------------

final_chain = classifier_chain | branch_chain

# ------------------------
# Test
# ------------------------

text1 = """
The $699 price tag is incredible for what you get.
The AMOLED display is buttery smooth,
and the battery easily lasts two days.
"""

result = final_chain.invoke({"text": text1})

print(result)

final_chain.get_graph().print_ascii()