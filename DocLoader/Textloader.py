from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import TextLoader
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

loader = TextLoader("city.txt", encoding="utf-8")

docs = loader.load()

prompt = PromptTemplate(
    template = "Write a summary on the following topic: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': docs[0].page_content})
print(result)