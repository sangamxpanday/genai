from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import TextLoader, WebBaseLoader
import os

api_token = os.getenv("HUGGINGFACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

url = "https://www.daraz.com.np/products/flying-sensor-helicopter-toy-with-remote-control-infrared-gravity-sensor-for-kids-i127867300-s1034850548.html?pvid=44108423-bdfd-4171-bef1-713721765d27&search=jfy&scm=1007.51705.413671.0&spm=a2a0e.tm80335409.just4u.d_127867300"

loader = WebBaseLoader(url)

docs = loader.load()

content = docs[0].page_content

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Answer {question} on : {topic}",
    input_variables=["topic", "question"]
)

chain = prompt | model | parser

result = chain.invoke({'topic': content, 'question': "What is the rating?"})

print(result)