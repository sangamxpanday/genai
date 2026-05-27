from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
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

prompt1 = PromptTemplate(
    template = "Write a detailed report on the following topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Write a short summary on the following topic: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt1 ,model ,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2 ,model ,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain)

result = final_chain.invoke({'topic': "Russia vs ukraine war"})

print(result)

final_chain.get_graph().print_ascii()
