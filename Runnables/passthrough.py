from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

parser = StrOutputParser()

template1 = PromptTemplate(
    template = "Write a joke about the following topic: {topic}",
    input_variables=["topic"]
)

joke_gen_chain = RunnableSequence(template1 ,model ,parser)

template2 = PromptTemplate(
    template = "Explain the joke: {joke}",
    input_variables=["joke"]
)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(template2, model, parser)
})

combined_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = combined_chain.invoke({"topic": "programming"})

print(result)