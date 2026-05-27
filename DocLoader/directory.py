from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
import os

api_token = os.getenv("HUGGINGFACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=100,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

loader1 = DirectoryLoader(
    path = "",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

pdf_docs = loader1.load()

#lazy loading
pdf_docs_lazy = loader1.lazy_load()

loader2 = DirectoryLoader(
    path = "",
    glob = "**/*.txt",
    loader_cls = TextLoader
)

txt_docs = loader2.load()

#lazy loading

txt_docs_lazy = loader2.lazy_load()

final = pdf_docs + txt_docs

print(len(pdf_docs_lazy))
print(len(txt_docs_lazy))
print(len(final))

print(final[0].page_content)
print("-"*100)
print(pdf_docs_lazy[0].page_content)