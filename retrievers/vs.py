from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

documents = [
    Document(
        page_content="Lionel Messi is a talented right winger for barcelona."),
    Document(
        page_content="Xavi Hernandez is a midfielder for barcelona."),
    Document(
        page_content="Cristiano Ronaldo is a talented left winger for real madrid."),
    Document(
        page_content="Sergio Ramos is a talented defender for real madrid.")
]

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma.from_documents(
    embedding=embeddings,
    documents=documents,
    persist_directory="chroma_db_for_retriever",
    collection_name="Retriever_Collection",
)

retriver = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "Who is a right winger?"

result = retriver.invoke(query)

for i, doc in enumerate(result):
    print(f"Document {i+1}:")
    print(f"  Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}\n")

