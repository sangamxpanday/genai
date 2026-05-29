from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

doc1 = Document(
    page_content="Lionel Messi is a talented right winger for barcelona.",
    metadata={"source": "doc1"},
)
doc2 = Document(
    page_content="Xavi Hernandez is a midfielder for barcelona.",
    metadata={"source": "doc2"},
)
doc3 = Document(
    page_content="Cristiano Ronaldo is a talented left winger for real madrid.",
    metadata={"source": "doc3"},
)
doc4 = Document(
    page_content="Sergio Ramos is a talented defender for real madrid.",
    metadata={"source": "doc4"},
)

vectorstore = Chroma.from_documents(
    embedding=embeddings,
    documents=[doc1],
    persist_directory="chroma_db_final_final",
    collection_name="my_collection",
)


print("Adding documents to the vectorstore...")
#Add documents to the vectorstore
vectorstore.add_documents([doc2, doc3, doc4])

print("Documents on the vectorstore are:")
#View Documents
result = vectorstore.get(include=["documents", "metadatas"])
for doc, metadata in zip(result['documents'], result['metadatas']):
    print(f"Document: {doc}, Metadata: {metadata}")

print("Performing similarity search... (who is the right winger?)")
result = vectorstore.similarity_search(query = "Who is a right winger?", k=2)
for doc in result:
    print(f"Document: {doc.page_content}, Metadata: {doc.metadata}")

print("Performing similarity search with score... (who is a midfielder?)")
result_with_score = vectorstore.similarity_search_with_score(query = "Who is a midfielder?", k=2)
for i in range(len(result_with_score)):
    doc ,score = result_with_score[i]
    print(f"Document: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

print("Performing similarity search with score and filter... (who is the right winger in doc1?)")
result_filtered = vectorstore.similarity_search_with_score(query = "Who is a right winger?", filter = {"source": "doc1"}, k=2)
for i in range(len(result_filtered)):
    doc, score = result_filtered[i]
    print(f"Document: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

updated_doc1 = Document(
    page_content="Lionel Messi is a talented right winger for barcelona and argentina.",
    metadata={"source": "doc1"},
)

# Get the actual Chroma-generated IDs
print("Getting document IDs...")
ids_result = vectorstore.get()
print(f"Available IDs: {ids_result['ids']}")


print("Updating document in the vectorstore...")
vectorstore.update_documents(documents=[updated_doc1], ids = ["36bd5cc5-240c-4f8f-ac48-61fbb9520465"])


#printing all documents after update
print("Documents on the vectorstore after update are:")
result = vectorstore.get(include=["documents", "metadatas"])
for doc, metadata in zip(result['documents'], result['metadatas']):
    print(f"Document: {doc}, Metadata: {metadata}")