from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

texts = """
Lionel Messi is a talented football player.
Xavi Hernandez is also a talented football player.
Roses are red.
Violets are blue.
"""

# LangChain-compatible embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

docs = splitter.create_documents([texts])

print(len(docs))
print(docs)