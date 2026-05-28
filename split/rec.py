from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path = "",
    glob = "*.txt",
    loader_cls = TextLoader
)

txt_docs = loader.lazy_load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap = 0,
    # separators = ["\n\n", "\n", " ", ""]
)

split_docs = splitter.split_documents(txt_docs)

print(len(split_docs))

print(split_docs)