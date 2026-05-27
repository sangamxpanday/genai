from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path = "",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

pdf_docs = loader.lazy_load()

splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap = 0,
    separator = ''
)

print("Splitting documents...")

result = splitter.split_documents(pdf_docs)
print(result[0].page_content)
