from langchain_community.retrievers import WikipediaRetriever
import time

def fetch_wikipedia(query, max_retries=3):
    retriever = WikipediaRetriever(top_k_results=3, lang="en")
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Fetching '{query}'")
            docs = retriever.invoke(query)
            print(f"✓ Got {len(docs)} documents\n")
            
            for i, doc in enumerate(docs):
                print(f"Document {i+1}:")
                print(f"  Title: {doc.metadata.get('title', 'N/A')}")
                print(f"  Source: {doc.metadata.get('source', 'N/A')}")
                print(f"  Content: {doc.page_content[:150]}...\n")
            return True
            
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {str(e)[:80]}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt 
                print(f"  Waiting {wait_time} seconds before retry...\n")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts.")
                return False

query = "Manchester city"
fetch_wikipedia(query)

