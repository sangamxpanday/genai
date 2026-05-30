import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent="my-rag-app/1.0",
    language="en",
    top_k_results=3
)

page = wiki.page("Manchester City F.C.")

print(page.title)
print(page.summary[:500])