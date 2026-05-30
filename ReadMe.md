# RAG
## Component of RAG:
1. Document Loader
2. Text Splitter
3. Vector Store
4. Retrievers

## What is RAG
* Query(Prompt) -> LLM -> Response
* LLM are giant Transformer based Neural Network  Architecture
 
* Where does llm store all knowledge? ->Parameter's
In form of numbers. 
* Knowledge are called parametric knowledge.

* How to access these knowledge? ->Prompting
We send prompt to LLM.
* It searches through it's parametric knowledge and returns answer/knowledge.

* We cannot generate best knowledge based on only it's parametric knowledge
- eg:Personal information. (What is my teacher's name?)
* ALso known as Hallucination(Giving factual answer know on confidence)

* Problem:
- Personal information
- Recent data
- Hallucination

### Fine Tuning
* How can we solve this problem?
- Fine tuning.
- Train a pretrain model on smaller domain specific dataset

* Different ways of fine tuning
1. Supervised fine tuning(labelled)
2. Continue pre tranning(Unsupervised)
3. RLHF
4. LORA

* Problem with fine tuning
- Traning big model is Computationally expensive
- Technical Expertise
- Changing data 

* So we dont use fine tuning

### In Context Learning 
* Another technique for solving these problems are:
In context learning
- LLM solve a task by looking at examples.
- **Few short prompting**
- Small model didn't have in context learning
- Larger's model are capable
- Research Paper: **Language Models are Few-Shot Learners**

### RAG
- What if instead of few shot prompting we sent the entire context
- This is called RAG
- (Query + Context) -> (Prompt) -> (LLM) -> (Response)

## Understanding RAG

- Information Retrieval + Text Generation

### **Steps of RAG**:
- Indexing
- Retrieval
- Augmentation 
- Generation

### Indexing
- Context from external knowledge base
- Creating external knowlege base
* ***Steps:***
1. Document Ingestion(Document Loader)
2. Text Chunking (Text Splitter)
3. Embedding Generation
4. Store in vector store

- Vector store = External knowledge base

### Retrieval
- Similar to query from external knowledge base
- finding the most relevent data
- Generates embedding vector of our query
- Searches from the similar vector to our query from vector store and gives each vector rank
- context = vector with highest rank 

### Augmentation
- Prompt creation from our query and retrieved text
- Relevent Chunks = Context
- Prompt Template

### Generation
- Text generated from LLM's
- Gives prompt to LLM