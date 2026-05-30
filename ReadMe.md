# RAG (Retrieval-Augmented Generation)

## Components of RAG
1. Document Loader
2. Text Splitter
3. Vector Store
4. Retriever

---

## What is RAG?

### Traditional LLM Workflow

`Query (Prompt) → LLM → Response`

- LLMs are giant transformer-based neural network architectures.
- LLMs store knowledge in their **parameters**.
- This stored knowledge is called **Parametric Knowledge**.

### How does an LLM access its knowledge?
- Through **prompting**.
- We send a prompt to the LLM.
- The model searches through its parametric knowledge and generates a response.

### Limitations of Parametric Knowledge
LLMs cannot always provide the best answer using only their stored knowledge.

**Examples:**
- Personal information (e.g., "What is my teacher's name?")
- Recent or real-time information

This can lead to **hallucinations**, where the model generates incorrect or fabricated information with high confidence.

### Problems
- Personal information is unavailable.
- Recent data may not be included.
- Hallucinations can occur.

---

## Fine-Tuning

### How can we solve these problems?
One approach is **Fine-Tuning**.

Fine-tuning means training a pre-trained model on a smaller, domain-specific dataset.

### Types of Fine-Tuning
1. Supervised Fine-Tuning (SFT)
2. Continued Pretraining
3. RLHF (Reinforcement Learning from Human Feedback)
4. LoRA (Low-Rank Adaptation)

### Problems with Fine-Tuning
- Training large models is computationally expensive.
- Requires technical expertise.
- Difficult to keep up with frequently changing data.

Because of these limitations, fine-tuning is not always the preferred solution.

---

## In-Context Learning

Another technique for solving these problems is **In-Context Learning**.

### What is In-Context Learning?
- The LLM learns how to solve a task by looking at examples provided in the prompt.
- This is commonly known as **Few-Shot Prompting**.
- Smaller models usually have limited in-context learning capabilities.
- Larger models perform much better at it.

### Research Paper
**Language Models are Few-Shot Learners**

---

## RAG (Retrieval-Augmented Generation)

### Core Idea
Instead of sending only a few examples, we provide the model with relevant external context.

`(Query + Context) → Prompt → LLM → Response`

RAG combines:
- **Information Retrieval**
- **Text Generation**

---

# Understanding RAG

## Main Stages of RAG
1. Indexing
2. Retrieval
3. Augmentation
4. Generation

---

## 1. Indexing

The goal is to create an external knowledge base from your documents.

### Steps
1. Document Ingestion (Document Loader)
2. Text Chunking (Text Splitter)
3. Embedding Generation
4. Store Embeddings in a Vector Store

### Note
A **Vector Store** acts as the external knowledge base.

---

## 2. Retrieval

The goal is to find the most relevant information from the vector store.

### Process
1. Convert the user's query into an embedding vector.
2. Compare it with vectors stored in the vector database.
3. Rank the stored vectors based on similarity.
4. Retrieve the most relevant chunks as context.

`Query → Embedding → Similarity Search → Relevant Chunks`

The retrieved chunks become the **Context**.

---

## 3. Augmentation

The retrieved context is combined with the user's query to create a better prompt.

### Components
- User Query
- Retrieved Chunks (Context)
- Prompt Template

### Output

`Prompt = Query + Retrieved Context`

---

## 4. Generation

The final prompt is sent to the LLM.

### Process

`Prompt → LLM → Response`

The LLM generates a response using both:
- Its Parametric Knowledge
- The Retrieved Context
