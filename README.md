# Building a Retrieval-Augmented Generation (RAG) Application with OpenAI and Pinecone

## Overview

This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** application using **OpenAI** for embeddings and language models, and **Pinecone** as the vector database. The project is built using the **LangChain** framework, which provides tools for integrating large language models (LLMs) with retrieval systems to create powerful question-answering (Q&A) applications.

The RAG application is designed to answer questions based on a specific data source by retrieving relevant information and generating accurate responses using OpenAI's GPT model. The project is divided into two main components:

1. **Indexing**: A pipeline for ingesting and indexing data from a source.
2. **Retrieval and Generation**: The RAG chain that retrieves relevant data and generates responses based on user queries.

## Project Architecture

The RAG application consists of the following components:

### 1. **Indexing Pipeline**

- **Document Loader**: Loads data from a source (e.g., a website or text file).
- **Text Splitter**: Splits large documents into smaller chunks for efficient processing.
- **Vector Store**: Stores the document chunks as embeddings in a vector database (Pinecone).
- **Embeddings Model**: Converts text into embeddings using OpenAI's `text-embedding-3-large` model.

### 2. **Retrieval and Generation**

- **Retriever**: Retrieves relevant document chunks from the vector store based on the user's query.
- **Language Model**: Generates a response using OpenAI's GPT model, incorporating the retrieved context.
- **Prompt Engineering**: A predefined prompt template is used to guide the model in generating concise and accurate answers.

### 3. **LangGraph Integration**

- **State Management**: Manages the state of the application, including the user's query, retrieved context, and generated answer.
- **Graph Execution**: Orchestrates the retrieval and generation steps using LangGraph, enabling multi-step retrieval and conversational interactions.

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nat15005/Taller-08-AREP-LLM-Parte-2.git
cd Taller-08-AREP-LLM-Parte-2
```

### Step 2: Install Dependencies

Install the required Python packages using `pip`:

```
pip install langchain langchain-openai langchain-pinecone pinecone-client langgraph beautifulsoup4
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

plaintext

Copy

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### Step 4: Initialize Pinecone

Run the following script to initialize the Pinecone index:

```python
import os
import pinecone
from pinecone import ServerlessSpec

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
```

---

## Running the Application

### Step 1: Load and Index Data

Run the following script to load data from a website, split it into chunks, and index it in Pinecone:

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load data
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore(index_name="langchain-test-index", embedding=embeddings)

# Index documents
vector_store.add_documents(documents=all_splits)
```

### Step 2: Run the RAG Application

Use the following script to query the indexed data and generate responses:

```python
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START

# Define the RAG prompt
prompt = hub.pull("rlm/rag-prompt")

# Define the state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define the retrieval and generation steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build and run the graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Query the application
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
```

---

## Examples

### Example 1: Simple Query

**Query**: "What is Task Decomposition?"

**Response**:

```
Task Decomposition is the process of breaking down a complicated task into smaller, manageable steps. This technique, often enhanced by methods like Chain of Thought (CoT) or Tree of Thoughts, allows models to reason through tasks systematically and improves performance by clarifying the thought process.
```

### Example 2: Conversational Query

**Query**: "What are common ways of doing it?"

**Response**:

```
Common ways of performing Task Decomposition include: (1) using Large Language Models (LLMs) with simple prompts like "Steps for XYZ" or "What are the subgoals for achieving XYZ?", (2) employing task-specific instructions such as "Write a story outline" for specific tasks, and (3) incorporating human inputs to guide the decomposition process.
```

---

## Screenshots

![image](https://github.com/user-attachments/assets/5c46cfe0-6ec8-4367-b1e0-e7e5793c0e70)

![image](https://github.com/user-attachments/assets/25f234bb-a64e-4b1e-b603-3d42dd0e0aee)

![image](https://github.com/user-attachments/assets/903c59a9-a96d-4e23-8925-d36827a60dbc)

![image](https://github.com/user-attachments/assets/610b256d-db72-44a3-a3f8-4817655e1d74)

![image](https://github.com/user-attachments/assets/37725989-e20b-4865-9295-2a23de1e16d3)

![image](https://github.com/user-attachments/assets/9c17cb64-3b4f-4850-9c00-9af236d66349)

![image](https://github.com/user-attachments/assets/90518697-efba-4a17-a075-53a67ee7487f)

![image](https://github.com/user-attachments/assets/a9de9c43-1a7a-4d9b-95a1-b828dfe978a4)

![image](https://github.com/user-attachments/assets/f4216d91-2e9c-4f96-b589-e7158c6eb708)

## Dependencies

- **LangChain**: Framework for building applications with LLMs.
- **OpenAI**: Provides embeddings and language models.
- **Pinecone**: Vector database for storing and retrieving document embeddings.
- **LangGraph**: Orchestration framework for multi-step retrieval and generation.

## Author

Developed by [Natalia Rojas](https://github.com/Nat15005)
