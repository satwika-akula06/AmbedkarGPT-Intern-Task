# AmbedkarGPT – Retrieval-Augmented Question Answering System

This project implements a strict Retrieval-Augmented Generation (RAG) system that answers user questions using only the content present in `speech.txt`. The pipeline ensures grounded, factual, non-hallucinated responses by restricting the model to retrieved context only.

---

## 📌 Overview

AmbedkarGPT:
- Loads a speech document
- Splits it into semantic chunks
- Converts chunks into embeddings
- Stores them inside a Chroma vector database
- Retrieves the most relevant chunks when you ask a question
- Uses the Phi model (via Ollama) to generate **strict, context-only answers**
- Replies exactly with  
  **"The document does not contain this information."**  
  when the answer is not found in the text

This ensures reliability, accuracy, and eliminates hallucinations.

---

## 📁 Project Structure

```
AmbedkarGPT/

│-- main.py
│-- speech.txt
│-- requirements.txt
│-- chroma_store/   # auto-generated on first run
```

---

## ⚙️ Installation Guide

### 1️⃣ Create a virtual environment
```
python -m venv venv
```

### 2️⃣ Activate the environment
```
venv\Scripts\activate
```

### 3️⃣ Install all dependencies
```
pip install -r requirements.txt
```

---

## 🤖 Setting Up Ollama

### Step 1 — Install Ollama
Download from: https://ollama.com/download

### Step 2 — Pull the model  
I am using the Phi model because my system does not support running Mistral.  
To install it, I used:
```
ollama pull phi
```

### Step 3 — Start the Ollama server  
(If not already running)
```
ollama serve
```
---

## ▶️ Running the RAG System

Run the program:
```
python main.py
```
You will see:

❓ Ask a question (type 'exit' to quit)

Examples:

You: What is the central message of the speech?

Answer: The real enemy is the belief in the shastras.

If the information is missing:
```
The document does not contain this information.
```

---

## 🧠 RAG Workflow

1. **Load Text**  
   Reads `speech.txt` using LangChain’s `TextLoader`.

2. **Split Text Into Chunks**  
   500-token chunks with 100 overlap to preserve meaning.

3. **Embedding Creation**  
   Uses MiniLM from Sentence Transformers.

4. **Vector Store**  
   Embeddings stored in ChromaDB.

5. **Retriever**  
   Fetches top-k relevant chunks.

6. **Strict Prompting**  
   Ensures:
   - No assumptions  
   - No examples  
   - No extended interpretation  
   - No reasoning beyond context  

7. **Grounded Answer Generation**  
   Phi model produces concise answers only from retrieved text.

---

## 📦 Requirements (requirements.txt)

- langchain==0.1.12
- langchain-community==0.0.24
- langchain-core==0.1.53
- langchain-text-splitters==0.0.2
- chromadb==0.4.22
- sentence-transformers==5.1.2
- huggingface-hub==0.36.0
- ollama==0.1.8
- pydantic==2.12.4
- pydantic_core==2.41.5
- numpy==1.26.4
- requests==2.32.5

---

## 📌 Notes

- You must keep the **Ollama server running** for the program to work.
- `chroma_store/` is auto-created and can be reused without regeneration.
- If you want a fresh vector DB, simply delete `chroma_store/`.

---

## 🌟 Acknowledgements

This project makes use of:
- **LangChain** (RAG framework)
- **ChromaDB** (vector storage)
- **HuggingFace** (embeddings)
- **Ollama + Phi** (local LLM inference)

---

## 📫 Contact
For any questions or suggestions, please open an issue in this repository.


