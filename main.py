from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# -------------------------------------------------
# 1. Load document
# -------------------------------------------------
def load_document(path):
    loader = TextLoader(path)
    return loader.load()


# -------------------------------------------------
# 2. Create text chunks
# -------------------------------------------------
def create_chunks(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(data)


# -------------------------------------------------
# 3. Embeddings + Vector DB
# -------------------------------------------------
def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_store"
    )

    vectordb.persist()
    return vectordb


# -------------------------------------------------
# 4. QA Chain with STRICT grounded prompt
# -------------------------------------------------
def build_qa_chain(vectordb):

    grounded_prompt = """
You must answer ONLY using the text in the context.

STRICT RULES:
- If the answer is NOT in the context, reply EXACTLY:
  "The document does not contain this information."
- Do NOT add examples.
- Do NOT add stories.
- Do NOT add explanations.
- Do NOT add opinions.
- Do NOT add anything outside the context.
- Keep answer VERY short.
- No extra sentences.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=grounded_prompt,
        input_variables=["context", "question"]
    )

    # Retrieve more chunks to reduce hallucinations
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Strong restrictions for phi model to stop imagination
    llm = Ollama(
        model="phi",
        temperature=0,
        num_predict=60,
        top_k=20,
        top_p=0.1
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa


# -------------------------------------------------
# 5. Main program
# -------------------------------------------------
def main():
    print("📘 Loading document...")
    data = load_document("speech.txt")

    print("🔍 Splitting into chunks...")
    chunks = create_chunks(data)

    print("🧠 Generating embeddings & creating vector store...")
    vectordb = embed_and_store(chunks)

    print("🤖 Building grounded QA system...")
    qa = build_qa_chain(vectordb)

    print("\n❓ Ask a question (type 'exit' to quit)")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        print("🧠 Thinking...\n")
        response = qa.invoke({"query": query})
        print("💬 Answer:", response["result"])


# -------------------------------------------------
# Run the Program
# -------------------------------------------------
if __name__ == "__main__":
    main()
