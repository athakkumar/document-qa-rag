"""
RAG Pipeline — LangChain 1.x + Gemini + ChromaDB
Document Q&A system using Retrieval-Augmented Generation (RAG).
Uses LCEL (LangChain Expression Language) — the modern approach.
"""

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Config 
CHROMA_DIR       = "./chroma_store"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
GEMINI_MODEL     = "gemini-2.5-flash"
COLLECTION_NAME  = "documents"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# Ingest PDF
def ingest_pdf(pdf_path: str) -> dict:
    # Extract text
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    page_count = len(doc)
    doc.close()

    if not full_text.strip():
        raise ValueError("PDF appears empty or scanned (no extractable text).")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_text(full_text)

    # Clear existing collection and re-embed (avoid duplicates on re-ingest)
    Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    ).delete_collection()

    # Embed and store
    Chroma.from_texts(
        texts=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    return {
        "status": "success",
        "pages_processed": page_count,
        "chunks_created": len(chunks),
        "vector_store": CHROMA_DIR,
    }


# Answer questions (LCEL pipeline)
def answer_question(question: str, api_key: str, top_k: int = 4) -> dict:
    # Load vector store
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    if vectorstore._collection.count() == 0:
        raise ValueError("Vector store is empty. Please ingest a PDF first via POST /ingest.")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # Prompt
    prompt = PromptTemplate.from_template("""You are a helpful assistant that answers questions based strictly on the provided document context.

Use ONLY the context provided below to answer the question.
If the answer is not found in the context, say "This information is not available in the uploaded document."
Be concise, accurate, and technical where appropriate.

Context:
{context}

Question: {question}

Answer:""")

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "sources_used": [doc.page_content[:200] + "..." for doc in source_docs],
        "chunks_retrieved": len(source_docs),
    }
