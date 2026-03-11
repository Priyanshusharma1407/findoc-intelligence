import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)

# ── Context Builder ────────────────────────────────────────

def build_context(docs: list[Document]) -> str:
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        element_type = doc.metadata.get("element_type", "text")
        original = doc.metadata.get("original", doc.page_content)
        context_parts.append(
            f"[Source {i+1} | {source} | {element_type}]\n{original}"
        )
    return "\n\n---\n\n".join(context_parts)

# ── Citation Builder ───────────────────────────────────────

def build_citations(docs: list[Document]) -> str:
    citations = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        citations.append(f"[{i+1}] {source} — chunk {chunk_index}")
    return "\n".join(citations)

# ── Prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FinDoc Intelligence, an expert financial document analyst.

You answer questions strictly based on the provided document context.
If the answer is not in the context, say "I could not find this information in the uploaded documents."

Always be precise, cite source numbers like [1], [2] when referencing specific content.
For financial figures, preserve exact numbers from the source.

Context:
{context}"""

# ── Generation ─────────────────────────────────────────────

def generate_answer(
    query: str,
    docs: list[Document],
    chat_history: list = [],
) -> dict:

    context = build_context(docs)
    citations = build_citations(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "query": query,
    })

    return {
        "answer": response.content,
        "citations": citations,
        "source_docs": docs,
    }

# ── History Manager ────────────────────────────────────────

def update_history(chat_history: list, query: str, answer: str) -> list:
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))
    return chat_history


if __name__ == "__main__":
    from ingestion import ingest_document
    from retrieval import build_bm25_index, retrieve, upload_documents, client, COLLECTION_NAME

    # ── Load or skip ingestion ──
    count = client.count(collection_name=COLLECTION_NAME).count if COLLECTION_NAME in [c.name for c in client.get_collections().collections] else 0

    if count == 0:
        docs = ingest_document("Financial-Reporting-FR-FAQ-Revised-Final.pdf")
        upload_documents(docs)
    else:
        from qdrant_client.models import ScrollRequest
        results = client.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)
        from langchain_core.documents import Document
        docs = [Document(
            page_content=r.payload["page_content"],
            metadata={
                "source": r.payload["source"],
                "element_type": r.payload["element_type"],
                "original": r.payload["original"],
                "chunk_index": r.payload["chunk_index"],
            }
        ) for r in results[0]]

    bm25, all_docs = build_bm25_index(docs)
    chat_history = []

    # ── Multi-turn test ──
    questions = [
        "What is GAAP and why is it important?",
        "Can you elaborate on the first point?",
        "What are the types of accounting?",
    ]

    for question in questions:
        print(f"\n── Question: {question}")
        retrieved = retrieve(question, bm25, all_docs)
        result = generate_answer(question, retrieved, chat_history)
        chat_history = update_history(chat_history, question, result["answer"])
        print(f"\n── Answer:\n{result['answer']}")
        print(f"\n── Citations:\n{result['citations']}")
        print("\n" + "="*60)