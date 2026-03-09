# ── Imports ───────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import time

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")

# ── Unstructured client ────────────────────────────────────────────────────────
client = UnstructuredClient(api_key_auth=UNSTRUCTURED_API_KEY)


# ── Step 1: Extract PDF elements via Unstructured API ─────────────────────────
def extract_pdf_elements(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    response = client.general.partition(
        request={
            "partition_parameters": {
                "files": {
                    "content": data,
                    "file_name": os.path.basename(file_path),
                },
                "strategy": "hi_res",
                "infer_table_structure": True,
                "extract_image_block_types": ["Image"],
                "chunking_strategy": "by_title",
                "max_characters": 4000,
                "new_after_n_chars": 3800,
                "combine_text_under_n_chars": 2000,
            }
        }
    )
    return response.elements

# ── Step 2: Categorize elements ──────────────────────────────
def categorize_elements(raw_elements):
    texts  = []
    tables = []

    for element in raw_elements:
        el_type = element.get("type", "")
        text    = element.get("text", "")

        if el_type == "Table":
            tables.append(text)
        elif el_type == "CompositeElement":
            texts.append(text)

    return texts, tables


# ── Step 3: Summarize with Groq  ─────────────


def summarize_elements(texts, tables):
    prompt_text = """You are an assistant summarizing financial document content for retrieval.
    Give a concise summary optimized for retrieval. Content: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    chain = prompt | model

    text_summaries = []
    for t in texts:
        result = chain.invoke({"element": t})
        text_summaries.append(result.content)
        time.sleep(2)

    table_summaries = []
    for t in tables:
        result = chain.invoke({"element": t})
        table_summaries.append(result.content)
        time.sleep(2)

    return text_summaries, table_summaries

# ── Step 4: Convert to LangChain Documents with metadata ──────────────────────
def convert_to_documents(texts, tables, text_summaries, table_summaries, source_name):
    docs = []

    for i, (text, summary) in enumerate(zip(texts, text_summaries)):
        docs.append(Document(
            page_content=summary,
            metadata={
                "source":       source_name,
                "element_type": "text",
                "original":     text,
                "chunk_index":  i,
            }
        ))

    for i, (table, summary) in enumerate(zip(tables, table_summaries)):
        docs.append(Document(
            page_content=summary,
            metadata={
                "source":       source_name,
                "element_type": "table",
                "original":     table,
                "chunk_index":  len(texts) + i,
            }
        ))

    return docs


# ── Main ingestion function ────────────────────────────────────────────────────
def ingest_document(file_path: str) -> list[Document]:
    source_name = os.path.basename(file_path)
    print(f"\n[Ingestion] Processing: {source_name}")

    # Step 1 — Extract
    print("[Ingestion] Calling Unstructured API...")
    raw_elements = extract_pdf_elements(file_path)
    print(f"[Ingestion] Raw elements: {len(raw_elements)}")

    # Step 2 — Categorize
    texts, tables = categorize_elements(raw_elements)
    print(f"[Ingestion] Texts: {len(texts)} | Tables: {len(tables)}")

    # Step 3 — Summarize
    print("[Ingestion] Summarizing with Groq...")
    text_summaries, table_summaries = summarize_elements(texts, tables)

    # Step 4 — Convert
    all_docs = convert_to_documents(texts, tables, text_summaries, table_summaries, source_name)
    print(f"[Ingestion] Total documents ready: {len(all_docs)}\n")

    return all_docs


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path_to_pdf>")
        sys.exit(1)

    docs = ingest_document(sys.argv[1])

    print(f"Total: {len(docs)}")
    if docs:
        print(f"\nSample:\n  Summary  : {docs[0].page_content[:200]}")
        print(f"  Original : {docs[0].metadata['original'][:200]}")
        print(f"  Metadata : {docs[0].metadata}")
    else:
        print("No documents extracted — check your API key or PDF file.")