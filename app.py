import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ingestion import ingest_document
from retrieval import build_bm25_index, retrieve, upload_documents, client, COLLECTION_NAME
from generation import generate_answer, update_history

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)

# ── Page Config ────────────────────────────────────────────

st.set_page_config(
    page_title="FinDoc Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FinDoc Intelligence")
st.caption("Upload financial documents and ask questions about them.")

# ── Session State ──────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "docs" not in st.session_state:
    st.session_state.docs = []

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

# ── Query Rewriter ─────────────────────────────────────────

def rewrite_query(query: str, chat_history: list) -> str:
    if not chat_history:
        return query

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and a follow-up question, rewrite the follow-up question 
to be a standalone question that can be understood without the chat history.
Only return the rewritten question, nothing else."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Follow-up question: {query}\n\nRewritten standalone question:"),
    ])

    chain = rewrite_prompt | llm
    result = chain.invoke({"chat_history": chat_history, "query": query})
    return result.content.strip()

# ── Sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and not st.session_state.uploaded:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Ingest
            docs = ingest_document(temp_path)

            # Upload to Qdrant
            upload_documents(docs)

            # Build BM25
            st.session_state.bm25, st.session_state.docs = build_bm25_index(docs)
            st.session_state.uploaded = True

            # Cleanup temp file
            os.remove(temp_path)

        st.success(f"✅ Processed {len(docs)} chunks from {uploaded_file.name}")

    if st.session_state.uploaded:
        st.info(f"📄 {len(st.session_state.docs)} chunks loaded")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain · Groq · Qdrant · HuggingFace")

# ── Chat Interface ─────────────────────────────────────────

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
if query := st.chat_input("Ask a question about your document..."):

    if not st.session_state.uploaded:
        st.warning("Please upload a document first.")
    else:
        # Show user message
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Step 1 — Rewrite query if follow-up
                standalone_query = rewrite_query(query, st.session_state.chat_history)

                # Step 2 — Retrieve
                retrieved_docs = retrieve(
                    standalone_query,
                    st.session_state.bm25,
                    st.session_state.docs
                )

                # Step 3 — Generate
                result = generate_answer(
                    query,
                    retrieved_docs,
                    st.session_state.chat_history
                )

                # Step 4 — Display answer
                st.write(result["answer"])

                # Step 5 — Show citations
                with st.expander("📚 Sources"):
                    st.text(result["citations"])

                # Step 6 — Update history
                st.session_state.chat_history = update_history(
                    st.session_state.chat_history,
                    query,
                    result["answer"]
                )