import os
import tempfile
from pathlib import Path
from typing import Any, List

import streamlit as st
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# ✅ Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jarvis")

# Key checks
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY in .env")
    st.stop()
if not PINECONE_API_KEY:
    st.error("❌ Missing PINECONE_API_KEY in .env")
    st.stop()

if "local_vectorstore" not in st.session_state:
    st.session_state["local_vectorstore"] = None
    st.session_state["local_doc_count"] = 0
# Initialize Pinecone client and embedding stack
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")


class PineconeRetriever(BaseRetriever):
    """Bridges the Pinecone SDK and LangChain retriever interface."""

    index: Any
    embeddings: OpenAIEmbeddings
    top_k: int = 3
    text_key: str = "text"

    model_config = {"arbitrary_types_allowed": True}

    def _matches_to_documents(self, matches: Any) -> List[Document]:
        documents: List[Document] = []
        for match in matches or []:
            if isinstance(match, dict):
                metadata = dict(match.get("metadata", {}) or {})
                text = metadata.get(self.text_key) or match.get("text", "")
                match_id = match.get("id")
                score = match.get("score")
            else:
                metadata = dict(getattr(match, "metadata", {}) or {})
                text = metadata.get(self.text_key) or getattr(match, "text", "")
                match_id = getattr(match, "id", None)
                score = getattr(match, "score", None)

            if match_id and "id" not in metadata:
                metadata["id"] = match_id
            if score is not None:
                metadata["score"] = score

            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        response = self.index.query(vector=vector, top_k=self.top_k, include_metadata=True)
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])
        return self._matches_to_documents(matches)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def build_local_vectorstore(files: List[Any]):
    """Embed uploaded documents into an in-memory vectorstore."""

    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis-upload-"))
    documents = []

    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        temp_path = tmp_dir / uploaded.name
        with open(temp_path, "wb") as handle:
            handle.write(uploaded.getbuffer())

        if suffix == ".pdf":
            loader = PyPDFLoader(str(temp_path))
        else:
            loader = TextLoader(str(temp_path), encoding="utf-8")

        documents.extend(loader.load())

    if not documents:
        raise ValueError("No readable pages found in uploaded files.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


connection_message = None
total_vectors = 0
try:
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0) if isinstance(stats, dict) else getattr(stats, "total_vector_count", 0)
    connection_message = f"Connected to index '{INDEX_NAME}' with {total_vectors} vectors."
except Exception as e:
    st.error(f"❌ Pinecone index check failed: {e}. Run ingestion script first? (Check index name matches.)")
    st.stop()

retriever = PineconeRetriever(index=index, embeddings=embeddings, top_k=3)
import langchain
langchain_version = langchain.__version__

st.sidebar.title("Workspace")
st.sidebar.caption("Upload documents to chat about them instantly.")
uploaded_files = st.sidebar.file_uploader(
    "Drop PDF or TXT files",
    accept_multiple_files=True,
    type=["pdf", "txt"],
    help="Files stay local for this session only."
)

if st.sidebar.button("Build ad-hoc index", use_container_width=True, disabled=not uploaded_files):
    with st.spinner("Embedding uploaded docs..."):
        try:
            local_vs, chunk_count = build_local_vectorstore(uploaded_files)
            st.session_state["local_vectorstore"] = local_vs
            st.session_state["local_doc_count"] = chunk_count
            st.sidebar.success(f"Ready! {chunk_count} chunks indexed for this session.")
        except Exception as exc:
            st.sidebar.error(f"Upload failed: {exc}")

if st.session_state.get("local_vectorstore"):
    st.sidebar.info(
        f"Using uploaded docs ({st.session_state.get('local_doc_count', 0)} chunks).",
        icon="📁",
    )
    if st.sidebar.button("Clear uploaded docs", use_container_width=True):
        st.session_state["local_vectorstore"] = None
        st.session_state["local_doc_count"] = 0
        st.sidebar.success("Cleared local session index.")

active_retriever = retriever
data_source_label = f"Pinecone index '{INDEX_NAME}'"

if st.session_state.get("local_vectorstore"):
    active_retriever = st.session_state["local_vectorstore"].as_retriever(search_kwargs={"k": 3})
    data_source_label = f"Uploaded docs ({st.session_state.get('local_doc_count', 0)} chunks)"

diagnostics: List[str] = []
chain_error: str | None = None
qa = None
chain_type = None

try:
    from langchain.chains.retrieval_qa.base import RetrievalQA

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=active_retriever,
        return_source_documents=True,
    )
    chain_type = "Legacy RetrievalQA"
    diagnostics.append("Using Legacy RetrievalQA pipeline")
except ImportError as legacy_error:
    diagnostics.append(f"Legacy RetrievalQA unavailable: {legacy_error}")
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n{context}\n\nQuestion: {input}\n\nAnswer:"
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa = create_retrieval_chain(active_retriever, document_chain)
        chain_type = "Modern LCEL Chain"
        diagnostics.append("Using Modern LCEL retrieval chain")
    except ImportError as modern_error:
        chain_error = (
            "LangChain optional modules missing. Install `langchain` with retrieval extras "
            "to enable question answering."
        )
        diagnostics.append(f"Modern chain unavailable: {modern_error}")
    except Exception as modern_generic_error:
        chain_error = f"Retrieval chain initialization failed: {modern_generic_error}"
else:
    chain_error = None

if qa is None and chain_error is None:
    chain_error = "Retrieval chain did not initialize. Check diagnostics for details."

# Streamlit UI
st.title("🤖 Jarvis AI Assistant")
st.markdown("Ask about your uploaded docs—powered by Pinecone + OpenAI!")
st.markdown(
    """
    <style>
    .jarvis-hero {
        background: radial-gradient(circle at 20% 20%, #20304f, #0d1117);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 32px;
        color: #f5f7ff;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 45px rgba(15, 27, 51, 0.55);
    }
    .jarvis-hero h2 {
        margin-bottom: 0.1rem;
    }
    .jarvis-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }
    .jarvis-pill {
        border-radius: 999px;
        padding: 6px 16px;
        background: rgba(255, 255, 255, 0.1);
        font-size: 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .jarvis-steps {
        margin-top: 1.5rem;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
    }
    .jarvis-step {
        padding: 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .jarvis-step span {
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

session_chunks = st.session_state.get("local_doc_count", 0)
query = st.text_input("Ask me anything:", placeholder="e.g., What does the doc say about AI ethics?")

if not query:
    st.markdown(
        f"""
        <div class="jarvis-hero">
            <h2>👋 Welcome to Jarvis</h2>
            <p>Your always-on AI analyst for PDFs, research decks, and knowledge bases.</p>
            <div class="jarvis-pills">
                <div class="jarvis-pill">Pinecone index: {INDEX_NAME}</div>
                <div class="jarvis-pill">LangChain {langchain_version}</div>
                <div class="jarvis-pill">OpenAI GPT-4o-mini</div>
            </div>
            <div class="jarvis-steps">
                <div class="jarvis-step">
                    <span>Step 1</span>
                    <h4>Upload files</h4>
                    <p>Drop PDF/TXT documents into the sidebar uploader.</p>
                </div>
                <div class="jarvis-step">
                    <span>Step 2</span>
                    <h4>Build local index</h4>
                    <p>Create a temporary vector store just for this session.</p>
                </div>
                <div class="jarvis-step">
                    <span>Step 3</span>
                    <h4>Ask anything</h4>
                    <p>Jarvis cites the exact snippets used in every answer.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Pinecone Vectors", f"{total_vectors:,}")
    col2.metric("Active Source", data_source_label)
    col3.metric("Session Chunks", session_chunks)
    st.info("Tip: Use the sidebar to toggle between persistent data and fresh uploads.")

if query:
    if qa is None:
        st.warning("Jarvis cannot answer questions until LangChain retrieval components are installed. See Diagnostics for fix steps.")
    else:
        with st.spinner("Jarvis is thinking..."):
            try:
                # Invoke (works for both chain types)
                if chain_type == "Legacy RetrievalQA":
                    result = qa.invoke({"query": query})
                    response = result["result"]
                    sources = result.get("source_documents", [])
                else:  # Modern
                    result = qa.invoke({"input": query})
                    response = result["answer"]
                    sources = result.get("context", [])
                
                st.subheader("**Jarvis:**")
                st.write(response)
                
                if sources:
                    with st.expander("📚 Sources (for transparency)"):
                        for i, doc in enumerate(sources):
                            source_name = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'N/A'
                            st.write(f"**Source {i+1}:** {source_name}")
                            snippet = doc.page_content[:200] + "..." if hasattr(doc, 'page_content') else str(doc)[:200] + "..."
                            st.write(f"**Snippet:** {snippet}")
            except Exception as e:
                st.error(f"⚠️ Query failed: {e}. Check console for details (e.g., empty index?).")

with st.expander("Diagnostics", expanded=False):
    if connection_message:
        st.success(f"✅ {connection_message}")
    st.caption(f"LangChain version: {langchain_version}")
    st.caption(f"📦 Answer source: {data_source_label}")
    if diagnostics:
        for note in diagnostics:
            st.write(f"• {note}")
    if chain_error:
        st.error(chain_error)

# Footer
st.markdown("---")
st.caption("💡 Tip: Ensure your ingestion script uses INDEX_NAME='jarvis' and has uploaded vectors!")