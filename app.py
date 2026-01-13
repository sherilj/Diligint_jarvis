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

# Initialize Pinecone (FIXED: Use env var, not hardcoded)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)  # FIXED: Use INDEX_NAME, not 'jarvis'

# Initialize embeddings & LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")
class PineconeRetriever(BaseRetriever):
    """Lightweight retriever for the new Pinecone SDK + LangChain."""

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


try:
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0) if isinstance(stats, dict) else getattr(stats, "total_vector_count", 0)
    st.success(f"✅ Connected to index '{INDEX_NAME}' with {total_vectors} vectors.")
except Exception as e:
    st.error(f"❌ Pinecone index check failed: {e}. Run ingestion script first? (Check index name matches.)")
    st.stop()

retriever = PineconeRetriever(index=index, embeddings=embeddings, top_k=3)

# LangChain version check & chain setup
import langchain
st.caption(f"LangChain version: {langchain.__version__}")

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

st.caption(f"📦 Answer source: {data_source_label}")

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
    st.caption("✅ Using Legacy RetrievalQA")
except ImportError as legacy_error:
    st.warning(f"Legacy import failed ({legacy_error}). Falling back to modern chain.")
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
        st.caption("✅ Using Modern LCEL Chain")
    except ImportError as modern_error:
        st.error(f"❌ Modern imports also failed: {modern_error}. Reinstall LangChain per guide.")
        st.stop()

# Streamlit UI
st.title("🤖 Jarvis AI Assistant")
st.markdown("Ask about your uploaded docs—powered by Pinecone + OpenAI!")

query = st.text_input("Ask me anything:", placeholder="e.g., What does the doc say about AI ethics?")

if query:
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

# Footer
st.markdown("---")
st.caption("💡 Tip: Ensure your ingestion script uses INDEX_NAME='jarvis' and has uploaded vectors!")