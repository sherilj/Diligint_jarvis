import os
from pathlib import Path
from typing import List, Sequence, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# 1️⃣ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jarvis")
DATA_PATH = Path("data")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing Pinecone API key in .env file")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OpenAI API key in .env file")

# 2️⃣ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def _extract_index_names(raw_indexes):
    """Normalize Pinecone list_indexes response into a list of names."""
    if hasattr(raw_indexes, "indexes"):
        candidates = raw_indexes.indexes
    elif isinstance(raw_indexes, dict) and "indexes" in raw_indexes:
        candidates = raw_indexes["indexes"]
    else:
        candidates = raw_indexes

    names = []
    for item in candidates:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name")
            if name:
                names.append(name)
        else:
            name = getattr(item, "name", None)
            if name:
                names.append(name)
    return names


def _ensure_index(pc_client: Pinecone, index_name: str) -> None:
    existing_indexes = _extract_index_names(pc_client.list_indexes())
    print(f"📜 Existing indexes: {existing_indexes}")

    if index_name not in existing_indexes:
        print(f"🆕 Creating Pinecone index '{index_name}'...")
        pc_client.create_index(
            name=index_name,
            dimension=1536,  # Must match embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )


_ensure_index(pc, INDEX_NAME)

# 4️⃣ Load your documents
if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Data directory '{DATA_PATH}' not found")

print("📄 Loading documents (PDF + TXT)...")
loaders = [
    DirectoryLoader(str(DATA_PATH), glob="**/*.pdf", loader_cls=PyPDFLoader),
    DirectoryLoader(str(DATA_PATH), glob="**/*.txt", loader_cls=TextLoader),
]

docs = []
for loader in loaders:
    try:
        docs.extend(loader.load())
    except Exception as exc:
        print(f"⚠️ Skipping loader {loader}: {exc}")

if not docs:
    raise ValueError("❌ No documents found in data/. Add .pdf or .txt files to ingest.")

# 5️⃣ Split into chunks
print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)
print(f"📦 Prepared {len(docs)} chunks for embedding.")

# 6️⃣ Generate embeddings and upload to Pinecone
print("🧠 Generating embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
texts = [doc.page_content for doc in docs]
vectors = embeddings.embed_documents(texts)

print("🚀 Uploading to Pinecone...")
index = pc.Index(INDEX_NAME)

def _prepare_metadata(doc) -> dict:
    metadata = dict(doc.metadata or {})
    metadata.setdefault("source", metadata.get("file_path", "unknown"))
    metadata["text"] = doc.page_content
    return metadata


def _batched(iterable: Sequence, batch_size: int) -> Sequence[Sequence]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


batch_size = 50
doc_batches = list(_batched(list(range(len(docs))), batch_size))

for batch_indices in tqdm(doc_batches, desc="Upserting", unit="batch"):
    to_upsert: List[Tuple[str, List[float], dict]] = []
    for idx in batch_indices:
        doc = docs[idx]
        vector = vectors[idx]
        metadata = _prepare_metadata(doc)
        vector_id = f"chunk-{idx}"
        to_upsert.append((vector_id, vector, metadata))
    index.upsert(vectors=to_upsert)

print("🎉 Ingestion complete! Documents successfully embedded and uploaded.")
