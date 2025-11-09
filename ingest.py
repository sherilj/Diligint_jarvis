import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# 1️⃣ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "jarvis-assistant-index"

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing Pinecone API key in .env file")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OpenAI API key in .env file")

# 2️⃣ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 3️⃣ Create index if it doesn’t exist
existing_indexes = [i["name"] for i in pc.list_indexes()]
print(f"📜 Existing indexes: {existing_indexes}")

if INDEX_NAME not in existing_indexes:
    print("🆕 Creating Pinecone index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Must match embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 4️⃣ Connect to index
index = pc.Index(INDEX_NAME)
print("✅ Connected to Pinecone index:", INDEX_NAME)

# 5️⃣ Load your documents
print("📄 Loading documents...")
loader = DirectoryLoader("data", glob="**/*.txt")  # Change if using PDF/Docs
docs = loader.load()

# 6️⃣ Split into chunks
print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)

# 7️⃣ Generate embeddings
print("🧠 Generating embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 8️⃣ Upload to Pinecone
print("🚀 Uploading to Pinecone...")
for doc in tqdm(docs):
    metadata = {"source": doc.metadata.get("source", "unknown")}
    vector = embeddings.embed_query(doc.page_content)
    index.upsert(vectors=[(str(hash(doc.page_content)), vector, metadata)])

print("🎉 Ingestion complete! Documents successfully embedded and uploaded.")
