import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA

# ✅ Load environment variables
load_dotenv()

# ✅ Get API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key or not pinecone_api_key:
    st.error("❌ Missing API keys in .env file.")
    st.stop()

# ✅ Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "jarvis-assistant-index"

# ✅ Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# ✅ Create Chat Model and QA chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 🧠 Streamlit App UI
st.set_page_config(page_title="Jarvis Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Jarvis AI Assistant")
st.markdown("Ask questions about your documents — powered by **OpenAI + Pinecone**")

# ✅ Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa.invoke({"query": user_input})
            response = result["result"]
            st.markdown(response)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
