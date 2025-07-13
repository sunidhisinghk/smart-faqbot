import os
import pickle
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import time

# Load env and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Load embedder and FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
index = faiss.read_index("faq_index.faiss")
with open("faq_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Retrieve top chunks
def get_top_k_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# Gemini call with retry
def generate_answer(query, memory):
    top_chunks = get_top_k_chunks(query)
    context = "\n".join(top_chunks)

    prompt = f"""You are an HR assistant chatbot. Use the following policy content to answer.

HR Policy:
{context}

Conversation history:
{memory}

User: {query}
Answer:"""

    for attempt in range(3):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except ResourceExhausted:
            time.sleep(5)
        except Exception as e:
            return f"Error: {str(e)}"
    return "Could not get a response after retrying."

# -------------------------------
# 🖥️ Streamlit UI
# -------------------------------
st.set_page_config(page_title="HR FAQBot", page_icon="💼")
st.title("💼 HR FAQBot")
st.write("Ask anything from the HR policy document. I’ll remember this session.")

# Session state for full chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field
user_input = st.chat_input("Say something like 'Hi' or 'What is the leave policy?'")

# If user submits a message
if user_input:
    # Generate memory from previous chats
    memory = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history])
    answer = generate_answer(user_input, memory)

    # Save the new turn
    st.session_state.chat_history.append((user_input, answer))

# Display full conversation (chat-like)
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
