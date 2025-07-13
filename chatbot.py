import os
import pickle
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Load SentenceTransformer model for retrieval
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and policy chunks
index = faiss.read_index("faq_index.faiss")
with open("faq_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Retrieve top matching chunks
def get_top_k_chunks(query, k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# Generate answer using Gemini
def generate_answer(query):
    top_chunks = get_top_k_chunks(query)
    context = "\n".join(top_chunks)

    prompt = f"""You are an HR assistant. Use the following HR policy content to answer the question.

HR Policy Context:
{context}

Question: {query}

Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text

# Chatbot loop
if __name__ == "__main__":
    print("[Gemini HR FAQBot] Ask a question or type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        answer = generate_answer(user_input)
        print("Bot:", answer)
