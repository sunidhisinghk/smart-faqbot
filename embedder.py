import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Initialize model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Directory with documents
data_dir = "data/"
chunks = []

# Simple chunking function
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Read all .txt files and chunk them
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            content = f.read()
            chunks += chunk_text(content)

# Create embeddings
embeddings = embedder.encode(chunks)

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and chunks
faiss.write_index(index, "faq_index.faiss")
with open("faq_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Embeddings created from all documents and saved successfully!")
