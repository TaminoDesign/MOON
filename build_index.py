import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read school data
with open("school_data.txt", "r") as f:
    raw_text = f.read().strip().split("\n")

# Create embeddings for each line of text
embeddings = model.encode(raw_text)

# Initialize FAISS index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings, dtype=np.float32))

# Save index and data
faiss.write_index(index, "faiss.index")
with open("data.pkl", "wb") as f:
    pickle.dump(raw_text, f)

print("FAISS index built and saved.")

