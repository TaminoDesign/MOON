from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def load_docs(folder="data"):
    docs = []
    for file_path in Path(folder).rglob("*.*"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(Document(page_content=text, metadata={"source": str(file_path)}))
    return docs


splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = load_docs()
chunks = splitter.split_documents(docs)

# For now, just print out a sample
print(f"Loaded {len(docs)} documents and split into {len(chunks)} chunks.")


from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text chunks into embeddings
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

# Create a FAISS index
dimension = embeddings.shape[1]  # Should be 384 for this model
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "index.faiss")
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print(f"✅ Saved {len(texts)} embeddings to index.faiss and texts.pkl")
