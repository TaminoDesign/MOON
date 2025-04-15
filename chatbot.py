import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Load your Hugging Face token
import os
client = InferenceClient(model="...", token=os.environ["HF_TOKEN"])
  # <-- Replace this with your real token


# Load the index and text chunks
index = faiss.read_index("index.faiss")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Hugging Face Inference Client with a good instruct model
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token="HF_TOKEN")

def ask_question(question, k=3):
    # Embed user question
    query_embedding = embedder.encode([question])
    
    # Search FAISS index for relevant chunks
    distances, indices = index.search(np.array(query_embedding), k)
    context = "\n---\n".join([texts[i] for i in indices[0]])

    # Format prompt
    prompt = f"""You are a helpful assistant for a school chatbot named MOON. Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    # Generate answer using Hugging Face API
    response = client.text_generation(prompt, max_new_tokens=200, temperature=0.5)
    return response

# Command line loop
if __name__ == "__main__":
    while True:
        q = input("\n🎓 Ask MOON something (or type 'quit'): ")
        if q.lower() in ["quit", "exit"]:
            break
        answer = ask_question(q)
        print("\n🤖 Answer:\n", answer)
