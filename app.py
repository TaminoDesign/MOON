from flask import Flask, request, jsonify, render_template
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Load models and data
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index_plain.faiss")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token="your-huggingface-token-here"  # Replace this
)


def get_answer(question, k=3):
    embedding = embedder.encode([question])
    distances, indices = index.search(np.array(embedding), k)
    context = "\n---\n".join([texts[i] for i in indices[0]])
    prompt = f"""You are a helpful assistant for a school chatbot. Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    return client.text_generation(prompt, max_new_tokens=200, temperature=0.5)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer = get_answer(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
