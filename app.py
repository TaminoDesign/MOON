from flask import Flask, request, jsonify
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import subprocess

app = Flask(__name__)

# Load FAISS index and school data
index = faiss.read_index("faiss.index")
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_info(query):
    """Retrieve the best matching school information from FAISS."""
    query_embedding = model.encode([query]).astype(np.float32)
    _, indices = index.search(query_embedding, 1)
    return data[indices[0][0]]

def generate_response(query, school_info):
    """Use a local LLM (Mistral via Ollama) to generate a natural response."""
    prompt = f"""
    You are a friendly high school assistant. Answer the user's question naturally using the following school info:

    {school_info}

    Question: {query}
    """

    # Run the Ollama model and get the response
    result = subprocess.run(["ollama", "run", "mistral", prompt], capture_output=True, text=True)
    return result.stdout.strip()

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Get the most relevant school info
    school_info = get_relevant_info(user_query)

    # Generate a conversational response
    response = generate_response(user_query, school_info)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

