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

from flask import Flask, request, jsonify
import os
import slack
import json

app = Flask(__name__)

SLACK_TOKEN = "670PokSihTB5DdRvk67vePgP"  # You'll get this from Slack
client = slack.WebClient(token=SLACK_TOKEN)

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        user_query = event.get("text").replace("<@A08HJJWC1ME>", "").strip()
        response = get_bot_response(user_query)
        client.chat_postMessage(
            channel=event["channel"],
            text=response
        )
    return jsonify({"status": "ok"})


def get_bot_response(query):
    # Here, you can use your existing logic to get the response from the chatbot
    return f"Here's the answer to: {query}"

if __name__ == "__main__":
    app.run(debug=True)
