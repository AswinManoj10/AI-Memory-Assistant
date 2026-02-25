from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import os
import pickle

app = Flask(__name__)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
index_file = "faiss_index.bin"
memory_file = "memories.pkl"

# Load existing index if available
if os.path.exists(index_file) and os.path.exists(memory_file):
    index = faiss.read_index(index_file)
    with open(memory_file, "rb") as f:
        memories = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dimension)
    memories = []

# Store memory
def store_memory(text):
    embedding = model.encode([text])
    index.add(np.array(embedding))
    memories.append(text)

    faiss.write_index(index, index_file)
    with open(memory_file, "wb") as f:
        pickle.dump(memories, f)

# Retrieve relevant memories
def retrieve_context(query, k=3):
    if len(memories) == 0:
        return "No stored memories."

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    for i in indices[0]:
        if i < len(memories):
            results.append(memories[i])

    return "\n".join(results)

# Ask Ollama
def ask_llm(query):
    context = retrieve_context(query)

    prompt = f"""
You are a memory assistant.

Stored memories:
{context}

Answer ONLY using the stored memories.
Do not invent anything.

User Question:
{query}

Answer:
"""

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "mistral:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        data = response.json()

        if "response" in data:
            return data["response"]
        elif "error" in data:
            return f"Ollama Error: {data['error']}"
        else:
            return "No response received."

    except Exception as e:
        return f"Connection Error: {str(e)}"

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    memory = data.get("memory")

    if memory:
        store_memory(memory)
        return jsonify({"message": "Memory saved successfully."})
    else:
        return jsonify({"message": "No memory provided."})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"response": "No question provided."})

    answer = ask_llm(question)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
