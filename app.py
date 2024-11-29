from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import json
from waitress import serve
from app import app  # replace with the actual app instance

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)

app = Flask(__name__)
CORS(app)
if __name__ == "__main__":
    app.run(debug=True)

# Load pre-trained models
model_name = "llama3.2"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model for embeddings

# Load FAQ data
with open("faqs.json", "r") as f:
    faq_data = json.load(f)

faq_questions = [item["question"] for item in faq_data]
faq_embeddings = embedder.encode(faq_questions, convert_to_tensor=True)

def get_answer(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)
    best_match_idx = scores.argmax().item()
    return faq_data[best_match_idx]["answer"]

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query", "")
    answer = get_answer(user_query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

