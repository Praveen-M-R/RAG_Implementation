from flask import Flask, request, jsonify, render_template_string
import pickle
from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

app = Flask(__name__)

api_key = "jCQDmCM6T3mx85GUFFNFt8BNAEpwbzv9"

# Load the vector store index
with open("vector_store_index.pkl", "rb") as f:
    index = pickle.load(f)

# Define LLM and embedding model
llm = MistralAI(api_key=api_key, model="mistral-medium")
embed_model = MistralAIEmbedding(api_key=api_key, model_name="mistral-embed")
Settings.llm = llm
Settings.embed_model = embed_model

# Create query engine
query_engine = index.as_query_engine()

# Initialize chat history
chat_history = []

@app.route('/')
def home():
    return render_template_string('''
        <!doctype html>
        <title>Chatbot</title>
        <h1>Chatbot</h1>
        <div id="chat-history" style="border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll;">
            {% for message in chat_history %}
                <div style="margin-bottom: 10px;">
                    <strong>{{ message['role'] }}:</strong> {{ message['content'] }}
                </div>
            {% endfor %}
        </div>
        <form action="/query" method="post">
            <input type="text" name="query" placeholder="Enter your query">
            <input type="submit" value="Send">
        </form>
    ''', chat_history=chat_history)

@app.route('/query', methods=['POST'])
def query():
    global chat_history
    user_query = request.form['query']

    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_query})

    # Get response from query engine
    response = query_engine.query(user_query)

    # Add bot response to chat history
    chat_history.append({"role": "assistant", "content": response})

    return render_template_string('''
        <!doctype html>
        <title>Chatbot</title>
        <h1>Chatbot</h1>
        <div id="chat-history" style="border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll;">
            {% for message in chat_history %}
                <div style="margin-bottom: 10px;">
                    <strong>{{ message['role'] }}:</strong> {{ message['content'] }}
                </div>
            {% endfor %}
        </div>
        <form action="/query" method="post">
            <input type="text" name="query" placeholder="Enter your query">
            <input type="submit" value="Send">
        </form>
    ''', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
