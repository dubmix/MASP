import os
from dotenv import load_dotenv
import requests
import chromadb
import uuid

from models import GPTResponse
from pydantic import TypeAdapter
from flask import Flask, request, jsonify
from flask.logging import default_handler
from flask_cors import CORS
from logger import logger
from chromadb.utils import embedding_functions

GPT_MODEL = "gpt-3.5-turbo"
GPT_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

app = Flask(__name__)
app.logger.handlers = logger.handlers
# wip, using 2>> to redirect stderr logs to flask.log
app.logger.removeHandler(default_handler)
app.logger.setLevel(logger.level)
CORS(app)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# for docker build, use this line:
client = chromadb.HttpClient(host="chroma", port=8000)
# client = chromadb.HttpClient(host="localhost", port=8000)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small")

collection = client.get_or_create_collection(
                name="test_collection", 
                embedding_function=openai_ef)

app.logger.info("Server ready")

def stringify_headers(headers):
    header_string = ""
    for key, value in headers.items():
        header_string += f"{key}: {value}; "
    return header_string.rstrip("; ")

def validate_request(bubble_id: str):
    headers = stringify_headers(request.headers)
    logger.debug(f"{bubble_id} | HTTP request headers: {headers}")
    method = request.method
    logger.debug(f"{bubble_id} | HTTP request method: {method}")
    url = request.url
    logger.debug(f"{bubble_id} | HTTP request url: {url}")

@app.route('/api/title')
def title():
    return "MASP"

@app.route('/api/process-request', methods=['POST'])
def process_request():
    bubble_id = (uuid.uuid4().hex)[:6]
    logger.info(f"Processing request {bubble_id}")
    validate_request(bubble_id)
    question = request.json.get('request')
    logger.debug(f"{bubble_id} | Received question from frontend: {question}")

    query = collection.query(query_texts=[question], n_results=2, include=["documents"])
    context = query["documents"]
    logger.debug(f"{bubble_id} | Context: {context}")

    prompt = (f"""
    Answer the question based only on the following context:
    Context: {context} 
    Question: {question}
    """)
    data = {
        "model": GPT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    logger.debug(f"{bubble_id} | Request made to OpenAI: {data}")

    gpt_response = (requests.post(GPT_API_ENDPOINT, headers=headers, json=data)).json()
    logger.debug(f"{bubble_id} | Response from OpenAI: {gpt_response}")

    model = TypeAdapter(GPTResponse).validate_python(gpt_response)
    answer = model.choices[0].message.content
    logger.debug(f"{bubble_id} | Answer sent to frontend: {answer}")
    logger.info(f"{bubble_id} | Prompt tokens used: {model.usage.prompt_tokens}")
    logger.info(f"{bubble_id} | Completion tokens used: {model.usage.completion_tokens}")
    logger.info(f"{bubble_id} | Total tokens used: {model.usage.total_tokens}")

    logger.info(f"Generated bubble {bubble_id}")
    return jsonify({'status': f'Generated bubble {bubble_id}', 'answer': answer})

if __name__ == '__main__':
    # for docker build, use this line:
    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=True, host='127.0.0.1')