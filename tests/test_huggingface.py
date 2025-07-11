# tests/test_huggingface.py
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()
hf_token = os.getenv("HF_API_TOKEN")

headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

API_URL = "https://api-inference.huggingface.co/models/gpt2"

payload = {
    "inputs": "The capital of Italy is",
    "parameters": {"max_new_tokens": 20}
}

response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    print("✅ HF Response:", response.json()[0]["generated_text"])
else:
    print(f"❌ HF Request failed: {response.status_code} - {response.text}")
