from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create a new OpenAI client (v1.0+ syntax)
client = OpenAI(api_key=api_key)

# Make a chat completion request
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! What's the capital of France?"}
    ]
)

# Print the result
print("✅ OpenAI Response:", response.choices[0].message.content)
