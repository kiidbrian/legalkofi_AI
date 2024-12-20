import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"
HEADERS = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}

def query_llama3(messages):
   # Payload for the API
   payload = {
       "inputs": messages,
       "parameters": {"max_new_tokens": 500, "temperature": 0.3},
   }
  
   # Make the API request
   response = requests.post(API_URL, headers=HEADERS, json=payload)
   if response.status_code == 200:
       result = response.json()
      
       # Extract the response text
       response = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
      
       # Further clean any leading colons or formatting
       if ":" in response:
           response = response.split(":", 1)[-1].strip()
      
       return response or "No explanation available."
   else:
       return f"Error: {response.status_code} - {response.text}"