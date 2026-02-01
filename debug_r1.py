import os
import json
import re
import requests

# 1. SETUP
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing!")

MODEL = "deepseek/deepseek-r1"
# We manually force the headers and body to guarantee OpenRouter sees them
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "Debug Script",
    "Content-Type": "application/json"
}

# 2. THE PAYLOAD (With strict provider forcing)
payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ],
    "provider": {
        "order": ["novita", "azure"], # FORCE these providers
        "allow_fallbacks": True
    },
    "temperature": 0.0
}

print(f"--- Sending Request to {MODEL} ---")
print(f"Providers requested: {payload['provider']['order']}")

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=30
    )
    
    # 3. DEBUGGING OUTPUT
    if response.status_code != 200:
        print(f"\nCRITICAL ERROR {response.status_code}:")
        print(response.text)
    else:
        data = response.json()
        raw_content = data['choices'][0]['message']['content']
        
        # 4. R1 CLEANUP (Strip <think> tags)
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        
        print("\nSUCCESS!")
        print(f"Raw Output (first 100 chars): {raw_content[:100]}...")
        print(f"Clean Output: {clean_content}")

except Exception as e:
    print(f"Exception: {e}")