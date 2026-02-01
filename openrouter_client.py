import os
import re
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI

class OpenAICompatibleChatClient:
    def __init__(self, model: str, http_referer: Optional[str] = None, x_title: Optional[str] = None):
        self.model = model
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("No API key found. Set OPENROUTER_API_KEY.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": http_referer or "http://localhost:8000",
                "X-Title": x_title or "Eval Script",
            }
        )

    def complete(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        # We use the 'extra_body' parameter but we structure it specifically
        # to ensure the 'provider' object is at the top level of the JSON request.
        
        # Merge your evaluation kwargs with our provider force
        request_params = {
            "model": self.model,
            "messages": messages,
            "extra_body": {
                "provider": {
                    "order": ["novita", "azure"],
                    "allow_fallbacks": True
                }
            },
            **kwargs
        }

        try:
            resp = self.client.chat.completions.create(**request_params)
            raw_content = resp.choices[0].message.content or ""
            
            # R1 Stripper: Removes the thinking process so it doesn't break your eval
            clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
            return clean_content.strip()

        except Exception as e:
            # If it still fails, let's see exactly what the client sent
            print(f"\n[DEBUG] Request sent for model: {self.model}")
            raise e