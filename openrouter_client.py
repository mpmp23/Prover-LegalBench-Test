import os
from typing import List, Dict, Any, Optional

from openai import OpenAI


from openai import OpenAI
import os

class OpenAICompatibleChatClient:
    def __init__(self, model: str):
        self.model = model

        # Use OpenRouter if you're passing an OpenRouter model id like "deepseek/deepseek-r1"
        if "/" in model and model.startswith("deepseek/"):
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY='sk-or-...'")

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        else:
            # Otherwise default to HF router (your current working setup)
            api_key = os.environ.get("HF_TOKEN")
            if not api_key:
                raise ValueError("HF_TOKEN is not set. Run: export HF_TOKEN='hf_...'")

            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=api_key,
            )
