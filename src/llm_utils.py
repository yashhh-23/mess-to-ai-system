import os
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

class LLMClient:
    """
    Multi-Provider LLM Client supporting:
    - OpenAI ('openai'): https://api.openai.com/v1/chat/completions
    - Gemini ('gemini'): https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
    - Ollama ('ollama'): http://localhost:11434/v1/chat/completions
    - Auto ('auto'): Auto-detects based on environment variables or local endpoint
    
    Includes detailed tracking of provider, model name, status, and fallback state.
    """
    def __init__(self, provider: str = "auto", api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider.lower()
        self.base_url = base_url
        
        # Determine Provider & Key
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        
        if self.provider == "auto":
            if api_key:
                self.api_key = api_key
                self.provider = "openai"
            elif openai_key:
                self.api_key = openai_key
                self.provider = "openai"
            elif gemini_key:
                self.api_key = gemini_key
                self.provider = "gemini"
            else:
                self.api_key = None
                self.provider = "none"
        else:
            self.api_key = api_key or (openai_key if self.provider == "openai" else gemini_key)

        # Set default models per provider
        if model_name:
            self.model_name = model_name
        elif self.provider == "gemini":
            self.model_name = "gemini-1.5-flash"
        elif self.provider == "ollama":
            self.model_name = "llama3"
        else:
            self.model_name = "gpt-3.5-turbo"

    def generate_detailed(self, prompt: str, system_prompt: str = "", temperature: float = 0.2, max_tokens: int = 250) -> Dict[str, Any]:
        """
        Executes request and returns detailed metadata dictionary:
        {
            'content': str or None,
            'provider': str,
            'model_name': str,
            'used_fallback': bool,
            'error': str or None,
            'raw_response': str or None
        }
        """
        if self.provider == "none" or (self.provider in ["openai", "gemini"] and not self.api_key):
            return {
                'content': None,
                'provider': self.provider,
                'model_name': self.model_name,
                'used_fallback': True,
                'error': "No API key configured for provider",
                'raw_response': None
            }

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.provider in ["openai", "gemini"]:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Endpoint selection
        if self.base_url:
            endpoint = self.base_url
        elif self.provider == "gemini":
            endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif self.provider == "ollama":
            endpoint = "http://localhost:11434/v1/chat/completions"
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                raw_body = response.read().decode("utf-8")
                res_json = json.loads(raw_body)
                content = res_json["choices"][0]["message"]["content"].strip()
                return {
                    'content': content,
                    'provider': self.provider,
                    'model_name': self.model_name,
                    'used_fallback': False,
                    'error': None,
                    'raw_response': raw_body
                }
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return {
                'content': None,
                'provider': self.provider,
                'model_name': self.model_name,
                'used_fallback': True,
                'error': error_msg,
                'raw_response': None
            }

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.2, max_tokens: int = 250) -> Optional[str]:
        """Convenience method returning content string or None."""
        res = self.generate_detailed(prompt, system_prompt, temperature, max_tokens)
        return res['content']
