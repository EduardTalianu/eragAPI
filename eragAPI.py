# Standard library imports
import os
import sys
import argparse
import uvicorn
import threading
import signal
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Third-party imports
import google.generativeai as genai
import cohere
import requests
import subprocess
import pystray
from PIL import Image, ImageDraw
from groq import Groq
from openai import OpenAI

load_dotenv()

class BaseClient:
    def chat(self, messages, temperature, max_tokens, stream):
        raise NotImplementedError
    
    def complete(self, prompt, temperature, max_tokens, stream):
        raise NotImplementedError

class GroqClient(BaseClient):
    def __init__(self, model="mixtral-8x7b-32768"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        return completion if stream else completion.choices[0].message.content

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        completion = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        return completion if stream else completion.choices[0].text

class GeminiClient(BaseClient):
    def __init__(self, model="gemini-pro"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        formatted = [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages]
        response = self.model.generate_content(
            formatted,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=stream
        )
        return response if stream else response.text

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=stream
        )
        return response if stream else response.text

class CohereClient(BaseClient):
    def __init__(self, model="command"):
        self.client = cohere.Client(api_key=os.getenv("CO_API_KEY"))
        self.model = model

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        history = [{"role": "USER" if m["role"] == "user" else "CHATBOT", 
                   "message": m["content"]} for m in messages]
        return self.client.chat(
            message=history[-1]["message"],
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            chat_history=history[:-1]
        )

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        return self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )

class DeepSeekClient(BaseClient):
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        self.model = model

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        return response if stream else response.choices[0].message.content

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        return self.chat([{"role": "user", "content": prompt}], temperature, max_tokens, stream)

class OllamaClient(BaseClient):
    def __init__(self, model="llama2"):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model = model

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        return response if stream else response.choices[0].message.content

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        return response if stream else response.choices[0].text

class EragAPI:
    CLIENTS = {
        "groq": GroqClient,
        "gemini": GeminiClient,
        "cohere": CohereClient,
        "deepseek": DeepSeekClient,
        "ollama": OllamaClient
    }

    def __init__(self, api_type, model=None):
        self.client = self.CLIENTS[api_type](model or self.default_model(api_type))

    @staticmethod
    def default_model(api_type):
        return {
            "groq": "mixtral-8x7b-32768",
            "gemini": "gemini-pro",
            "cohere": "command",
            "deepseek": "deepseek-chat",
            "ollama": "llama2"
        }[api_type]

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        return self.client.chat(messages, temperature, max_tokens, stream)

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        return self.client.complete(prompt, temperature, max_tokens, stream)

app = FastAPI(title="EragAPI", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = None

def parse_model_string(model_string):
    """Parse model string in format 'provider-model_name' where model_name may contain hyphens"""
    # Known providers
    providers = ["groq", "gemini", "cohere", "deepseek", "ollama"]
    
    for provider in providers:
        if model_string.startswith(provider + "-"):
            # Extract the model name after the provider prefix
            model_name = model_string[len(provider) + 1:]
            return provider, model_name
    
    # If no known provider found, try to split by first hyphen as fallback
    parts = model_string.split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return model_string, None

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Parse the model string properly
        provider, model_name = parse_model_string(request.model)
        
        if not model_name:
            raise HTTPException(400, f"Invalid model format: {request.model}")
        
        if provider not in EragAPI.CLIENTS:
            raise HTTPException(400, f"Unknown provider: {provider}")
        
        erag = EragAPI(provider, model_name)
        
        if request.stream:
            def stream_generator():
                for chunk in erag.chat(request.messages, request.temperature, request.max_tokens, True):
                    yield f"data: {chunk}\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
        return {"message": erag.chat(request.messages, request.temperature, request.max_tokens)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        # Parse the model string properly
        provider, model_name = parse_model_string(request.model)
        
        if not model_name:
            raise HTTPException(400, f"Invalid model format: {request.model}")
        
        if provider not in EragAPI.CLIENTS:
            raise HTTPException(400, f"Unknown provider: {provider}")
        
        erag = EragAPI(provider, model_name)
        
        if request.stream:
            def stream_generator():
                for chunk in erag.complete(request.prompt, request.temperature, request.max_tokens, True):
                    yield f"data: {chunk}\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
        return {"response": erag.complete(request.prompt, request.temperature, request.max_tokens)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/models/{provider}")
async def get_models_endpoint(provider: str):
    """Get available models for a specific provider"""
    try:
        if provider not in EragAPI.CLIENTS:
            raise HTTPException(404, f"Provider '{provider}' not found")
        
        models = get_available_models(provider)
        return {"provider": provider, "models": models}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/models")
async def get_all_models_endpoint():
    """Get available models for all providers"""
    try:
        all_models = {}
        for provider in EragAPI.CLIENTS.keys():
            models = get_available_models(provider)
            if models:  # Only include providers that have models
                all_models[provider] = models
        return {"models": all_models}
    except Exception as e:
        raise HTTPException(500, str(e))

def create_tray_icon():
    def create_image():
        image = Image.new("RGB", (64, 64), "#3b82f6")
        dc = ImageDraw.Draw(image)
        dc.rectangle([10, 10, 54, 54], fill="white")
        dc.rectangle([20, 20, 30, 44], fill="#3b82f6")
        dc.rectangle([20, 20, 44, 30], fill="#3b82f6")
        dc.rectangle([20, 34, 44, 44], fill="#3b82f6")
        return image

    icon = pystray.Icon("eragapi", create_image(), "ERAG API", menu=pystray.Menu(
        pystray.MenuItem("Quit", lambda: (icon.stop(), os.kill(os.getpid(), signal.SIGTERM)))
    ))
    icon.run()

def start_server(host, port, tray=False):
    if tray:
        threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": host, "port": port}, daemon=True).start()
        create_tray_icon()
    else:
        uvicorn.run(app, host=host, port=port)

def get_available_models(api_type):
    try:
        if api_type == "ollama":
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error listing Ollama models: {result.stderr}")
                return []
            models = [model.split()[0] for model in result.stdout.strip().split('\n')[1:] 
                     if model.split() and model.split()[0] not in ['failed', 'NAME']]
            return models
        
        elif api_type == "groq":
            if not os.getenv("GROQ_API_KEY"):
                print("GROQ_API_KEY not found in environment")
                return []
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            return [model.id for model in client.models.list().data]
        
        elif api_type == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                print("GEMINI_API_KEY not found in environment")
                return []
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            return [model.name for model in genai.list_models() 
                   if 'generateContent' in model.supported_generation_methods]
        
        elif api_type == "cohere":
            if not os.getenv("CO_API_KEY"):
                print("CO_API_KEY not found in environment")
                return []
            # Synchronous version for Cohere
            client = cohere.Client(api_key=os.getenv("CO_API_KEY"))
            response = client.models.list()
            return [model.name for model in response.models if 'chat' in model.endpoints]
        
        elif api_type == "deepseek":
            if not os.getenv("DEEPSEEK_API_KEY"):
                print("DEEPSEEK_API_KEY not found in environment")
                return []
            # Return both chat and reasoner models
            return ["deepseek-chat", "deepseek-reasoner"]
        
        else:
            print(f"Unknown API type: {api_type}")
            return []
            
    except Exception as e:
        print(f"Error getting models for {api_type}: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description="EragAPI - Unified AI Service")
    parser.add_argument("--api", choices=EragAPI.CLIENTS.keys(), help="API provider override")
    
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=11436)
    serve_parser.add_argument("--tray", action="store_true")
    
    model_parser = subparsers.add_parser("model", help="Model operations")
    model_subparsers = model_parser.add_subparsers(dest="model_command")
    
    list_parser = model_subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--api", choices=EragAPI.CLIENTS.keys(), help="Only list models for specific API")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        start_server(args.host, args.port, args.tray)
    elif args.command == "model" and args.model_command == "list":
        api_to_check = []
        if hasattr(args, 'api') and args.api:
            api_to_check = [args.api]
        else:
            api_to_check = EragAPI.CLIENTS.keys()
        
        for api_type in api_to_check:
            print(f"{api_type.upper()} models:")
            models = get_available_models(api_type)
            if models:
                for model in models:
                    print(f"  - {model}")
            else:
                print(f"  - {EragAPI.default_model(api_type)} (default)")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()