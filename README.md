# EragAPI - Unified AI Service Interface

EragAPI provides a unified FastAPI-based interface for multiple AI providers, offering seamless integration with Groq, Gemini, Cohere, DeepSeek, and Ollama. Features streaming responses, CLI tools, and a system tray icon for server management.

## Features
- **Multi-Provider Support**: Connect to Groq, Gemini, Cohere, DeepSeek, and Ollama.
- **Streaming Responses**: Server-Sent Events (SSE) for real-time outputs.
- **Simple CLI**: Start the server/list models with terminal commands.
- **System Tray Integration**: Background server management with tray icon.
- **Ollama Compatibility**: Works with locally hosted Ollama models.

## Installation
### 1. Requirements:
- Python 3.9+
- [Ollama](https://ollama.ai/) (for Ollama integration)

### 2. Install dependencies:
```bash
pip install fastapi uvicorn pydantic python-dotenv google-generativeai cohere requests pystray pillow groq openai
```

## Configuration
Create a `.env` file with your API keys:

```env
GROQ_API_KEY="your_groq_key"
GEMINI_API_KEY="your_gemini_key"
CO_API_KEY="your_cohere_key"
DEEPSEEK_API_KEY="your_deepseek_key"
```

## Usage
### Start the API Server
```bash
python erag_API_v10.py serve
```
Add `--tray` to enable system tray icon.

### List Available Models
```bash
# List all providers' models
python erag_API_v10.py model list

# Filter by provider
python erag_API_v10.py model list --api groq
```

## API Endpoints
### Chat Completion
**POST** `/api/chat`

```json
{
  "model": "groq-mixtral-8x7b-32768",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false,
  "temperature": 0.7
}
```

### Text Generation
**POST** `/api/generate`

```json
{
  "model": "gemini-gemini-pro",
  "prompt": "Write a poem about AI",
  "stream": true,
  "temperature": 0.5
}
```

## Example Requests
### Basic Chat (cURL)
```bash
curl -X POST http://localhost:11436/api/chat \
-H "Content-Type: application/json" \
-d '{
  "model": "groq-mixtral-8x7b-32768",
  "messages": [{"role": "user", "content": "Explain quantum computing"}]
}'
```

### Streaming Response (Python)
```python
import requests

response = requests.post(
    "http://localhost:11436/api/generate",
    json={
        "model": "ollama-llama2",
        "prompt": "Tell me a joke",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8').replace('data: ', ''))
```

## Model Naming Convention
**Format:** `{provider}-{model_name}`

**Examples:**
- `groq-mixtral-8x7b-32768`
- `ollama-llama2-uncensored`
- `deepseek-deepseek-chat`

## Notes
- Ollama models must be pulled first (e.g., `ollama pull llama2`).
- Default models are used if no specific model is provided.
- Temperature range: `0.0` (deterministic) to `1.0` (creative).
- Stream responses for long-generation tasks to avoid timeouts.

## License
MIT - Add appropriate license file.
