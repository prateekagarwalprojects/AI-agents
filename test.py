import os
from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, tool
# 1. Setup the Brain (Ollama)
# Make sure your Ollama is running: 'ollama run qwen2.5-coder:7b'
model = LiteLLMModel(api_key="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")