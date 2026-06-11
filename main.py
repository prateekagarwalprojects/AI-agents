from smolagents import CodeAgent, LiteLLMModel

# 1. Connect to your local brain (Ollama)
# api_base is the default address where Ollama runs on your computer
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:7b", 
    api_base="http://localhost:11434"
)

# 2. Initialize the Agent 
# (We provide zero tools because this is a 'knowledge' question)
agent = CodeAgent(tools=[], model=model)

# 3. Run the simplest mission
print("🤖 Agent is thinking...")
response = agent.run(
    "Answer from your internal knowledge only. "
    "Do not write or execute code. Do not use tools. "
    "Who won the men's cricket world cup in 2023?"
)

print(f"\nFinal Answer: {response}")