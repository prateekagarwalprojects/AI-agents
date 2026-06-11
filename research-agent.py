import os
from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, tool

# 1. Setup the Brain (Ollama)
# Make sure your Ollama is running: 'ollama run qwen2.5-coder:7b'
# model = LiteLLMModel(
#     model_id="ollama_chat/qwen2.5-coder:7b", 
#     api_base="http://localhost:11434"
# )


model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile", 
    api_key="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"  # 👈 Put your gsk_... key here
)
# 2. Define a Custom 'Save' Tool
@tool
def save_report(filename: str, content: str) -> str:
    """
    Saves the final research report to a local file.
    Args:
        filename: The name of the file (e.g., 'ai_report.md').
        content: The actual markdown text of the report.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully saved report to {filename}"

# 3. Create the Agent
# We use DuckDuckGoSearchTool (free, no key needed) 
# OR you can use TavilyTool if you have an API key.
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), save_report], 
    model=model
)

# 4. The Goal
# mission = """
# Research the current state of 'Battery Technology in 2026'. 
# Find 3 major breakthroughs, summarize them, and save the final 
# results into a file called 'battery_research_2026.md'.
# """

mission = """
You are an expert technology journalist. Your goal is to research 'Battery Technology in 2026'.

Execute these steps precisely:
1. Search the web and find 3 distinct major breakthroughs.
2. For EACH breakthrough, write a highly detailed, comprehensive 2-3 paragraph explanation. Include specific company names (like Factorial Energy or Mercedes), metrics (like the 745-mile range), and why this matters for the industry.
3. Format the final content beautifully in clean Markdown with professional headings.
4. Save the full, multi-paragraph report to 'battery_research_2026.md'.

Do NOT just copy the search snippets. Write a deep, thorough analysis.
"""

print("🕵️ Agent is going online to research...")
agent.run(mission)