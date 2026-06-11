import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, tool

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = LiteLLMModel(
    model_id="groq/llama-3.3-70b-versatile",
    api_key=api_key,
)

HISTORY_FILE = "chat_history.jsonl"
MAX_HISTORY_TURNS = 8


@tool
def save_report(filename: str, content: str) -> str:
    """
    Saves text or research content to a local file.
    Args:
        filename: The name of the file (e.g., 'notes.txt').
        content: The actual text to save.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully saved to {filename}"


local_agent = CodeAgent(
    tools=[save_report],
    model=model,
)

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), save_report],
    model=model,
)


def load_history(limit: int = MAX_HISTORY_TURNS) -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []

    items: list[dict] = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items[-limit:]


def append_history(question: str, answer: str, answer_mode: str) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": answer_mode,
        "question": question,
        "answer": answer,
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_history_context() -> str:
    history = load_history()
    if not history:
        return "No previous chat history available."

    lines = []
    for idx, item in enumerate(history, start=1):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        lines.append(f"{idx}. User: {question}")
        lines.append(f"   Assistant: {answer[:300]}")
    return "\n".join(lines)


def is_code_request(user_prompt: str) -> bool:
    text = user_prompt.lower()
    code_signals = [
        "write code",
        "generate code",
        "python code",
        "code example",
        "script",
        "function",
        "class",
        "api",
        "debug",
        "fix my code",
        "implement",
        "algorithm",
        "sql query",
    ]
    return any(signal in text for signal in code_signals)


def make_local_prompt(user_prompt: str) -> str:
    history_context = build_history_context()
    code_requested = is_code_request(user_prompt)
    code_policy = (
        "The user explicitly asked for code. You may include concise, runnable code."
        if code_requested
        else "Do NOT output code, pseudocode, or implementation steps unless user explicitly asks for code."
    )

    return (
        "You are a smart assistant that answers from internal knowledge first. "
        "Do NOT use web search.\n\n"
        "Conversation memory (recent turns):\n"
        f"{history_context}\n\n"
        "Instruction priority:\n"
        f"- {code_policy}\n"
        "- If the question is conceptual (AI, math, science, theory), explain clearly with intuition first.\n\n"
        "Response style requirements:\n"
        "1) Start with a direct answer in plain language.\n"
        "2) Give 1-2 clear examples.\n"
        "3) If useful, add a short step-by-step method the user can apply immediately.\n"
        "4) If and only if the user asked for code, include a concise working code example.\n"
        "5) Keep it practical, not generic.\n\n"
        f"User question: {user_prompt}"
    )


def make_web_prompt(user_prompt: str) -> str:
    history_context = build_history_context()
    code_requested = is_code_request(user_prompt)
    code_policy = (
        "The user explicitly asked for code. You may include concise, runnable code."
        if code_requested
        else "Do NOT output code, pseudocode, or implementation steps unless user explicitly asks for code."
    )

    return (
        "Use web search only when needed for fresh facts. "
        "Then provide a clear answer with examples, not raw snippets.\n\n"
        "Conversation memory (recent turns):\n"
        f"{history_context}\n\n"
        f"Instruction priority: {code_policy}\n\n"
        f"User question: {user_prompt}"
    )


def prompt_needs_fresh_web_data(user_prompt: str) -> bool:
    web_signals = [
        "latest",
        "today",
        "current",
        "news",
        "recent",
        "this week",
        "this month",
        "2026",
        "price now",
        "release date",
    ]
    text = user_prompt.lower()
    return any(signal in text for signal in web_signals)


def answer_looks_uncertain(answer: str) -> bool:
    uncertain_patterns = [
        r"\bi (don't|do not) know\b",
        r"\bnot sure\b",
        r"\buncertain\b",
        r"\bcannot verify\b",
        r"\bmight be outdated\b",
    ]
    lower = answer.lower()
    return any(re.search(pattern, lower) for pattern in uncertain_patterns)


mode = "auto"

print("Smart Chat Agent is online.")
print("Default mode: auto (local-first, web only when needed)")
print("Commands: /mode local, /mode auto, /mode web")
print("History commands: /history, /history clear")
print("Type 'exit' or 'quit' to close the chat.\n")

while True:
    user_prompt = input("You: ").strip()

    if not user_prompt:
        continue

    if user_prompt.lower() in ["exit", "quit"]:
        print("Agent shutting down. Goodbye!")
        break

    if user_prompt.lower().startswith("/mode"):
        parts = user_prompt.lower().split()
        if len(parts) == 2 and parts[1] in {"local", "auto", "web"}:
            mode = parts[1]
            print(f"Mode set to: {mode}\n")
        else:
            print("Usage: /mode local | /mode auto | /mode web\n")
        continue

    if user_prompt.lower() == "/history":
        history = load_history(limit=20)
        if not history:
            print("No chat history yet.\n")
            continue
        print("Recent chat history:\n")
        for item in history:
            print(f"[{item.get('timestamp', '')}] Q: {item.get('question', '')}")
            print(f"A: {str(item.get('answer', ''))[:220]}\n")
        continue

    if user_prompt.lower() == "/history clear":
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        print("Chat history cleared.\n")
        continue

    print("\nAgent is processing your request...")
    try:
        if mode == "local":
            response = local_agent.run(make_local_prompt(user_prompt))
        elif mode == "web":
            response = web_agent.run(make_web_prompt(user_prompt))
        else:
            if prompt_needs_fresh_web_data(user_prompt):
                response = web_agent.run(make_web_prompt(user_prompt))
                effective_mode = "web-auto"
            else:
                response = local_agent.run(make_local_prompt(user_prompt))
                effective_mode = "local-auto"
                if answer_looks_uncertain(str(response)):
                    response = web_agent.run(make_web_prompt(user_prompt))
                    effective_mode = "web-fallback"

        if mode == "local":
            effective_mode = "local"
        elif mode == "web":
            effective_mode = "web"

        print(f"\nAgent Answer:\n{response}\n")
        print("-" * 80)
        append_history(user_prompt, str(response), effective_mode)
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")