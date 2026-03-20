from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq, RateLimitError, APIStatusError
from duckduckgo_search import DDGS
import os
import re
import json
import time
import math

app = FastAPI()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────
#  MODEL ROSTER  (ordered best → fastest fallback)
#  Each model has its own separate free-tier quota on Groq
#  Free tier: 30 RPM · 1,000 RPD · 6,000 TPM  per model
# ─────────────────────────────────────────────────────────────────────
CHAT_MODELS = [
    # Best quality — try first
    {"id": "llama-3.3-70b-versatile",  "name": "Llama 3.3 70B",   "ctx": 128_000},
    # Fast mid-tier
    {"id": "llama-3.1-8b-instant",     "name": "Llama 3.1 8B",    "ctx": 128_000},
    # Google model — different quota pool
    {"id": "gemma2-9b-it",             "name": "Gemma 2 9B",      "ctx":   8_192},
    # Older Meta — still works, separate quota
    {"id": "llama3-70b-8192",          "name": "Llama 3 70B",     "ctx":   8_192},
    {"id": "llama3-8b-8192",           "name": "Llama 3 8B",      "ctx":   8_192},
]

# Small/fast models only for the THINKING step (saves tokens)
THINK_MODELS = [
    {"id": "llama-3.1-8b-instant",     "name": "Llama 3.1 8B"},
    {"id": "gemma2-9b-it",             "name": "Gemma 2 9B"},
    {"id": "llama3-8b-8192",           "name": "Llama 3 8B"},
    {"id": "llama-3.3-70b-versatile",  "name": "Llama 3.3 70B"},  # fallback thinking
]

# ─────────────────────────────────────────────────────────────────────
#  RATE-LIMIT TRACKER
#  Stores Unix timestamp of when each model is available again.
#  Keys are model IDs.  Values are float Unix timestamps.
# ─────────────────────────────────────────────────────────────────────
model_cooldown: dict[str, float] = {}   # model_id -> available_at (epoch seconds)

def is_available(model_id: str) -> bool:
    return time.time() >= model_cooldown.get(model_id, 0)

def mark_rate_limited(model_id: str, retry_after_seconds: float):
    """Mark a model as unavailable for `retry_after_seconds` seconds."""
    available_at = time.time() + retry_after_seconds
    model_cooldown[model_id] = available_at
    print(f"[rate-limit] {model_id} blocked for {retry_after_seconds:.0f}s "
          f"(until {time.strftime('%H:%M:%S', time.localtime(available_at))})")

def earliest_available_seconds(models: list[dict]) -> float:
    """Return how many seconds until the soonest blocked model resets."""
    now = time.time()
    waits = [max(0, model_cooldown.get(m["id"], 0) - now) for m in models]
    return min(waits) if waits else 0

def all_blocked(models: list[dict]) -> bool:
    return all(not is_available(m["id"]) for m in models)

def parse_retry_after(exc: Exception) -> float:
    """
    Extract retry-after seconds from a Groq RateLimitError.
    Groq embeds it as: 'Please try again in Xs.' or 'retry after Xs'
    Falls back to 60s if we can't parse it.
    """
    try:
        msg = str(exc)
        # "Please try again in 12.345s." or "retry after 5s"
        match = re.search(r'(?:try again in|retry after)\s*([\d.]+)s', msg, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2   # small safety buffer
        # "Please try again in 1m30s."
        match2 = re.search(r'(\d+)m(\d+)s', msg)
        if match2:
            return int(match2.group(1)) * 60 + int(match2.group(2)) + 2
        # "Please try again in 2m."
        match3 = re.search(r'(\d+)m', msg)
        if match3:
            return int(match3.group(1)) * 60 + 2
    except Exception:
        pass
    return 62.0   # safe default: just over 1 minute

def format_wait_time(seconds: float) -> str:
    """Format seconds into a human-friendly wait string."""
    if seconds < 5:
        return "a few seconds"
    if seconds < 90:
        return f"{math.ceil(seconds)} seconds"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


# ─────────────────────────────────────────────────────────────────────
#  CORE: call Groq with automatic model fallback
# ─────────────────────────────────────────────────────────────────────
def call_with_fallback(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> tuple[str, str]:
    """
    Try each model in order, skipping ones that are rate-limited.
    Returns (reply_text, model_id_used).
    Raises RuntimeError with a user-friendly message if ALL models fail.
    """
    tried = []

    for model in models:
        mid = model["id"]

        # Skip if still in cooldown
        if not is_available(mid):
            wait = model_cooldown[mid] - time.time()
            tried.append(f"{model['name']} (cooling down, {format_wait_time(wait)} left)")
            continue

        try:
            resp = client.chat.completions.create(
                model=mid,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            text = resp.choices[0].message.content.strip()
            print(f"[model] used {mid}")
            return text, mid

        except RateLimitError as e:
            wait = parse_retry_after(e)
            mark_rate_limited(mid, wait)
            tried.append(f"{model['name']} (rate limited, retry in {format_wait_time(wait)})")
            print(f"[rate-limit] {mid} → {e}")
            continue  # try next model

        except APIStatusError as e:
            # 503 / 500 / model not found etc. — skip but don't set long cooldown
            mark_rate_limited(mid, 10)
            tried.append(f"{model['name']} (API error {e.status_code})")
            print(f"[api-error] {mid} status={e.status_code} → {e}")
            continue

        except Exception as e:
            # Network error, timeout, etc.
            tried.append(f"{model['name']} (error: {type(e).__name__})")
            print(f"[error] {mid} → {e}")
            continue

    # All models exhausted — build the friendliest possible message
    soonest = earliest_available_seconds(models)
    wait_str = format_wait_time(soonest) if soonest > 0 else "a moment"
    tried_str = "\n  • ".join(tried) if tried else "all models"

    raise RuntimeError(
        f"QUOTA_EXCEEDED|{wait_str}|{tried_str}"
    )


# ─────────────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────────────
SYSTEM = """You are Zippy, a smart AI assistant made by Arul Vethathiri.

Tone:
- Talk like a smart, calm, helpful friend. Not overly excited. Not poetic. Not dramatic.
- Short and natural like a real conversation.
- Greetings: just say hi back naturally. Short and casual.

Rules:
- Simple questions: 1-2 sentences. No fluff.
- Explanations: short bullet points, no long intros.
- Code: give the code directly, one line comment if needed.
- Creative tasks: complete the full piece, no preamble.
- Math: step by step, brief.
- NEVER say: Certainly!, Great question!, Of course!, Absolutely!, As an AI, traveler, delightful.
- Never repeat the question back.
- Always end with 1 relevant emoji."""

THINK_PROMPT = """You are a reasoning engine for an AI assistant called Zippy.

Given a user's message, output ONLY a valid JSON object (no markdown, no backticks) with:
- "needs_search": boolean — true ONLY if real-time/recent info is needed
- "search_query": string — best search query if needs_search is true, else ""
- "reasoning": string — 1 sentence explaining your decision

Search IS needed for: current events, breaking news, live prices/stocks, sports scores,
weather, recent product releases, "latest", "today", "this week", "right now" questions.

Search is NOT needed for: coding, math, history, concepts, creative writing, opinions,
greetings, identity questions, timeless knowledge.

Output raw JSON only. Example:
{"needs_search": true, "search_query": "India vs Australia cricket today", "reasoning": "User asked for a live sports score."}"""

# ─────────────────────────────────────────────────────────────────────
#  STATIC REPLIES
# ─────────────────────────────────────────────────────────────────────
IDENTITY = {
    "who are you":       "I'm Zippy, a powerful AI chatbot made by Arul Vethathiri! 🤖",
    "what are you":      "I'm Zippy, an AI made by Arul Vethathiri. 🤖",
    "who made you":      "I was made by Arul Vethathiri. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "who built you":     "I was built by Arul Vethathiri. 👨‍💻",
    "what is your name": "My name is Zippy! 😊",
    "are you an ai":     "Yes! I'm Zippy, an AI made by Arul Vethathiri. 🤖",
    "are you human":     "Nope! I'm Zippy — an AI but great at conversation! 😄",
}

SOCIAL = {
    "thanks":     "You're welcome! 😊",
    "thank you":  "Happy to help anytime! 🌟",
    "bye":        "Goodbye! Take care! 👋",
    "goodbye":    "See you later! 👋",
    "good night": "Good night! Sleep well! 🌙",
}

# ─────────────────────────────────────────────────────────────────────
#  WEB SEARCH
# ─────────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 4) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"[search] DuckDuckGo failed: {e}")
        return []

def format_search_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title','')}\n   {r.get('body','')[:300]}\n   Source: {r.get('href','')}")
    return "\n\n".join(lines)

# ─────────────────────────────────────────────────────────────────────
#  THINKING STEP
# ─────────────────────────────────────────────────────────────────────
def run_thinking(user_input: str) -> dict:
    """Use a small/fast model just for the structured thinking JSON."""
    messages = [
        {"role": "system", "content": THINK_PROMPT},
        {"role": "user",   "content": user_input},
    ]
    try:
        raw, _ = call_with_fallback(
            messages,
            THINK_MODELS,
            max_tokens=200,
            temperature=0.2,
            top_p=0.95,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)
    except RuntimeError:
        # Thinking step failed due to quota — degrade gracefully
        return {
            "needs_search": False,
            "search_query": "",
            "reasoning": "Thinking step skipped (quota).",
        }
    except Exception as e:
        return {
            "needs_search": False,
            "search_query": "",
            "reasoning": f"Thinking step skipped ({type(e).__name__}).",
        }

# ─────────────────────────────────────────────────────────────────────
#  REQUEST MODEL
# ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ─────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    # Show current model status for debugging
    now = time.time()
    status = {}
    for m in CHAT_MODELS:
        mid = m["id"]
        cooldown = model_cooldown.get(mid, 0)
        if cooldown > now:
            status[mid] = f"blocked for {format_wait_time(cooldown - now)}"
        else:
            status[mid] = "available"
    return {"status": "Zippy backend is running!", "models": status}


@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Clean filler sounds
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message, flags=re.IGNORECASE
    ).strip()

    if not user_input:
        return {
            "reply":    "Didn't catch that, say again? 😊",
            "thinking": "Empty input.",
            "searched": False,
            "sources":  [],
        }

    text = user_input.lower().strip()

    # 2. Static replies — no model needed
    if text in IDENTITY:
        return {"reply": IDENTITY[text], "thinking": "Identity question — hardcoded.", "searched": False, "sources": []}
    if text in SOCIAL:
        return {"reply": SOCIAL[text],   "thinking": "Social phrase — hardcoded.",    "searched": False, "sources": []}

    # 3. Thinking step (cheap model, structured JSON)
    think         = run_thinking(user_input)
    needs_search  = think.get("needs_search", False)
    search_query  = think.get("search_query", "")
    reasoning     = think.get("reasoning", "")

    # 4. Web search if needed
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search and search_query:
        results = web_search(search_query)
        if results:
            search_context = format_search_context(results)
            searched = True
            search_sources = [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:200]}
                for r in results[:4]
            ]

    # 5. Build message list for final answer
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    messages += req.history[-20:]

    final_user = (
        f"{search_context}\n\nUsing the above search results, answer this:\n{user_input}"
        if search_context else user_input
    )
    messages.append({"role": "user", "content": final_user})

    # 6. Try each chat model until one works
    try:
        reply, model_used = call_with_fallback(
            messages,
            CHAT_MODELS,
            max_tokens=512,
            temperature=0.85,
            top_p=0.92,
        )

        return {
            "reply":      reply,
            "thinking":   reasoning,
            "searched":   searched,
            "sources":    search_sources,
            "model_used": model_used,
        }

    except RuntimeError as exc:
        # Parse our structured error string: "QUOTA_EXCEEDED|wait_str|tried_str"
        parts = str(exc).split("|", 2)
        if len(parts) == 3 and parts[0] == "QUOTA_EXCEEDED":
            wait_str  = parts[1]
            tried_str = parts[2]
        else:
            wait_str  = "a few minutes"
            tried_str = str(exc)

        quota_reply = (
            f"⚠️ **All AI models are currently rate-limited.**\n\n"
            f"Please wait at least **{wait_str}** before trying again.\n\n"
            f"Groq free tier allows 30 requests/minute and 1,000 requests/day per model. "
            f"All 5 available models have been temporarily exhausted.\n\n"
            f"Models tried:\n"
            + "\n".join(f"• {line}" for line in tried_str.split("\n  • ") if line)
        )

        return {
            "reply":      quota_reply,
            "thinking":   "All models are rate-limited — returning quota exceeded message.",
            "searched":   False,
            "sources":    [],
            "model_used": "none",
        }
