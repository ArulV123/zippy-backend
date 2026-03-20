from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq, RateLimitError, APIStatusError
import requests
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
#  MODEL ROSTER
#  Each model has its own independent free-tier quota:
#  30 RPM · 1,000 RPD · 6,000 TPM per model
# ─────────────────────────────────────────────────────────────────────
CHAT_MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B",  "ctx": 128_000},
    {"id": "llama-3.1-8b-instant",    "name": "Llama 3.1 8B",   "ctx": 128_000},
    {"id": "gemma2-9b-it",            "name": "Gemma 2 9B",     "ctx":   8_192},
    {"id": "llama3-70b-8192",         "name": "Llama 3 70B",    "ctx":   8_192},
    {"id": "llama3-8b-8192",          "name": "Llama 3 8B",     "ctx":   8_192},
]

THINK_MODELS = [
    {"id": "llama-3.1-8b-instant",    "name": "Llama 3.1 8B"},
    {"id": "gemma2-9b-it",            "name": "Gemma 2 9B"},
    {"id": "llama3-8b-8192",          "name": "Llama 3 8B"},
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B"},
]

# ─────────────────────────────────────────────────────────────────────
#  RATE-LIMIT TRACKER
# ─────────────────────────────────────────────────────────────────────
model_cooldown: dict[str, float] = {}

def is_available(model_id: str) -> bool:
    return time.time() >= model_cooldown.get(model_id, 0)

def mark_rate_limited(model_id: str, retry_after: float):
    available_at = time.time() + retry_after
    model_cooldown[model_id] = available_at
    print(f"[rate-limit] {model_id} blocked {retry_after:.0f}s → "
          f"available at {time.strftime('%H:%M:%S', time.localtime(available_at))}")

def parse_retry_after(exc: Exception) -> float:
    try:
        msg = str(exc)
        m = re.search(r'(?:try again in|retry after)\s*([\d.]+)s', msg, re.IGNORECASE)
        if m:
            return float(m.group(1)) + 2
        m = re.search(r'(\d+)m(\d+)s', msg)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2)) + 2
        m = re.search(r'(\d+)m', msg)
        if m:
            return int(m.group(1)) * 60 + 2
    except Exception:
        pass
    return 62.0

def earliest_available_in(models: list[dict]) -> float:
    now = time.time()
    waits = [max(0.0, model_cooldown.get(m["id"], 0) - now) for m in models]
    return min(waits) if waits else 0.0

def format_wait(seconds: float) -> str:
    if seconds < 5:
        return "a few seconds"
    if seconds < 90:
        return f"{math.ceil(seconds)} seconds"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


# ─────────────────────────────────────────────────────────────────────
#  CORE FALLBACK CALLER
# ─────────────────────────────────────────────────────────────────────
def call_with_fallback(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, str]:
    tried: list[str] = []

    for model in models:
        mid = model["id"]

        if not is_available(mid):
            wait = model_cooldown[mid] - time.time()
            tried.append(f"{model['name']} — cooling down ({format_wait(wait)} left)")
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
            print(f"[model] ✓ {mid}")
            return text, mid

        except RateLimitError as e:
            wait = parse_retry_after(e)
            mark_rate_limited(mid, wait)
            tried.append(f"{model['name']} — rate limited (retry in {format_wait(wait)})")
            continue

        except APIStatusError as e:
            mark_rate_limited(mid, 15)
            tried.append(f"{model['name']} — API error {e.status_code}")
            print(f"[api-error] {mid} status={e.status_code}")
            continue

        except Exception as e:
            tried.append(f"{model['name']} — {type(e).__name__}")
            print(f"[error] {mid} → {e}")
            continue

    soonest = earliest_available_in(models)
    raise RuntimeError(f"QUOTA_EXCEEDED|{format_wait(soonest)}|{'||'.join(tried)}")


# ─────────────────────────────────────────────────────────────────────
#  WEB SEARCH  (requests only — 3-layer fallback for reliability)
# ─────────────────────────────────────────────────────────────────────
SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

def _ddg_json(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo Instant Answer JSON API — best for facts & definitions."""
    r = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": "1",
                "no_redirect": "1", "skip_disambig": "1"},
        headers=SEARCH_HEADERS,
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()
    results = []

    if data.get("Answer"):
        results.append({
            "title":   "Direct Answer",
            "url":     data.get("AbstractURL", ""),
            "snippet": str(data["Answer"])[:500],
        })
    if data.get("AbstractText"):
        results.append({
            "title":   data.get("Heading") or "Summary",
            "url":     data.get("AbstractURL", ""),
            "snippet": data["AbstractText"][:500],
        })
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title":   topic["Text"][:80],
                "url":     topic.get("FirstURL", ""),
                "snippet": topic["Text"][:300],
            })
    return results[:max_results]


def _wikipedia(query: str) -> list[dict]:
    """Wikipedia REST summary — excellent for factual/encyclopaedic queries."""
    # Strip question words for cleaner lookup
    clean = re.sub(
        r'\b(what is|who is|what are|tell me about|explain|define|'
        r'latest|current|today|news|right now|recently|about)\b',
        '', query, flags=re.IGNORECASE
    ).strip()
    clean = ' '.join(clean.split()[:6])
    if not clean:
        clean = query

    r = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(clean)}",
        headers=SEARCH_HEADERS,
        timeout=7,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("extract") and len(data["extract"]) > 60:
        return [{
            "title":   data.get("title", "Wikipedia"),
            "url":     data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "snippet": data["extract"][:600],
        }]
    return []


def _ddg_html_fallback(query: str, max_results: int) -> list[dict]:
    """
    Scrape DuckDuckGo HTML search page as a last-resort fallback.
    Extracts result snippets using a simple regex — no HTML parser needed.
    """
    r = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers=SEARCH_HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    html = r.text

    # Extract result snippets between <a class="result__snippet"> tags
    snippets = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>',
        html, re.DOTALL
    )
    results = []
    for url, title, snippet in snippets[:max_results]:
        clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        clean_title   = re.sub(r'<[^>]+>', '', title).strip()
        if clean_snippet:
            results.append({
                "title":   clean_title[:100],
                "url":     url,
                "snippet": clean_snippet[:300],
            })
    return results


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    3-layer search — tries each source in order, merges best results.
    Uses only `requests`. Returns [] only if all three layers fail.
    """
    results: list[dict] = []

    # Layer 1: DDG Instant Answer API
    try:
        results += _ddg_json(query, max_results)
        print(f"[search] DDG JSON → {len(results)} results")
    except Exception as e:
        print(f"[search] DDG JSON failed: {e}")

    # Layer 2: Wikipedia (always try — different source, often great)
    try:
        wiki = _wikipedia(query)
        # Only add if not duplicate
        existing_urls = {r["url"] for r in results}
        for w in wiki:
            if w["url"] not in existing_urls:
                results.append(w)
        print(f"[search] Wikipedia → {len(wiki)} results")
    except Exception as e:
        print(f"[search] Wikipedia failed: {e}")

    # Layer 3: DDG HTML scrape — only if still empty
    if not results:
        try:
            results += _ddg_html_fallback(query, max_results)
            print(f"[search] DDG HTML fallback → {len(results)} results")
        except Exception as e:
            print(f"[search] DDG HTML fallback failed: {e}")

    return results[:max_results]


def format_search_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["[Web Search Results — use these to answer the question accurately]"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('title', '')}\n"
            f"   {r.get('snippet', '')[:350]}\n"
            f"   Source: {r.get('url', '')}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  IMAGE GENERATION DETECTION
#  Catch these requests early and return a clear "I can't do that" reply
# ─────────────────────────────────────────────────────────────────────
IMAGE_GEN_PATTERNS = [
    r'\bgenerate\s+(?:an?\s+)?image\b',
    r'\bcreate\s+(?:an?\s+)?image\b',
    r'\bmake\s+(?:an?\s+)?image\b',
    r'\bdraw\s+(?:me\s+)?(?:an?\s+)?\b',
    r'\bpaint\s+(?:me\s+)?\b',
    r'\bdesign\s+(?:an?\s+)?image\b',
    r'\bgenerate\s+(?:a\s+)?(?:picture|photo|illustration|artwork|graphic|wallpaper|logo)\b',
    r'\bcreate\s+(?:a\s+)?(?:picture|photo|illustration|artwork|graphic|wallpaper|logo)\b',
    r'\bshow\s+me\s+(?:a\s+)?(?:picture|photo|image)\b',
    r'^/imagine\b',
    r'\bimagine\s+(?:a\s+)?(?:picture|photo|image|scene)\b',
]
IMAGE_GEN_RE = re.compile('|'.join(IMAGE_GEN_PATTERNS), re.IGNORECASE)

def is_image_request(text: str) -> bool:
    return bool(IMAGE_GEN_RE.search(text))

IMAGE_DECLINE = (
    "I can't generate or create images — I'm a text-only AI. 🙅\n\n"
    "For image generation, try one of these free tools:\n"
    "• **[Adobe Firefly](https://firefly.adobe.com)** — free, high quality\n"
    "• **[Microsoft Designer](https://designer.microsoft.com)** — free with Microsoft account\n"
    "• **[Ideogram](https://ideogram.ai)** — free tier available\n"
    "• **[Craiyon](https://www.craiyon.com)** — completely free\n\n"
    "I'm great at writing prompts for those tools though — want me to write a detailed prompt for your idea? ✍️"
)


# ─────────────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────────────
SYSTEM = """You are Zippy, a smart AI assistant made by Arul Vethathiri.

## Core identity
- You are a text-only AI. You CANNOT generate, create, or display images. Ever.
- You CAN search the web for current information when needed.
- You were created by Arul Vethathiri, a Class 11 student (2026).

## Tone
- Smart, calm, helpful. Like a knowledgeable friend, not a corporate bot.
- Natural and conversational — not stiff, not over-excited.
- Greetings: short and casual. "Hey!" or "Hi, what's up?" — not a paragraph.

## Response rules
1. ALWAYS answer what the user actually asked. Stay on topic.
2. If a question has a specific answer, give that answer first — don't pad around it.
3. Simple factual questions → 1-2 sentences maximum.
4. Explanations → short bullet points, no long preamble.
5. Code questions → give the code directly with a brief comment if needed.
6. Math → step by step but brief. Show working.
7. Creative writing → complete the full piece without preamble.
8. If web search results are provided, use them. Cite the source naturally in your reply.
9. If asked about something you don't know and no search results are provided, say so honestly.
10. NEVER say: Certainly!, Great question!, Of course!, Absolutely!, As an AI, I'd be happy to.
11. NEVER repeat the question back to the user.
12. NEVER go off-topic or add unsolicited advice.
13. Always end with exactly 1 relevant emoji.

## What you cannot do (be honest about these)
- Generate, draw, or create images of any kind
- Access the internet in real-time yourself (search results are injected by the system)
- Remember previous conversations (each chat is fresh)
- Execute code or run programs"""

THINK_PROMPT = """You are the reasoning engine for Zippy AI.

Given a user message, decide if a web search is needed.
Output ONLY raw JSON — no markdown, no backticks, no explanation outside the JSON.

JSON keys:
- "needs_search": boolean
- "search_query": string (empty string if needs_search is false)
- "reasoning": string (one sentence)

Search IS needed when the question requires:
  current events, breaking news, live prices, stock prices, sports scores,
  weather, "latest" or "recent" news, "today", "this week", "right now",
  new product releases, who currently holds a position/title,
  anything that changes over time and the answer from 2023 might be wrong.

Search is NOT needed for:
  coding help, algorithms, math, science concepts, history (pre-2023),
  grammar, language, creative writing, general advice, definitions,
  how-to guides, programming concepts, explanations of technology,
  opinions, greetings, identity questions, jokes, poems.

Examples:
User: "what is the current price of bitcoin"
{"needs_search": true, "search_query": "bitcoin price today", "reasoning": "Bitcoin price changes by the second."}

User: "write a python function to reverse a string"
{"needs_search": false, "search_query": "", "reasoning": "This is a coding question answerable from training data."}

User: "who won the IPL 2025"
{"needs_search": true, "search_query": "IPL 2025 winner", "reasoning": "Recent sports result not in training data."}

User: "explain recursion"
{"needs_search": false, "search_query": "", "reasoning": "Timeless programming concept."}

Output only raw JSON."""


# ─────────────────────────────────────────────────────────────────────
#  STATIC REPLIES
# ─────────────────────────────────────────────────────────────────────
IDENTITY = {
    "who are you":       "I'm Zippy, a text-based AI assistant made by Arul Vethathiri! 🤖",
    "what are you":      "I'm Zippy, a text-only AI made by Arul Vethathiri. 🤖",
    "who made you":      "I was made by Arul Vethathiri, a Class 11 student. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "who built you":     "I was built by Arul Vethathiri. 👨‍💻",
    "what is your name": "My name is Zippy! 😊",
    "what can you do":   "I can answer questions, help with code, explain concepts, do math, write text, and search the web for current info. I can't generate images though. 💬",
    "are you an ai":     "Yes! I'm Zippy, a text-only AI made by Arul Vethathiri. 🤖",
    "are you human":     "Nope! I'm Zippy — an AI, but a pretty good one at conversation. 😄",
}

SOCIAL = {
    "thanks":     "You're welcome! 😊",
    "thank you":  "Happy to help! 🌟",
    "bye":        "Goodbye! Take care! 👋",
    "goodbye":    "See you later! 👋",
    "good night": "Good night! Sleep well! 🌙",
    "hello":      "Hey! What can I help you with? 👋",
    "hi":         "Hi! What's up? 😊",
    "hey":        "Hey! What do you need? 👋",
}


# ─────────────────────────────────────────────────────────────────────
#  THINKING STEP
# ─────────────────────────────────────────────────────────────────────
def run_thinking(user_input: str) -> dict:
    messages = [
        {"role": "system", "content": THINK_PROMPT},
        {"role": "user",   "content": user_input},
    ]
    try:
        raw, _ = call_with_fallback(
            messages, THINK_MODELS,
            max_tokens=150, temperature=0.1, top_p=0.9,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Sometimes the model wraps with extra text before/after — extract JSON object
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        return json.loads(raw)
    except RuntimeError:
        return {"needs_search": False, "search_query": "", "reasoning": "Thinking skipped (quota)."}
    except Exception as e:
        print(f"[thinking] parse error: {e} | raw was: {raw if 'raw' in dir() else 'N/A'}")
        return {"needs_search": False, "search_query": "", "reasoning": "Thinking skipped (parse error)."}


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
    now = time.time()
    model_status = {}
    for m in CHAT_MODELS:
        cd = model_cooldown.get(m["id"], 0)
        model_status[m["name"]] = (
            "available" if cd <= now
            else f"cooling down — {format_wait(cd - now)} left"
        )
    return {"status": "Zippy backend is running!", "models": model_status}


@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Clean filler sounds at start of message
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message, flags=re.IGNORECASE
    ).strip()

    if not user_input:
        return {"reply": "Didn't catch that — say again? 😊",
                "thinking": "Empty input.", "searched": False, "sources": []}

    text_lower = user_input.lower().strip()

    # 2. Image generation — catch and decline immediately, no model call needed
    if is_image_request(user_input):
        return {
            "reply":      IMAGE_DECLINE,
            "thinking":   "User asked for image generation — Zippy is text-only, declined with alternatives.",
            "searched":   False,
            "sources":    [],
            "model_used": "static",
        }

    # 3. Static identity / social replies
    if text_lower in IDENTITY:
        return {"reply": IDENTITY[text_lower], "thinking": "Identity — hardcoded.",
                "searched": False, "sources": [], "model_used": "static"}
    if text_lower in SOCIAL:
        return {"reply": SOCIAL[text_lower], "thinking": "Social — hardcoded.",
                "searched": False, "sources": [], "model_used": "static"}

    # 4. Thinking step — should we search?
    think        = run_thinking(user_input)
    needs_search = think.get("needs_search", False)
    search_query = think.get("search_query", "").strip()
    reasoning    = think.get("reasoning", "")

    # 5. Web search (3-layer fallback — very reliable)
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search and search_query:
        print(f"[search] querying: {search_query}")
        results = web_search(search_query)
        if results:
            search_context = format_search_context(results)
            searched = True
            search_sources = results[:4]
            print(f"[search] got {len(results)} results")
        else:
            # Search was attempted but returned nothing — tell the model
            search_context = "[Web search was attempted but returned no results. Answer from your training knowledge and note the limitation.]"
            print("[search] no results returned")

    # 6. Build the final prompt — tightly scoped to what was asked
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]

    # Add conversation history (keep last 20 turns)
    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:1000]})

    # Compose the user message with optional search context
    if search_context:
        final_user = (
            f"{search_context}\n\n"
            f"---\n"
            f"Using the search results above, answer this question directly and stay on topic:\n"
            f"{user_input}"
        )
    else:
        final_user = (
            f"Answer this directly and stay on topic:\n{user_input}"
        )

    messages.append({"role": "user", "content": final_user})

    # 7. Call with full model fallback chain
    try:
        reply, model_used = call_with_fallback(
            messages, CHAT_MODELS,
            max_tokens=512, temperature=0.7, top_p=0.9,
        )
        return {
            "reply":      reply,
            "thinking":   reasoning,
            "searched":   searched,
            "sources":    search_sources,
            "model_used": model_used,
        }

    except RuntimeError as exc:
        parts     = str(exc).split("|", 2)
        wait_str  = parts[1] if len(parts) == 3 else "a few minutes"
        tried_raw = parts[2] if len(parts) == 3 else str(exc)
        tried_lines = "\n".join(f"• {l}" for l in tried_raw.split("||") if l.strip())

        quota_reply = (
            f"⚠️ **All AI models are rate-limited right now.**\n\n"
            f"Please wait at least **{wait_str}** and try again.\n\n"
            f"*Groq free tier: 30 requests/min · 1,000 requests/day per model. "
            f"All 5 models are temporarily exhausted.*\n\n"
            f"**Models tried:**\n{tried_lines}"
        )
        return {
            "reply":      quota_reply,
            "thinking":   "All models rate-limited — quota exceeded message returned.",
            "searched":   False,
            "sources":    [],
            "model_used": "none",
        }
