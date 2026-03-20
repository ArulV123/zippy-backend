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
#  MODELS
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
            continue
        except Exception as e:
            tried.append(f"{model['name']} — {type(e).__name__}")
            print(f"[error] {mid} → {e}")
            continue

    soonest = earliest_available_in(models)
    raise RuntimeError(f"QUOTA_EXCEEDED|{format_wait(soonest)}|{'||'.join(tried)}")


# ─────────────────────────────────────────────────────────────────────
#  WEB SEARCH  —  4 independent sources, all using `requests` only
#
#  Source 1: CoinGecko API        → crypto prices (dedicated, reliable)
#  Source 2: DuckDuckGo HTML      → general web search (PRIMARY for all queries)
#  Source 3: Wikipedia REST       → encyclopaedic / factual
#  Source 4: wttr.in JSON         → weather queries
#
#  The function tries ALL applicable sources and merges results.
#  It never silently fails — if results exist, they're returned.
# ─────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Crypto detection ──────────────────────────────────────────────────
CRYPTO_MAP = {
    "bitcoin": "bitcoin",  "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "solana": "solana",     "sol": "solana",
    "bnb": "binancecoin",   "binance coin": "binancecoin",
    "xrp": "ripple",        "ripple": "ripple",
    "cardano": "cardano",   "ada": "cardano",
    "litecoin": "litecoin", "ltc": "litecoin",
    "polkadot": "polkadot", "dot": "polkadot",
    "shiba inu": "shiba-inu", "shib": "shiba-inu",
}

def _detect_crypto(query: str) -> list[str]:
    q = query.lower()
    return list({cg_id for keyword, cg_id in CRYPTO_MAP.items() if keyword in q})

def _search_crypto(coin_ids: list[str]) -> list[dict]:
    """CoinGecko free API — no key needed, very reliable."""
    try:
        ids_str = ",".join(coin_ids)
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ids_str,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
            },
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for coin_id, info in data.items():
            price     = info.get("usd", "N/A")
            change    = info.get("usd_24h_change", None)
            mktcap    = info.get("usd_market_cap", None)
            change_str = f", 24h change: {change:.2f}%" if change is not None else ""
            cap_str    = f", market cap: ${mktcap:,.0f}" if mktcap else ""
            results.append({
                "title":   f"{coin_id.capitalize()} Price (Live)",
                "url":     f"https://www.coingecko.com/en/coins/{coin_id}",
                "snippet": f"Current {coin_id} price: ${price:,.2f} USD{change_str}{cap_str}.",
            })
        print(f"[search] CoinGecko → {len(results)} results")
        return results
    except Exception as e:
        print(f"[search] CoinGecko failed: {e}")
        return []

# ── Weather detection ─────────────────────────────────────────────────
WEATHER_WORDS = {"weather", "temperature", "forecast", "humidity", "rain", "sunny", "hot", "cold", "climate today"}

def _detect_weather_city(query: str) -> str | None:
    q = query.lower()
    if not any(w in q for w in WEATHER_WORDS):
        return None
    # Extract city: "weather in Chennai" → "Chennai"
    m = re.search(r'weather\s+(?:in|at|for|of)?\s+([A-Za-z\s]{2,30})', query, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # "Chennai weather" → "Chennai"
    m = re.search(r'([A-Za-z\s]{2,20})\s+weather', query, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def _search_weather(city: str) -> list[dict]:
    """wttr.in free weather API — no key needed."""
    try:
        r = requests.get(
            f"https://wttr.in/{requests.utils.quote(city)}",
            params={"format": "j1"},
            headers={"User-Agent": "ZippyAI/1.0"},
            timeout=7,
        )
        r.raise_for_status()
        data = r.json()
        cur = data["current_condition"][0]
        desc     = cur.get("weatherDesc", [{}])[0].get("value", "N/A")
        temp_c   = cur.get("temp_C", "N/A")
        temp_f   = cur.get("temp_F", "N/A")
        feels_c  = cur.get("FeelsLikeC", "N/A")
        humidity = cur.get("humidity", "N/A")
        wind     = cur.get("windspeedKmph", "N/A")
        snippet  = (
            f"Current weather in {city}: {desc}. "
            f"Temperature: {temp_c}°C ({temp_f}°F), feels like {feels_c}°C. "
            f"Humidity: {humidity}%, Wind: {wind} km/h."
        )
        print(f"[search] wttr.in → weather for {city}")
        return [{"title": f"Weather in {city}", "url": f"https://wttr.in/{city}", "snippet": snippet}]
    except Exception as e:
        print(f"[search] wttr.in failed: {e}")
        return []

# ── DuckDuckGo HTML search (PRIMARY general search) ───────────────────
def _search_ddg_html(query: str, max_results: int = 5) -> list[dict]:
    """
    Scrapes DuckDuckGo's HTML results page.
    This is the most reliable general-purpose free search — real web results.
    """
    try:
        r = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": "", "kl": "us-en"},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        html = r.text

        results = []

        # DDG HTML structure:
        # <h2 class="result__title"><a class="result__a" href="...">TITLE</a></h2>
        # <a class="result__snippet" ...>SNIPPET</a>

        # Split into individual result blocks
        blocks = re.split(r'<div class="result results_links', html)

        for block in blocks[1:]:  # skip first (header)
            if len(results) >= max_results:
                break

            # Extract title
            title_m = re.search(
                r'class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL
            )
            # Extract URL from the uddg redirect param
            url_m = re.search(
                r'href="//duckduckgo\.com/l/\?uddg=([^"&]+)', block
            )
            if not url_m:
                url_m = re.search(r'class="result__a"\s+href="([^"]+)"', block)

            # Extract snippet
            snip_m = re.search(
                r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL
            )

            if title_m and snip_m:
                title   = re.sub(r'<[^>]+>', '', title_m.group(1)).strip()
                snippet = re.sub(r'<[^>]+>', '', snip_m.group(1)).strip()
                snippet = re.sub(r'\s+', ' ', snippet)

                url = ""
                if url_m:
                    raw_url = url_m.group(1)
                    try:
                        from urllib.parse import unquote
                        url = unquote(raw_url)
                    except Exception:
                        url = raw_url

                if title and snippet and len(snippet) > 20:
                    results.append({
                        "title":   title[:120],
                        "url":     url,
                        "snippet": snippet[:400],
                    })

        print(f"[search] DDG HTML → {len(results)} results")
        return results

    except Exception as e:
        print(f"[search] DDG HTML failed: {e}")
        return []

# ── Wikipedia ─────────────────────────────────────────────────────────
def _search_wikipedia(query: str) -> list[dict]:
    try:
        clean = re.sub(
            r'\b(what is|who is|what are|tell me|explain|define|'
            r'latest|current|today|price|news|right now|recently|about|the)\b',
            '', query, flags=re.IGNORECASE
        ).strip()
        clean = ' '.join(clean.split()[:6])
        if not clean or len(clean) < 3:
            return []

        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(clean)}",
            headers={"User-Agent": "ZippyAI/1.0"},
            timeout=6,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("extract") and len(data["extract"]) > 80:
            return [{
                "title":   data.get("title", "Wikipedia"),
                "url":     data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "snippet": data["extract"][:600],
            }]
        return []
    except Exception as e:
        print(f"[search] Wikipedia failed: {e}")
        return []


# ── Master search function ─────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Runs ALL applicable search sources simultaneously (by query type),
    merges results, deduplicates by URL.
    Always returns whatever data was found — never fails silently.
    """
    results: list[dict] = []
    seen_urls: set[str] = set()

    def add(items: list[dict]):
        for item in items:
            url = item.get("url", "")
            key = url or item.get("title", "")
            if key and key not in seen_urls:
                seen_urls.add(key)
                results.append(item)

    # 1. Crypto prices — dedicated API, very accurate
    coins = _detect_crypto(query)
    if coins:
        add(_search_crypto(coins))

    # 2. Weather — dedicated API
    city = _detect_weather_city(query)
    if city:
        add(_search_weather(city))

    # 3. DuckDuckGo HTML — real web search, best for current events/news
    add(_search_ddg_html(query, max_results))

    # 4. Wikipedia — good complement for factual context
    add(_search_wikipedia(query))

    print(f"[search] total merged results: {len(results)}")
    return results[:max_results]


def format_search_context(query: str, results: list[dict]) -> str:
    """Format search results clearly so the AI can use them accurately."""
    if not results:
        return (
            f"[SEARCH ATTEMPTED for: '{query}' — no results returned. "
            f"Be honest that you don't have current data.]"
        )

    lines = [
        f"[LIVE WEB SEARCH RESULTS for: '{query}']",
        f"[Retrieved: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}]",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"Result {i}: {r.get('title', 'No title')}")
        lines.append(f"  Info: {r.get('snippet', '').strip()}")
        if r.get("url"):
            lines.append(f"  URL: {r['url']}")
        lines.append("")

    lines.append("[END OF SEARCH RESULTS]")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  IMAGE DETECTION + DECLINE
# ─────────────────────────────────────────────────────────────────────
IMAGE_GEN_RE = re.compile(
    r'\b(generate|create|make|draw|paint|design|produce|render)\b.{0,20}'
    r'\b(image|picture|photo|illustration|artwork|graphic|wallpaper|logo|poster|banner|sketch|portrait)\b'
    r'|^/imagine\b',
    re.IGNORECASE
)

def is_image_request(text: str) -> bool:
    return bool(IMAGE_GEN_RE.search(text))

IMAGE_DECLINE = (
    "I'm a text-only AI — I can't generate or create images. 🙅\n\n"
    "Try these free image generators instead:\n"
    "• **[Adobe Firefly](https://firefly.adobe.com)** — high quality, free\n"
    "• **[Microsoft Designer](https://designer.microsoft.com)** — free with Microsoft account\n"
    "• **[Ideogram](https://ideogram.ai)** — great for text in images\n"
    "• **[Craiyon](https://www.craiyon.com)** — completely free, no account needed\n\n"
    "Want me to write a detailed prompt for any of those tools? ✍️"
)


# ─────────────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────────────
SYSTEM = """You are Zippy, a smart AI assistant made by Arul Vethathiri.

## WHO YOU ARE
- Text-only AI. You CANNOT generate images. Period.
- You CAN search the web — live search results will be injected into your context when needed.
- Made by Arul Vethathiri, Class 11 student (2026).

## HOW TO USE SEARCH RESULTS
When you see [LIVE WEB SEARCH RESULTS] in the context:
- USE THAT DATA. It is real, current, and accurate.
- State the information directly. Do NOT say "I don't have real-time access."
- Quote numbers/prices exactly as given in the results.
- Mention where the data came from (e.g. "According to CoinGecko..." or "Based on current data...").
- If the search returned no results, say honestly: "I searched but couldn't find current data on this."

## TONE
- Smart, calm, friendly — like a knowledgeable friend.
- Casual but accurate. Not corporate, not over-excited.
- Short greetings. One sentence max.

## RESPONSE RULES
1. Answer EXACTLY what was asked. Stay on topic. Nothing extra.
2. Factual/numeric answers → lead with the number/fact immediately.
3. Simple questions → 1-3 sentences max.
4. Explanations → brief bullet points, no long intro.
5. Code → give the code directly, minimal comments.
6. Math → show steps, keep it brief.
7. Creative tasks → complete the full piece, no preamble.
8. NEVER say: "I don't have real-time access", "My training data", "As an AI",
   "Certainly!", "Great question!", "Of course!", "I'd be happy to".
9. NEVER repeat the question.
10. NEVER pad answers or add unsolicited advice.
11. End every response with exactly 1 relevant emoji.

## WHAT YOU CANNOT DO (say this clearly if asked)
- Generate, draw, or produce images
- Remember previous sessions
- Run/execute code"""

# The thinking prompt is strict and rule-based — less room for the model to hallucinate
THINK_PROMPT = """You decide if a user question needs a live web search.

Output ONLY a JSON object. No markdown. No extra text. Just JSON.

Rules:
- needs_search = true  → question requires data that changes over time
- needs_search = false → question can be answered from general knowledge alone

ALWAYS search for:
- Any price (crypto, stock, gold, currency, commodity)
- Any score, result, or standing (sports, elections, awards)
- Any "latest", "recent", "current", "today", "now", "this week", "2024", "2025" question
- Weather or temperature for a location
- Who currently holds a job/title/position
- New product releases or announcements
- Any news event that could have changed since 2023

NEVER search for:
- How to code something (algorithms, syntax, concepts)
- Math problems
- Science concepts (physics, chemistry, biology theory)
- History before 2023
- Grammar, language, translation
- Definitions of stable concepts
- Creative writing, poems, stories
- Opinions or advice
- Greetings or small talk

JSON format:
{
  "needs_search": true or false,
  "search_query": "optimized search query string or empty string",
  "reasoning": "one sentence"
}

Good search queries: short, specific, optimised for search engines.
Example: user asks "what is bitcoin price" → search_query: "bitcoin price USD today"
Example: user asks "who won IPL 2025" → search_query: "IPL 2025 winner"
Example: user asks "weather in Chennai" → search_query: "Chennai weather today"

Output only the JSON object."""


# ─────────────────────────────────────────────────────────────────────
#  STATIC REPLIES
# ─────────────────────────────────────────────────────────────────────
IDENTITY = {
    "who are you":       "I'm Zippy, a text-based AI made by Arul Vethathiri! 🤖",
    "what are you":      "I'm Zippy — a text-only AI assistant made by Arul Vethathiri. 🤖",
    "who made you":      "Arul Vethathiri, a Class 11 student. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "who built you":     "Built by Arul Vethathiri. 👨‍💻",
    "what is your name": "I'm Zippy! 😊",
    "what can you do":   (
        "I can answer questions, help with code, explain concepts, do math, "
        "write anything, and search the web for live data like prices and news. "
        "I can't generate images though. 💬"
    ),
    "are you an ai":     "Yes — Zippy AI, made by Arul Vethathiri. 🤖",
    "are you human":     "Nope, I'm Zippy — an AI. But a pretty good one! 😄",
}

SOCIAL = {
    "thanks":     "You're welcome! 😊",
    "thank you":  "Happy to help! 🌟",
    "bye":        "Goodbye! 👋",
    "goodbye":    "See you! 👋",
    "good night": "Good night! 🌙",
    "hello":      "Hey! What do you need? 👋",
    "hi":         "Hi! What's up? 😊",
    "hey":        "Hey! 👋",
    "ok":         "Sure, let me know if you need anything. 👍",
    "okay":       "Sure, let me know if you need anything. 👍",
}


# ─────────────────────────────────────────────────────────────────────
#  THINKING STEP
# ─────────────────────────────────────────────────────────────────────
def run_thinking(user_input: str) -> dict:
    """Use a small model to decide search strategy. Very low temperature = consistent JSON."""
    messages = [
        {"role": "system", "content": THINK_PROMPT},
        {"role": "user",   "content": user_input},
    ]
    try:
        raw, _ = call_with_fallback(
            messages, THINK_MODELS,
            max_tokens=120, temperature=0.05, top_p=0.9,
        )
        # Strip any markdown fences the model might add
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Extract first JSON object found (in case model adds commentary)
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if m:
            raw = m.group(0)
        result = json.loads(raw)
        # Validate expected keys exist
        if "needs_search" not in result:
            raise ValueError("Missing needs_search key")
        return result
    except RuntimeError:
        return {"needs_search": False, "search_query": "", "reasoning": "Thinking skipped (quota)."}
    except Exception as e:
        print(f"[thinking] failed: {e}")
        # Fallback: use keyword-based decision
        return _keyword_search_decision(user_input)


# Keyword-based fallback if the thinking model fails
SEARCH_KEYWORDS = [
    "price", "cost", "worth", "value", "rate", "exchange",
    "bitcoin", "crypto", "stock", "share", "market",
    "weather", "temperature", "forecast",
    "score", "result", "winner", "won", "beat",
    "latest", "recent", "current", "today", "now", "live",
    "this week", "this month", "2024", "2025", "2026",
    "news", "update", "announce", "release", "launch",
    "who is", "who's the current", "who won",
    "ipl", "cricket", "football", "nba", "nfl", "match",
]

def _keyword_search_decision(user_input: str) -> dict:
    lower = user_input.lower()
    for kw in SEARCH_KEYWORDS:
        if kw in lower:
            return {
                "needs_search": True,
                "search_query": user_input,
                "reasoning": f"Keyword '{kw}' detected — search needed.",
            }
    return {
        "needs_search": False,
        "search_query": "",
        "reasoning": "No time-sensitive keywords detected.",
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
    now = time.time()
    status = {}
    for m in CHAT_MODELS:
        cd = model_cooldown.get(m["id"], 0)
        status[m["name"]] = (
            "available" if cd <= now
            else f"cooling — {format_wait(cd - now)} left"
        )
    return {"status": "Zippy backend is running!", "models": status}


@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Strip filler sounds
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message, flags=re.IGNORECASE
    ).strip()

    if not user_input:
        return {"reply": "Didn't catch that — say again? 😊",
                "thinking": "Empty input.", "searched": False, "sources": []}

    text_lower = user_input.lower().strip()

    # 2. Image generation → decline immediately
    if is_image_request(user_input):
        return {
            "reply":      IMAGE_DECLINE,
            "thinking":   "Image generation request detected — Zippy is text-only.",
            "searched":   False,
            "sources":    [],
            "model_used": "static",
        }

    # 3. Static replies
    if text_lower in IDENTITY:
        return {"reply": IDENTITY[text_lower], "thinking": "Identity — hardcoded.",
                "searched": False, "sources": [], "model_used": "static"}
    if text_lower in SOCIAL:
        return {"reply": SOCIAL[text_lower], "thinking": "Social — hardcoded.",
                "searched": False, "sources": [], "model_used": "static"}

    # 4. Thinking step — decide search strategy
    think        = run_thinking(user_input)
    needs_search = think.get("needs_search", False)
    search_query = think.get("search_query", "").strip()
    reasoning    = think.get("reasoning", "")

    # 5. Web search — runs ALL applicable sources, merges results
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search and search_query:
        print(f"[search] → '{search_query}'")
        results = web_search(search_query)
        search_context = format_search_context(search_query, results)
        if results:
            searched = True
            search_sources = results
            print(f"[search] ✓ {len(results)} results ready for model")
        else:
            print("[search] ✗ all sources returned empty")

    # 6. Build the final messages for the AI
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]

    # Add conversation history
    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            messages.append({
                "role":    h["role"],
                "content": str(h["content"])[:1200],
            })

    # Compose user message — search context injected BEFORE the question
    if search_context:
        final_user = (
            f"{search_context}\n\n"
            f"Using ONLY the search results above, answer this question directly "
            f"and accurately. Do NOT say you lack real-time access:\n\n"
            f"{user_input}"
        )
    else:
        final_user = user_input

    messages.append({"role": "user", "content": final_user})

    # 7. Call AI with model fallback chain
    try:
        reply, model_used = call_with_fallback(
            messages, CHAT_MODELS,
            max_tokens=512, temperature=0.65, top_p=0.9,
        )
        return {
            "reply":      reply,
            "thinking":   reasoning,
            "searched":   searched,
            "sources":    search_sources,
            "model_used": model_used,
        }

    except RuntimeError as exc:
        parts       = str(exc).split("|", 2)
        wait_str    = parts[1] if len(parts) == 3 else "a few minutes"
        tried_raw   = parts[2] if len(parts) == 3 else str(exc)
        tried_lines = "\n".join(f"• {l}" for l in tried_raw.split("||") if l.strip())

        return {
            "reply": (
                f"⚠️ **All AI models are rate-limited right now.**\n\n"
                f"Please wait at least **{wait_str}** and try again.\n\n"
                f"*Groq free tier: 30 req/min · 1,000 req/day per model. "
                f"All 5 models temporarily exhausted.*\n\n"
                f"**Models tried:**\n{tried_lines}"
            ),
            "thinking":   "All models rate-limited.",
            "searched":   False,
            "sources":    [],
            "model_used": "none",
        }
