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
import threading

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
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B"},
    {"id": "llama-3.1-8b-instant",    "name": "Llama 3.1 8B"},
    {"id": "gemma2-9b-it",            "name": "Gemma 2 9B"},
    {"id": "llama3-70b-8192",         "name": "Llama 3 70B"},
    {"id": "llama3-8b-8192",          "name": "Llama 3 8B"},
]

# Fastest models first — thinking step must be quick
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
          f"{time.strftime('%H:%M:%S', time.localtime(available_at))}")

def parse_retry_after(exc: Exception) -> float:
    try:
        msg = str(exc)
        m = re.search(r'(?:try again in|retry after)\s*([\d.]+)s', msg, re.IGNORECASE)
        if m: return float(m.group(1)) + 2
        m = re.search(r'(\d+)m(\d+)s', msg)
        if m: return int(m.group(1)) * 60 + int(m.group(2)) + 2
        m = re.search(r'(\d+)m', msg)
        if m: return int(m.group(1)) * 60 + 2
    except Exception:
        pass
    return 62.0

def earliest_available_in(models: list[dict]) -> float:
    now = time.time()
    return min((max(0.0, model_cooldown.get(m["id"], 0) - now) for m in models), default=0.0)

def format_wait(seconds: float) -> str:
    if seconds < 5:  return "a few seconds"
    if seconds < 90: return f"{math.ceil(seconds)} seconds"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


# ─────────────────────────────────────────────────────────────────────
#  CORE FALLBACK CALLER
# ─────────────────────────────────────────────────────────────────────
def call_with_fallback(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 600,
    temperature: float = 0.65,
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
        except APIStatusError as e:
            mark_rate_limited(mid, 15)
            tried.append(f"{model['name']} — API error {e.status_code}")
        except Exception as e:
            tried.append(f"{model['name']} — {type(e).__name__}")
            print(f"[error] {mid} → {e}")
    soonest = earliest_available_in(models)
    raise RuntimeError(f"QUOTA_EXCEEDED|{format_wait(soonest)}|{'||'.join(tried)}")


# ─────────────────────────────────────────────────────────────────────
#  THINKING STEP  —  model decides, with a hard timeout + rule fallback
#
#  The AI model is ALWAYS asked first.
#  Rules only activate if the model times out or returns broken JSON.
#  This ensures the AI "thinks for itself" on every healthy request.
# ─────────────────────────────────────────────────────────────────────

THINK_PROMPT = """You are the thinking engine for Zippy AI.

A user sent a message. Your ONLY job is to decide:
1. Does answering this need live/current information from the web?
2. If yes, what is the best short search query to use?

Respond with ONLY a raw JSON object. No markdown, no extra text, just JSON.

{
  "needs_search": true or false,
  "search_query": "search query here, or empty string if not needed",
  "reasoning": "one sentence explaining your decision"
}

Think carefully:

NEEDS SEARCH — YES, when the question is about:
- Current prices of anything: crypto, stocks, gold, oil, forex, petrol
- Live weather or forecast for a location
- Recent or ongoing events, news, match results, scores
- Anything described with: today, now, current, latest, recent, live, 2024, 2025, 2026
- Who holds a position RIGHT NOW (president, CEO, champion, etc.)
- New product releases, software versions, app updates
- Country facts like capital, population, currency (these can change)

NEEDS SEARCH — NO, when the question is about:
- Coding concepts, algorithms, syntax, how to write something in code
- Maths problems or calculations
- Science theory (gravity, photosynthesis, etc.)
- History before 2023
- Definitions or explanations of stable concepts
- Creative writing: poems, stories, essays
- Grammar, translation, language
- General advice or opinions
- Greetings or small talk

For search_query: make it short, specific, search-engine optimised.
Example: "bitcoin price today", "weather in Chennai", "IPL 2025 winner"

Output only the JSON. Nothing else."""


# Minimal keyword rules — ONLY used when the model call fails completely
_ALWAYS_SEARCH = re.compile(
    r'\b(price|cost|weather|temperature|forecast|score|result|winner|'
    r'bitcoin|ethereum|crypto|stock|gold|silver|oil|petrol|forex|'
    r'today|right now|currently|latest|breaking news|live score)\b',
    re.IGNORECASE,
)

def _rule_fallback(user_input: str) -> dict:
    """
    Emergency keyword fallback — only runs when the model times out or crashes.
    Not the primary decision maker — just a safety net.
    """
    if _ALWAYS_SEARCH.search(user_input):
        q = user_input.strip().rstrip("?").strip()
        return {
            "needs_search": True,
            "search_query": q,
            "reasoning": "Rule fallback: time-sensitive keyword detected (model timed out).",
        }
    return {
        "needs_search": False,
        "search_query": "",
        "reasoning": "Rule fallback: no time-sensitive keywords (model timed out).",
    }


def run_thinking(user_input: str, timeout_seconds: float = 5.0) -> dict:
    """
    Ask the AI model to decide if search is needed.

    Uses a background thread with a hard deadline so a cold/slow model
    never blocks the main request. If the model responds in time → use it.
    If it times out or returns bad JSON → silently use rule fallback.

    This way:
    - First request (cold start): model may time out → rules handle it instantly
    - All subsequent requests: model warms up and handles it correctly
    - The AI's own judgment is used on every healthy request
    """
    result_holder: list[dict] = []
    error_holder:  list[str]  = []

    def _call():
        think_messages = [
            {"role": "system", "content": THINK_PROMPT},
            {"role": "user",   "content": user_input},
        ]
        try:
            raw, used_model = call_with_fallback(
                think_messages,
                THINK_MODELS,
                max_tokens=120,
                temperature=0.0,   # deterministic — we want consistent JSON
                top_p=1.0,
            )
            # Strip markdown fences if model adds them despite instructions
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            # Extract first valid JSON object
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                raw = m.group(0)
            parsed = json.loads(raw)
            if "needs_search" not in parsed:
                raise ValueError("Missing needs_search key")
            parsed["_model"] = used_model
            result_holder.append(parsed)
            print(f"[thinking] ✓ model={used_model} needs_search={parsed['needs_search']} "
                  f"query='{parsed.get('search_query', '')}'")
        except RuntimeError as e:
            # All models rate-limited — just record it
            error_holder.append(f"quota: {e}")
        except Exception as e:
            error_holder.append(f"{type(e).__name__}: {e}")

    # Run model call in a thread with a deadline
    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if result_holder:
        # Model responded in time — use its judgment
        return result_holder[0]

    # Model timed out or failed — log it and use rules
    if thread.is_alive():
        print(f"[thinking] ⏱ model timed out after {timeout_seconds}s → using rule fallback")
    else:
        print(f"[thinking] ✗ model failed ({error_holder}) → using rule fallback")

    return _rule_fallback(user_input)


# ─────────────────────────────────────────────────────────────────────
#  SEARCH SOURCES  (all free, no API keys, no HTML scraping)
# ─────────────────────────────────────────────────────────────────────
BASE_HEADERS = {"User-Agent": "ZippyAI/2.0 (educational project)"}

# ── Wikipedia: search API + full article extract ──────────────────────
def _wiki_search_title(query: str) -> str | None:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": 1,
                "format": "json", "origin": "*",
            },
            headers=BASE_HEADERS, timeout=7,
        )
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except Exception as e:
        print(f"[wiki-title] {e}")
    return None

def _wiki_fetch_extract(title: str, chars: int = 3000) -> str:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts",
                "exintro": False, "explaintext": True,
                "titles": title, "format": "json",
                "origin": "*", "exchars": chars,
            },
            headers=BASE_HEADERS, timeout=8,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract and len(extract) > 50:
                return extract
    except Exception as e:
        print(f"[wiki-extract] {e}")
    return ""

def search_wikipedia(query: str) -> dict | None:
    title = _wiki_search_title(query)
    if not title:
        return None
    content = _wiki_fetch_extract(title, chars=3000)
    if not content:
        return None
    url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"
    print(f"[wiki] ✓ '{title}' ({len(content)} chars)")
    return {"source": "Wikipedia", "title": title, "url": url, "content": content}


# ── CoinGecko: live crypto prices ─────────────────────────────────────
CRYPTO_MAP = {
    "bitcoin": "bitcoin",      "btc":  "bitcoin",
    "ethereum": "ethereum",    "eth":  "ethereum",
    "dogecoin": "dogecoin",    "doge": "dogecoin",
    "solana": "solana",        "sol":  "solana",
    "bnb": "binancecoin",      "binance": "binancecoin",
    "xrp": "ripple",           "ripple": "ripple",
    "cardano": "cardano",      "ada":  "cardano",
    "litecoin": "litecoin",    "ltc":  "litecoin",
    "polkadot": "polkadot",    "dot":  "polkadot",
    "shib": "shiba-inu",       "shiba": "shiba-inu",
    "tron": "tron",            "trx":  "tron",
    "avax": "avalanche-2",     "avalanche": "avalanche-2",
    "matic": "matic-network",  "polygon": "matic-network",
}

def _detect_crypto(q: str) -> list[str]:
    lower = q.lower()
    return list({cg_id for kw, cg_id in CRYPTO_MAP.items() if kw in lower})

def search_crypto(query: str) -> dict | None:
    coins = _detect_crypto(query)
    if not coins:
        return None
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ",".join(coins),
                "vs_currencies": "usd,inr",
                "include_24hr_change": "true",
                "include_market_cap":  "true",
                "include_24hr_vol":    "true",
            },
            headers=BASE_HEADERS, timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        lines = []
        for coin_id, info in data.items():
            usd  = info.get("usd", "N/A")
            inr  = info.get("inr", "N/A")
            chg  = info.get("usd_24h_change", None)
            mcap = info.get("usd_market_cap", None)
            vol  = info.get("usd_24h_vol",    None)
            lines.append(
                f"{coin_id.capitalize()}:\n"
                f"  Price:      ${usd:,.2f} USD  /  ₹{inr:,.2f} INR\n"
                f"  24h Change: {f'{chg:+.2f}%' if chg is not None else 'N/A'}\n"
                f"  Market Cap: {f'${mcap:,.0f}' if mcap else 'N/A'}\n"
                f"  24h Volume: {f'${vol:,.0f}'  if vol  else 'N/A'}"
            )
        print(f"[crypto] ✓ {coins}")
        return {
            "source":  "CoinGecko (live)",
            "title":   "Live Cryptocurrency Prices",
            "url":     "https://www.coingecko.com",
            "content": "\n\n".join(lines),
        }
    except Exception as e:
        print(f"[crypto] failed: {e}")
        return None


# ── wttr.in: live weather ─────────────────────────────────────────────
def _detect_city(query: str) -> str | None:
    q = query.lower()
    if not any(w in q for w in {"weather", "temperature", "forecast", "humidity", "rain", "wind"}):
        return None
    m = re.search(r'weather\s+(?:in|at|for|of)?\s*([A-Za-z][A-Za-z\s]{1,24})', query, re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'([A-Za-z][A-Za-z\s]{1,24})\s+weather', query, re.IGNORECASE)
    if m: return m.group(1).strip()
    return None

def search_weather(query: str) -> dict | None:
    city = _detect_city(query)
    if not city:
        return None
    try:
        r = requests.get(
            f"https://wttr.in/{requests.utils.quote(city)}",
            params={"format": "j1"},
            headers={"User-Agent": "ZippyAI/2.0"},
            timeout=8,
        )
        r.raise_for_status()
        d   = r.json()
        cur = d["current_condition"][0]
        desc     = cur.get("weatherDesc",   [{}])[0].get("value", "N/A")
        temp_c   = cur.get("temp_C",        "N/A")
        temp_f   = cur.get("temp_F",        "N/A")
        feels    = cur.get("FeelsLikeC",    "N/A")
        humidity = cur.get("humidity",      "N/A")
        wind     = cur.get("windspeedKmph", "N/A")
        uv       = cur.get("uvIndex",       "N/A")
        forecast = []
        for day in d.get("weather", [])[:3]:
            date  = day.get("date", "")
            maxc  = day.get("maxtempC", "N/A")
            minc  = day.get("mintempC", "N/A")
            desc2 = (day.get("hourly") or [{}])[4].get("weatherDesc", [{}])[0].get("value", "N/A")
            forecast.append(f"  {date}: {desc2}, {minc}°C – {maxc}°C")
        content = (
            f"Current weather in {city}:\n"
            f"  Condition:   {desc}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  Feels like:  {feels}°C\n"
            f"  Humidity:    {humidity}%\n"
            f"  Wind:        {wind} km/h\n"
            f"  UV Index:    {uv}\n\n"
            f"3-Day Forecast:\n" + "\n".join(forecast)
        )
        print(f"[weather] ✓ {city}")
        return {
            "source":  "wttr.in (live)",
            "title":   f"Weather in {city}",
            "url":     f"https://wttr.in/{city}",
            "content": content,
        }
    except Exception as e:
        print(f"[weather] failed: {e}")
        return None


# ── REST Countries API ────────────────────────────────────────────────
def _detect_country(query: str) -> str | None:
    if not any(w in query.lower() for w in
               {"capital", "population", "currency", "language", "continent", "area", "country", "nation"}):
        return None
    m = re.search(
        r'(?:of|in|about|for|capital of|population of|currency of|area of)\s+'
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', query
    )
    if m: return m.group(1).strip()
    m = re.search(r'\b([A-Z][a-z]{3,})\b', query)
    if m: return m.group(1).strip()
    return None

def search_country(query: str) -> dict | None:
    name = _detect_country(query)
    if not name:
        return None
    try:
        r = requests.get(
            f"https://restcountries.com/v3.1/name/{requests.utils.quote(name)}",
            headers=BASE_HEADERS, timeout=7,
        )
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict):
            return None
        c = data[0]
        common    = c.get("name", {}).get("common", name)
        capital   = ", ".join(c.get("capital", ["N/A"]))
        pop       = c.get("population", "N/A")
        region    = c.get("region", "N/A")
        subregion = c.get("subregion", "N/A")
        area      = c.get("area", "N/A")
        currencies = ", ".join(
            f"{v.get('name','')} ({v.get('symbol','')})"
            for v in c.get("currencies", {}).values()
        ) or "N/A"
        languages = ", ".join(c.get("languages", {}).values()) or "N/A"
        content = (
            f"Country: {common}\n"
            f"  Capital:    {capital}\n"
            f"  Population: {pop:,}\n"
            f"  Region:     {region} ({subregion})\n"
            f"  Area:       {area} km²\n"
            f"  Currencies: {currencies}\n"
            f"  Languages:  {languages}"
        )
        print(f"[country] ✓ {common}")
        return {
            "source":  "REST Countries",
            "title":   f"Country: {common}",
            "url":     "https://restcountries.com",
            "content": content,
        }
    except Exception as e:
        print(f"[country] failed: {e}")
        return None


# ── Master search runner ──────────────────────────────────────────────
def run_search(query: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()

    def add(item: dict | None):
        if not item:
            return
        key = item.get("url") or item.get("title", "")
        if key not in seen:
            seen.add(key)
            results.append(item)

    add(search_crypto(query))
    add(search_weather(query))
    add(search_country(query))
    add(search_wikipedia(query))

    print(f"[search] total sources: {len(results)}")
    return results


def build_search_context(query: str, results: list[dict]) -> str:
    now_str = time.strftime("%d %b %Y %H:%M UTC", time.gmtime())
    if not results:
        return (
            "<search_results>\n"
            f"SEARCHED FOR: {query}\n"
            f"RETRIEVED AT: {now_str}\n"
            "RESULT: No data found from any source.\n"
            "INSTRUCTION: Tell the user you searched but found no current data. "
            "Do NOT say 'I don't have real-time access'.\n"
            "</search_results>"
        )

    lines = [
        "<search_results>",
        f"SEARCHED FOR: {query}",
        f"RETRIEVED AT: {now_str}",
        f"SOURCES FOUND: {len(results)}",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"--- SOURCE {i}: {r['source']} ---",
            f"Title: {r['title']}",
            f"URL:   {r['url']}",
            "",
            r["content"],
            "",
        ]
    lines += [
        "--- END OF SEARCH RESULTS ---",
        "",
        "INSTRUCTIONS FOR AI:",
        "- This data is REAL and was fetched live moments ago.",
        "- Use the exact numbers and facts above in your answer.",
        "- NEVER say 'I don't have real-time access' or 'my training data'.",
        "- NEVER say 'I cannot provide current prices' — the data is above.",
        "- Cite the source naturally: 'According to CoinGecko...' etc.",
        "- If the data doesn't answer the question, say you searched but found nothing relevant.",
        "</search_results>",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────
IMAGE_GEN_RE = re.compile(
    r'\b(generate|create|make|draw|paint|design|produce|render)\b.{0,25}'
    r'\b(image|picture|photo|illustration|artwork|graphic|wallpaper|logo|poster|banner|sketch|portrait)\b'
    r'|^/imagine\b',
    re.IGNORECASE,
)

def is_image_request(text: str) -> bool:
    return bool(IMAGE_GEN_RE.search(text))

IMAGE_DECLINE = (
    "I'm text-only — I can't generate images. 🙅\n\n"
    "Try these free tools:\n"
    "• **[Adobe Firefly](https://firefly.adobe.com)** — free, high quality\n"
    "• **[Microsoft Designer](https://designer.microsoft.com)** — free with Microsoft account\n"
    "• **[Ideogram](https://ideogram.ai)** — great for text in images\n"
    "• **[Craiyon](https://www.craiyon.com)** — completely free, no account needed\n\n"
    "Want me to write a detailed prompt for any of these? ✍️"
)

# ─────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────
SYSTEM = """You are Zippy, a smart AI assistant made by Arul Vethathiri.

## IDENTITY
- Text-only AI. You cannot generate images. Period.
- Made by Arul Vethathiri, Class 11 student (2026).

## CRITICAL — HOW TO USE SEARCH RESULTS
When you see a <search_results> block:
- That data is REAL and was fetched live moments ago.
- READ it carefully and use the actual numbers and facts in your answer.
- NEVER say "I don't have real-time access" or "my training data is limited".
- NEVER say "I cannot provide current prices" — the price is in the data above.
- Answer using the exact figures. Cite the source naturally.
- If search found nothing useful, say "I searched but couldn't find current data on that."

## TONE
- Smart, calm, friendly — like a knowledgeable friend.
- Conversational, not corporate. Not over-excited.
- Greetings: one short sentence.

## RESPONSE RULES
1. Answer EXACTLY what was asked. Nothing extra.
2. Numbers/prices → state them immediately in the first sentence.
3. Simple questions → 1-3 sentences.
4. Explanations → short bullets, no long intro.
5. Code → give it directly with brief comments.
6. Math → show steps briefly.
7. Creative → complete the full piece, no preamble.
8. NEVER say: "Certainly!", "Great question!", "Of course!", "As an AI",
   "I don't have real-time access", "my training data", "I'd be happy to".
9. NEVER repeat the question.
10. End with exactly 1 relevant emoji."""

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
    "what can you do": (
        "I can answer questions, help with code, explain concepts, do maths, "
        "write content, and search the web for live data like prices, weather and news. "
        "I can't generate images. 💬"
    ),
    "are you an ai":  "Yes — Zippy AI, made by Arul Vethathiri. 🤖",
    "are you human":  "Nope, I'm Zippy — an AI. A pretty capable one though! 😄",
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

    # 1. Clean filler sounds
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message, flags=re.IGNORECASE
    ).strip()

    if not user_input:
        return {"reply": "Didn't catch that — say again? 😊",
                "thinking": "Empty input.", "searched": False, "sources": []}

    text_lower = user_input.lower().strip()

    # 2. Image request → decline immediately
    if is_image_request(user_input):
        return {
            "reply":      IMAGE_DECLINE,
            "thinking":   "Image generation request — text-only AI.",
            "searched":   False,
            "sources":    [],
            "model_used": "static",
        }

    # 3. Static identity / social replies
    if text_lower in IDENTITY:
        return {"reply": IDENTITY[text_lower], "thinking": "Identity.",
                "searched": False, "sources": [], "model_used": "static"}
    if text_lower in SOCIAL:
        return {"reply": SOCIAL[text_lower], "thinking": "Social.",
                "searched": False, "sources": [], "model_used": "static"}

    # 4. Thinking step — AI model decides, with timeout + rule fallback safety net
    think        = run_thinking(user_input, timeout_seconds=5.0)
    needs_search = think.get("needs_search", False)
    search_query = (think.get("search_query") or "").strip() or user_input
    reasoning    = think.get("reasoning", "")
    think_model  = think.get("_model", "fallback")

    print(f"[thinking] needs_search={needs_search} via={think_model} reasoning='{reasoning}'")

    # 5. Run search if needed
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search:
        print(f"[search] → '{search_query}'")
        results        = run_search(search_query)
        search_context = build_search_context(search_query, results)
        if results:
            searched       = True
            search_sources = [
                {"title": r["title"], "url": r["url"], "source": r["source"]}
                for r in results
            ]

    # 6. Build messages for the AI
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]

    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            messages.append({
                "role":    h["role"],
                "content": str(h["content"])[:1500],
            })

    if search_context:
        final_user = (
            f"{search_context}\n\n"
            f"Now answer this question using the data above:\n{user_input}"
        )
    else:
        final_user = user_input

    messages.append({"role": "user", "content": final_user})

    # 7. Call AI with full model fallback chain
    try:
        reply, model_used = call_with_fallback(
            messages, CHAT_MODELS,
            max_tokens=600, temperature=0.65, top_p=0.9,
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
