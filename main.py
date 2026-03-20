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
          f"{time.strftime('%H:%M:%S', time.localtime(available_at))}")

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
    if seconds < 5:    return "a few seconds"
    if seconds < 90:   return f"{math.ceil(seconds)} seconds"
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
#  SEARCH LAYER
#  Strategy: use only APIs that return actual content — no HTML scraping.
#  Every source below returns real text that gets injected into the AI.
#
#  Sources:
#   1. Wikipedia Search API  → finds the right article + returns FULL extract
#   2. CoinGecko API         → live crypto prices (no key needed)
#   3. wttr.in               → live weather (no key needed)
#   4. Open-Meteo geocoding  → location lookup for weather fallback
#   5. REST Countries API    → country facts (no key needed)
#   6. Numbers/date facts    → fun facts via numbersapi.com
#
#  ALL of these are free, stable, and return real structured data.
#  NO HTML scraping — server IPs get blocked.
# ─────────────────────────────────────────────────────────────────────

BASE_HEADERS = {"User-Agent": "ZippyAI/2.0 (educational chatbot)"}

# ── 1. Wikipedia: search → get full article extract ──────────────────

def _wiki_search_title(query: str) -> str | None:
    """Find the best matching Wikipedia article title for a query."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":  "query",
                "list":    "search",
                "srsearch": query,
                "srlimit": 1,
                "format":  "json",
                "origin":  "*",
            },
            headers=BASE_HEADERS,
            timeout=7,
        )
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except Exception as e:
        print(f"[wiki-search] {e}")
    return None


def _wiki_fetch_extract(title: str, chars: int = 3000) -> str:
    """Fetch the full extract text of a Wikipedia article by title."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":    "query",
                "prop":      "extracts",
                "exintro":   False,       # get more than just intro
                "explaintext": True,      # plain text, no HTML
                "titles":    title,
                "format":    "json",
                "origin":    "*",
                "exchars":   chars,       # how many characters to return
            },
            headers=BASE_HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract and len(extract) > 50:
                return extract
    except Exception as e:
        print(f"[wiki-fetch] {e}")
    return ""


def search_wikipedia(query: str) -> dict | None:
    """
    Full Wikipedia pipeline:
    1. Search for best matching article
    2. Fetch up to 3000 chars of its content
    Returns dict with title, url, content — or None if nothing found.
    """
    title = _wiki_search_title(query)
    if not title:
        return None

    content = _wiki_fetch_extract(title, chars=3000)
    if not content:
        return None

    url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"
    print(f"[wiki] ✓ '{title}' ({len(content)} chars)")
    return {
        "source":  "Wikipedia",
        "title":   title,
        "url":     url,
        "content": content,
    }


# ── 2. Crypto prices via CoinGecko ───────────────────────────────────

CRYPTO_MAP = {
    "bitcoin": "bitcoin",   "btc":  "bitcoin",
    "ethereum": "ethereum", "eth":  "ethereum",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "solana":   "solana",   "sol":  "solana",
    "bnb": "binancecoin",   "binance coin": "binancecoin",
    "xrp": "ripple",        "ripple": "ripple",
    "cardano": "cardano",   "ada":  "cardano",
    "litecoin": "litecoin", "ltc":  "litecoin",
    "polkadot": "polkadot", "dot":  "polkadot",
    "shib": "shiba-inu",    "shiba inu": "shiba-inu",
    "tron": "tron",         "trx":  "tron",
    "avax": "avalanche-2",  "avalanche": "avalanche-2",
    "matic": "matic-network", "polygon": "matic-network",
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
            headers=BASE_HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()

        lines = []
        for coin_id, info in data.items():
            usd    = info.get("usd",              "N/A")
            inr    = info.get("inr",              "N/A")
            chg    = info.get("usd_24h_change",   None)
            mcap   = info.get("usd_market_cap",   None)
            vol    = info.get("usd_24h_vol",      None)

            chg_str  = f"{chg:+.2f}%" if chg  is not None else "N/A"
            mcap_str = f"${mcap:,.0f}" if mcap is not None else "N/A"
            vol_str  = f"${vol:,.0f}"  if vol  is not None else "N/A"
            usd_str  = f"${usd:,.2f}"  if isinstance(usd, (int, float)) else str(usd)
            inr_str  = f"₹{inr:,.2f}"  if isinstance(inr, (int, float)) else str(inr)

            lines.append(
                f"{coin_id.capitalize()}:\n"
                f"  Price (USD): {usd_str}\n"
                f"  Price (INR): {inr_str}\n"
                f"  24h Change:  {chg_str}\n"
                f"  Market Cap:  {mcap_str}\n"
                f"  24h Volume:  {vol_str}"
            )

        content = "\n\n".join(lines)
        print(f"[crypto] ✓ {coins}")
        return {
            "source":  "CoinGecko (live)",
            "title":   "Live Cryptocurrency Prices",
            "url":     "https://www.coingecko.com",
            "content": content,
        }
    except Exception as e:
        print(f"[crypto] failed: {e}")
        return None


# ── 3. Weather via wttr.in ────────────────────────────────────────────

WEATHER_WORDS = {
    "weather", "temperature", "forecast", "humidity",
    "rain", "sunny", "hot", "cold", "wind", "climate",
}

def _detect_city(query: str) -> str | None:
    q = query.lower()
    if not any(w in q for w in WEATHER_WORDS):
        return None
    # "weather in Chennai" / "weather of London"
    m = re.search(r'weather\s+(?:in|at|for|of)?\s*([A-Za-z\s]{2,25})', query, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # "Chennai weather"
    m = re.search(r'([A-Za-z\s]{2,25})\s+weather', query, re.IGNORECASE)
    if m:
        return m.group(1).strip()
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

        desc    = cur.get("weatherDesc",  [{}])[0].get("value", "N/A")
        temp_c  = cur.get("temp_C",       "N/A")
        temp_f  = cur.get("temp_F",       "N/A")
        feels_c = cur.get("FeelsLikeC",   "N/A")
        humidity= cur.get("humidity",     "N/A")
        wind    = cur.get("windspeedKmph","N/A")
        vis     = cur.get("visibility",   "N/A")
        uv      = cur.get("uvIndex",      "N/A")

        # 3-day forecast
        forecast_lines = []
        for day in d.get("weather", [])[:3]:
            date    = day.get("date",       "")
            max_c   = day.get("maxtempC",   "N/A")
            min_c   = day.get("mintempC",   "N/A")
            desc_d  = day.get("hourly",     [{}])[4].get("weatherDesc", [{}])[0].get("value", "N/A")
            forecast_lines.append(f"  {date}: {desc_d}, {min_c}°C – {max_c}°C")

        content = (
            f"Current weather in {city}:\n"
            f"  Condition:   {desc}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  Feels like:  {feels_c}°C\n"
            f"  Humidity:    {humidity}%\n"
            f"  Wind speed:  {wind} km/h\n"
            f"  Visibility:  {vis} km\n"
            f"  UV Index:    {uv}\n\n"
            f"3-Day Forecast:\n" + "\n".join(forecast_lines)
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


# ── 4. REST Countries API ─────────────────────────────────────────────

COUNTRY_WORDS = {
    "country", "capital", "population", "currency", "language",
    "continent", "flag", "area", "gdp", "nation",
}

def _detect_country_query(query: str) -> bool:
    return any(w in query.lower() for w in COUNTRY_WORDS)

def search_country(query: str) -> dict | None:
    if not _detect_country_query(query):
        return None
    # Extract a country name: take words that are capitalised or after "of/in/about"
    m = re.search(
        r'(?:of|in|about|for|country|capital of|population of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        query
    )
    if not m:
        # Try first capitalised word as country name
        m = re.search(r'\b([A-Z][a-z]{2,})\b', query)
    if not m:
        return None
    country_name = m.group(1).strip()

    try:
        r = requests.get(
            f"https://restcountries.com/v3.1/name/{requests.utils.quote(country_name)}",
            params={"fullText": "false"},
            headers=BASE_HEADERS,
            timeout=7,
        )
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict):
            return None
        c = data[0]

        name       = c.get("name", {}).get("common", country_name)
        capital    = ", ".join(c.get("capital", ["N/A"]))
        population = c.get("population", "N/A")
        region     = c.get("region", "N/A")
        subregion  = c.get("subregion", "N/A")
        area       = c.get("area", "N/A")
        currencies = ", ".join(
            f"{v.get('name','')} ({v.get('symbol','')})"
            for v in c.get("currencies", {}).values()
        ) or "N/A"
        languages  = ", ".join(c.get("languages", {}).values()) or "N/A"
        timezones  = ", ".join(c.get("timezones", [])[:3])

        content = (
            f"Country: {name}\n"
            f"  Capital:    {capital}\n"
            f"  Population: {population:,}\n"
            f"  Region:     {region} ({subregion})\n"
            f"  Area:       {area} km²\n"
            f"  Currencies: {currencies}\n"
            f"  Languages:  {languages}\n"
            f"  Timezones:  {timezones}"
        )
        print(f"[country] ✓ {name}")
        return {
            "source":  "REST Countries API",
            "title":   f"Country Data: {name}",
            "url":     "https://restcountries.com",
            "content": content,
        }
    except Exception as e:
        print(f"[country] failed: {e}")
        return None


# ── Master search orchestrator ────────────────────────────────────────

# Keywords that indicate a need for current/live information
SEARCH_KEYWORDS = [
    "price", "cost", "worth", "value", "rate", "how much",
    "bitcoin", "ethereum", "crypto", "stock", "share", "market",
    "weather", "temperature", "forecast", "rain",
    "score", "result", "winner", "won", "beat", "champion",
    "latest", "recent", "current", "today", "now", "live",
    "this week", "this month", "this year", "2024", "2025", "2026",
    "news", "update", "announce", "release", "launch",
    "who is", "who's", "who won", "who leads",
    "ipl", "cricket", "football", "nba", "nfl", "match", "tournament",
    "capital", "population", "country", "currency",
    "what happened", "breaking",
]

def needs_search_by_keyword(query: str) -> bool:
    lower = query.lower()
    return any(kw in lower for kw in SEARCH_KEYWORDS)


def run_search(query: str) -> list[dict]:
    """
    Run all relevant search sources for the query.
    Returns list of source dicts, each with: source, title, url, content.
    Content is FULL text — not a snippet — so the AI has real data to answer from.
    """
    results: list[dict] = []

    # Crypto prices — highest priority for price queries
    c = search_crypto(query)
    if c:
        results.append(c)

    # Weather — if location + weather word detected
    w = search_weather(query)
    if w:
        results.append(w)

    # Country facts
    co = search_country(query)
    if co:
        results.append(co)

    # Wikipedia — always try, it's the most reliable free text source
    wiki = search_wikipedia(query)
    if wiki:
        results.append(wiki)

    print(f"[search] total sources collected: {len(results)}")
    return results


def build_search_context(query: str, results: list[dict]) -> str:
    """
    Build the context block that gets injected directly into the AI prompt.
    Uses clear delimiters so the model knows exactly what is live data.
    """
    now_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    if not results:
        return (
            f"<search_results>\n"
            f"SEARCH ATTEMPTED: '{query}'\n"
            f"RESULT: No data found from any source.\n"
            f"INSTRUCTION: Tell the user you searched but couldn't find current data. "
            f"Do NOT say 'I don't have real-time access' — you searched and found nothing.\n"
            f"</search_results>"
        )

    parts = [
        f"<search_results>",
        f"QUERY: {query}",
        f"RETRIEVED: {now_str}",
        f"SOURCES: {len(results)} found",
        "",
        "=" * 60,
    ]

    for i, r in enumerate(results, 1):
        parts += [
            f"",
            f"SOURCE {i}: {r['source']}",
            f"TITLE:  {r['title']}",
            f"URL:    {r['url']}",
            f"",
            f"CONTENT:",
            r['content'],
            "",
            "-" * 40,
        ]

    parts += [
        "",
        "INSTRUCTION: The above is REAL, LIVE data retrieved right now.",
        "Use it directly to answer the user's question.",
        "Do NOT say you lack real-time access — you have the data above.",
        "Answer based ONLY on what is shown above. Be specific and accurate.",
        "</search_results>",
    ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
#  IMAGE DETECTION + DECLINE
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
- Text-only AI. You cannot generate images.
- Made by Arul Vethathiri, Class 11 student (2026).

## CRITICAL: HOW TO USE SEARCH RESULTS
When you see a <search_results> block in the conversation:
- That data is REAL and was fetched live just now.
- Read it carefully and use the actual numbers/facts in your answer.
- NEVER say "I don't have real-time access" or "my training data" — you have live data.
- NEVER say "I cannot provide current prices" — the price is right there in the data.
- Answer using the exact figures from the search results.
- Mention the source naturally (e.g. "According to CoinGecko..." or "The current price is...").
- If search found nothing, say "I searched but couldn't find current data on that."

## TONE
- Smart, calm, friendly. Like a knowledgeable friend.
- Conversational, not corporate. Not over-excited.
- Greetings: one short sentence.

## RESPONSE RULES
1. Answer what was asked. Nothing more, nothing less.
2. Numeric answers → give the number first, immediately.
3. Simple questions → 1-3 sentences.
4. Explanations → short bullets, no long intro.
5. Code → give it directly with brief comments.
6. Math → show steps, keep brief.
7. Creative → complete the full piece, no preamble.
8. NEVER say: "Certainly!", "Great question!", "Of course!", "As an AI",
   "I don't have real-time access", "my training data is limited",
   "I cannot provide current prices", "I'd be happy to".
9. NEVER repeat the question.
10. End every response with exactly 1 relevant emoji."""

# ─────────────────────────────────────────────────────────────────────
#  THINKING STEP
# ─────────────────────────────────────────────────────────────────────
THINK_PROMPT = """Decide if this user question needs a live web search.

Output ONLY a JSON object. No markdown fences. No explanation. Just JSON.

{
  "needs_search": true or false,
  "search_query": "short optimised search string, or empty string",
  "reasoning": "one sentence"
}

ALWAYS search for:
- Any price (crypto, stock, gold, currency, oil)
- Sports scores, match results, tournament winners
- Weather or temperature for a location  
- "Latest", "recent", "current", "today", "now", "this week"
- Who currently holds a job/position/title
- New product releases or news events
- Country facts (capital, population, currency)
- Anything that changes over time

NEVER search for:
- How to code something
- Math or science theory
- History before 2023
- Grammar, definitions, concepts
- Creative writing, poems, stories
- Opinions or general advice

Good search_query examples:
- "bitcoin price USD today"
- "weather Chennai today"  
- "IPL 2025 final winner"
- "India capital population"

JSON only:"""


def run_thinking(user_input: str) -> dict:
    messages = [
        {"role": "system", "content": THINK_PROMPT},
        {"role": "user",   "content": user_input},
    ]
    try:
        raw, _ = call_with_fallback(
            messages, THINK_MODELS,
            max_tokens=100, temperature=0.0, top_p=1.0,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        m   = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if m:
            raw = m.group(0)
        result = json.loads(raw)
        if "needs_search" in result:
            return result
    except RuntimeError:
        pass
    except Exception as e:
        print(f"[thinking] parse failed: {e}")

    # Keyword fallback — always reliable
    if needs_search_by_keyword(user_input):
        return {
            "needs_search": True,
            "search_query": user_input,
            "reasoning":    "Keyword-based fallback: time-sensitive terms detected.",
        }
    return {
        "needs_search": False,
        "search_query": "",
        "reasoning":    "No time-sensitive keywords found.",
    }


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
        "I can answer questions, help with code, explain things, do math, write content, "
        "and search the web for live data like prices, weather, and news. "
        "I can't generate images though. 💬"
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
            "thinking":   "Image generation request — text-only AI, declined.",
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

    # 4. Thinking step — decide if search needed + best query
    think        = run_thinking(user_input)
    needs_search = think.get("needs_search", False)
    search_query = think.get("search_query", "").strip() or user_input
    reasoning    = think.get("reasoning", "")

    # 5. Search — always runs if needs_search, returns full content not snippets
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search:
        print(f"[search] query: '{search_query}'")
        results = run_search(search_query)
        search_context = build_search_context(search_query, results)
        if results:
            searched       = True
            search_sources = [
                {"title": r["title"], "url": r["url"], "source": r["source"]}
                for r in results
            ]

    # 6. Build messages for the AI
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]

    # Conversation history
    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            messages.append({
                "role":    h["role"],
                "content": str(h["content"])[:1500],
            })

    # User turn: inject search context as part of the user message
    # This placement (in user turn) is more reliable than a separate system injection
    if search_context:
        final_user = (
            f"{search_context}\n\n"
            f"Based on the search results above, answer this:\n{user_input}"
        )
    else:
        final_user = user_input

    messages.append({"role": "user", "content": final_user})

    # 7. Call AI with fallback chain
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
