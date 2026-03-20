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
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager

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
    model_cooldown[model_id] = time.time() + retry_after
    print(f"[rate-limit] {model_id} blocked {retry_after:.0f}s")

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
    return f"{math.ceil(seconds / 60)} minute{'s' if math.ceil(seconds / 60) != 1 else ''}"


# ─────────────────────────────────────────────────────────────────────
#  CONNECTION WARM-UP
#  On cold start (Render free tier), the first Groq request takes 8-15s.
#  We fire a tiny warm-up call in the background at startup so that by
#  the time the first real user request arrives, the connection is live.
# ─────────────────────────────────────────────────────────────────────
_groq_client: Groq | None = None

def get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

def _warm_up():
    """Fire a minimal Groq request so the connection is ready."""
    try:
        time.sleep(1)   # let the server finish starting
        get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print("[warmup] ✓ Groq connection warmed up")
    except Exception as e:
        print(f"[warmup] failed (ok, will retry on first request): {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            resp = get_client().chat.completions.create(
                model=mid, messages=messages,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            )
            print(f"[model] ✓ {mid}")
            return resp.choices[0].message.content.strip(), mid
        except RateLimitError as e:
            wait = parse_retry_after(e)
            mark_rate_limited(mid, wait)
            tried.append(f"{model['name']} — rate limited ({format_wait(wait)})")
        except APIStatusError as e:
            mark_rate_limited(mid, 15)
            tried.append(f"{model['name']} — API error {e.status_code}")
        except Exception as e:
            tried.append(f"{model['name']} — {type(e).__name__}")
            print(f"[error] {mid} → {e}")
    soonest = earliest_available_in(models)
    raise RuntimeError(f"QUOTA_EXCEEDED|{format_wait(soonest)}|{'||'.join(tried)}")


# ─────────────────────────────────────────────────────────────────────
#  THINKING STEP  —  pure AI, no rules at all
#
#  The AI model is the ONLY decision maker for whether to search.
#  No keyword matching. No fallback rules.
#  It understands context, normal conversation, and nuance.
#
#  The only engineering trick: a thread with a 9s timeout to survive
#  cold starts on Render. If it truly times out we return a safe
#  "don't search" so the chat still responds rather than hanging.
# ─────────────────────────────────────────────────────────────────────

THINK_PROMPT = """You are the search decision engine for Zippy AI.

Your job: read the user's message and decide if answering it requires fetching live data from the internet right now.

OUTPUT: only a raw JSON object. No markdown. No extra text. Just JSON.

{
  "needs_search": true or false,
  "search_query": "optimised search string, or empty string if needs_search is false",
  "reasoning": "one sentence"
}

━━━━━━━━━━━━━━━━━━━━━━━━
WHEN TO SEARCH (needs_search: true)
━━━━━━━━━━━━━━━━━━━━━━━━
Search when the question asks for something that CHANGES OVER TIME and your training data would be stale:

Prices & finance:
  - Any cryptocurrency price (bitcoin, ethereum, dogecoin, etc.)
  - Stock prices, indices (Sensex, Nifty, Nasdaq, S&P 500)
  - Gold, silver, oil, petrol, diesel prices
  - Currency exchange rates (USD to INR, etc.)

Live data:
  - Weather or temperature for any city
  - Sports scores, match results, tournament winners
  - Flight or train status

Current events & news:
  - "latest news about X", "what happened with X", "recent updates"
  - Breaking news, current events
  - Elections, political changes, government decisions

Recency signals — search when message contains:
  - "today", "right now", "currently", "latest", "recent", "this week", "this month"
  - Years: 2024, 2025, 2026
  - "just", "new", "update", "release", "announced"

People in current roles:
  - "who is the current president/PM/CEO/champion of X"
  - "who leads X right now"

━━━━━━━━━━━━━━━━━━━━━━━━
WHEN NOT TO SEARCH (needs_search: false)
━━━━━━━━━━━━━━━━━━━━━━━━
Do NOT search for anything that is timeless or conversational:

Normal conversation:
  - Greetings: "hi", "hello", "how are you", "good morning"
  - Small talk: "what do you think about X", "tell me a joke"
  - Opinions and preferences

Coding & tech (timeless knowledge):
  - "how do I write a for loop in Python"
  - "what is recursion", "explain async/await"
  - "write a function that does X"
  - "fix this code", "debug this"
  - Algorithms, data structures, syntax, programming concepts

Math & science theory:
  - Calculations, equations, algebra, calculus
  - Physics/chemistry/biology concepts and definitions
  - "what is the formula for X"

Stable knowledge:
  - Definitions of concepts that don't change
  - Historical events (before 2023)
  - "explain what X is", "what does X mean"
  - Grammar, language, translation

Creative tasks:
  - "write a poem", "write a story", "write an essay"
  - "help me write an email"
  - Any generation or creative writing task

━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH QUERY TIPS
━━━━━━━━━━━━━━━━━━━━━━━━
Make queries short, specific, and search-engine optimised:
  User: "what is bitcoin price" → "bitcoin price USD INR today"
  User: "weather in Chennai" → "Chennai weather today"
  User: "who won IPL 2025" → "IPL 2025 winner final"
  User: "latest news about AI" → "AI news today 2025"
  User: "what is the dollar to rupee rate" → "USD INR exchange rate today"

JSON ONLY. No extra text."""


def run_thinking(user_input: str, timeout: float = 9.0) -> dict:
    """
    Pure AI thinking. Returns the model's decision.
    Falls back to safe defaults only if the model completely fails/times out.
    """
    result_box: list[dict] = []
    error_box:  list[str]  = []

    def _call():
        try:
            raw, used = call_with_fallback(
                [{"role": "system", "content": THINK_PROMPT},
                 {"role": "user",   "content": user_input}],
                THINK_MODELS,
                max_tokens=120,
                temperature=0.0,  # deterministic JSON
                top_p=1.0,
            )
            # Clean any markdown the model might wrap around JSON
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            # Extract the JSON object
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed.get("needs_search"), bool):
                    parsed["_via"] = used
                    result_box.append(parsed)
                    return
            error_box.append(f"bad JSON: {raw[:100]}")
        except Exception as e:
            error_box.append(str(e))

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result_box:
        r = result_box[0]
        print(f"[think] via={r.get('_via','?')} needs_search={r['needs_search']} "
              f"q='{r.get('search_query','')}' | {r.get('reasoning','')}")
        return r

    # Model timed out or failed — log it, default to no search so chat still responds
    print(f"[think] failed/timeout ({error_box}) — defaulting needs_search=False")
    return {
        "needs_search": False,
        "search_query": "",
        "reasoning":    "Thinking timed out — defaulting to no search.",
    }


# ─────────────────────────────────────────────────────────────────────
#  SEARCH SOURCES
#  All free. No API keys. Tested to work on Render server IPs.
# ─────────────────────────────────────────────────────────────────────
BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZippyAI/2.0)",
    "Accept":     "application/json, text/html, */*",
}

# ── 1. CRYPTO PRICES  (CoinGecko primary, CryptoCompare fallback) ─────

CRYPTO_MAP = {
    "bitcoin":   "bitcoin",       "btc":      "bitcoin",
    "ethereum":  "ethereum",      "eth":      "ethereum",
    "dogecoin":  "dogecoin",      "doge":     "dogecoin",
    "solana":    "solana",        "sol":      "solana",
    "bnb":       "binancecoin",   "binance":  "binancecoin",
    "xrp":       "ripple",        "ripple":   "ripple",
    "cardano":   "cardano",       "ada":      "cardano",
    "litecoin":  "litecoin",      "ltc":      "litecoin",
    "polkadot":  "polkadot",      "dot":      "polkadot",
    "shib":      "shiba-inu",     "shiba":    "shiba-inu",
    "tron":      "tron",          "trx":      "tron",
    "avax":      "avalanche-2",   "avalanche":"avalanche-2",
    "matic":     "matic-network", "polygon":  "matic-network",
    "pepe":      "pepe",
}

CRYPTO_CC = {
    "bitcoin": "BTC",  "btc":  "BTC",
    "ethereum":"ETH",  "eth":  "ETH",
    "dogecoin":"DOGE", "doge": "DOGE",
    "solana":  "SOL",  "sol":  "SOL",
    "bnb":     "BNB",
    "xrp":     "XRP",  "ripple":"XRP",
    "cardano": "ADA",  "ada":  "ADA",
    "litecoin":"LTC",  "ltc":  "LTC",
    "polkadot":"DOT",  "dot":  "DOT",
    "shib":    "SHIB", "shiba":"SHIB",
    "tron":    "TRX",  "trx":  "TRX",
    "avax":    "AVAX", "avalanche":"AVAX",
    "matic":   "MATIC","polygon":"MATIC",
}

def _detect_crypto(q: str) -> tuple[list[str], list[str]]:
    lower = q.lower()
    cg = list({v for k, v in CRYPTO_MAP.items() if k in lower})
    cc = list({v for k, v in CRYPTO_CC.items()  if k in lower})
    return cg, cc

def _fmt(val, prefix="$", decimals=2) -> str:
    if isinstance(val, (int, float)):
        if val >= 1:
            return f"{prefix}{val:,.{decimals}f}"
        else:
            return f"{prefix}{val:.6f}"
    return "N/A"

def _coingecko(coin_ids: list[str]) -> dict | None:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(coin_ids), "vs_currencies": "usd,inr",
                    "include_24hr_change": "true", "include_market_cap": "true",
                    "include_24hr_vol": "true"},
            headers=BASE_HEADERS, timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        if not data: return None
        lines = []
        for cid, info in data.items():
            usd  = info.get("usd");  inr = info.get("inr")
            chg  = info.get("usd_24h_change")
            mcap = info.get("usd_market_cap"); vol = info.get("usd_24h_vol")
            lines.append(
                f"{cid.capitalize()} (as of {time.strftime('%H:%M UTC', time.gmtime())}):\n"
                f"  Price:      {_fmt(usd)} USD  /  {_fmt(inr, '₹')} INR\n"
                f"  24h Change: {f'{chg:+.2f}%' if chg is not None else 'N/A'}\n"
                f"  Market Cap: {_fmt(mcap, '$', 0)}\n"
                f"  24h Volume: {_fmt(vol, '$', 0)}"
            )
        print(f"[crypto] CoinGecko ✓ {coin_ids}")
        return {"source": "CoinGecko (live)", "title": "Live Crypto Prices",
                "url": "https://www.coingecko.com", "content": "\n\n".join(lines)}
    except Exception as e:
        print(f"[crypto] CoinGecko failed: {e}"); return None

def _cryptocompare(symbols: list[str]) -> dict | None:
    try:
        r = requests.get(
            "https://min-api.cryptocompare.com/data/pricemultifull",
            params={"fsyms": ",".join(symbols), "tsyms": "USD,INR"},
            headers=BASE_HEADERS, timeout=8,
        )
        r.raise_for_status()
        data = r.json().get("RAW", {})
        if not data: return None
        lines = []
        for sym, markets in data.items():
            u = markets.get("USD", {}); n = markets.get("INR", {})
            price_u = u.get("PRICE"); price_n = n.get("PRICE")
            chg     = u.get("CHANGEPCT24HOUR")
            mcap    = u.get("MKTCAP"); vol = u.get("VOLUME24HOURTO")
            lines.append(
                f"{sym} (as of {time.strftime('%H:%M UTC', time.gmtime())}):\n"
                f"  Price:      {_fmt(price_u)} USD  /  {_fmt(price_n, '₹')} INR\n"
                f"  24h Change: {f'{chg:+.2f}%' if isinstance(chg,(int,float)) else 'N/A'}\n"
                f"  Market Cap: {_fmt(mcap,'$',0)}\n"
                f"  24h Volume: {_fmt(vol,'$',0)}"
            )
        print(f"[crypto] CryptoCompare ✓ {symbols}")
        return {"source": "CryptoCompare (live)", "title": "Live Crypto Prices",
                "url": "https://www.cryptocompare.com", "content": "\n\n".join(lines)}
    except Exception as e:
        print(f"[crypto] CryptoCompare failed: {e}"); return None

def search_crypto(query: str) -> dict | None:
    cg, cc = _detect_crypto(query)
    if not cg: return None
    return _coingecko(cg) or (_cryptocompare(cc) if cc else None)


# ── 2. GOOGLE NEWS RSS  ───────────────────────────────────────────────
#  Google News RSS is free, has no API key, returns real news articles
#  from all major publishers, and works reliably on server IPs.
#  URL: https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en

def search_google_news(query: str, max_results: int = 5) -> dict | None:
    """
    Fetch Google News RSS for the query.
    Returns real headlines + source + publication date + description.
    No API key needed. Works on all server IPs.
    """
    try:
        rss_url = "https://news.google.com/rss/search"
        r = requests.get(
            rss_url,
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers={**BASE_HEADERS, "Accept": "application/rss+xml, application/xml, text/xml"},
            timeout=10,
        )
        r.raise_for_status()

        root = ET.fromstring(r.content)
        items = root.findall(".//item")[:max_results]

        if not items:
            return None

        lines = []
        for i, item in enumerate(items, 1):
            title   = (item.findtext("title")       or "").strip()
            link    = (item.findtext("link")        or "").strip()
            pub     = (item.findtext("pubDate")     or "").strip()
            source  = (item.findtext("source")      or "Google News").strip()
            desc    = (item.findtext("description") or "").strip()

            # Strip HTML from description
            desc = re.sub(r'<[^>]+>', '', desc).strip()
            # Google News titles often have " - Source Name" appended
            title_clean = re.sub(r'\s*-\s*[^-]+$', '', title).strip() or title

            lines.append(
                f"{i}. {title_clean}\n"
                f"   Source: {source}\n"
                f"   Published: {pub}\n"
                + (f"   {desc[:250]}\n" if desc else "")
                + f"   Link: {link}"
            )

        print(f"[news] Google News ✓ {len(lines)} articles for '{query}'")
        return {
            "source":  "Google News (live)",
            "title":   f"News: {query}",
            "url":     f"https://news.google.com/search?q={requests.utils.quote(query)}",
            "content": "\n\n".join(lines),
        }
    except Exception as e:
        print(f"[news] Google News failed: {e}")
        return None


# ── 3. WEATHER (wttr.in) ──────────────────────────────────────────────
def _detect_city(query: str) -> str | None:
    q = query.lower()
    weather_words = {"weather","temperature","forecast","humidity","rain",
                     "wind","climate","hot","cold","sunny","cloudy"}
    if not any(w in q for w in weather_words):
        return None
    patterns = [
        r'weather\s+(?:in|at|for|of)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'([A-Za-z][A-Za-z\s]{1,24})\s+weather',
        r'(?:temperature|forecast)\s+(?:in|of|at)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'(?:how).{0,10}(?:hot|cold|warm).{0,10}in\s+([A-Za-z][A-Za-z\s]{1,20})',
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            city = re.sub(
                r'\b(today|now|currently|like|is|the|a|an|going|to|be)\b',
                '', m.group(1), flags=re.IGNORECASE
            ).strip()
            if len(city) > 1:
                return city
    return None

def search_weather(query: str) -> dict | None:
    city = _detect_city(query)
    if not city: return None
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
            date  = day.get("date","")
            maxc  = day.get("maxtempC","N/A"); minc = day.get("mintempC","N/A")
            desc2 = (day.get("hourly") or [{}])[4].get(
                "weatherDesc",[{}])[0].get("value","N/A")
            forecast.append(f"  {date}: {desc2}, {minc}°C – {maxc}°C")
        content = (
            f"Current weather in {city} (as of {time.strftime('%H:%M UTC', time.gmtime())}):\n"
            f"  Condition:   {desc}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  Feels like:  {feels}°C\n"
            f"  Humidity:    {humidity}%\n"
            f"  Wind:        {wind} km/h\n"
            f"  UV Index:    {uv}\n\n"
            f"3-Day Forecast:\n" + "\n".join(forecast)
        )
        print(f"[weather] ✓ {city}")
        return {"source": "wttr.in (live)", "title": f"Weather in {city}",
                "url": f"https://wttr.in/{city}", "content": content}
    except Exception as e:
        print(f"[weather] failed: {e}"); return None


# ── 4. WIKIPEDIA ──────────────────────────────────────────────────────
def search_wikipedia(query: str) -> dict | None:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":query,
                    "srlimit":1,"format":"json","origin":"*"},
            headers=BASE_HEADERS, timeout=7,
        )
        r.raise_for_status()
        results = r.json().get("query",{}).get("search",[])
        if not results: return None
        title = results[0]["title"]
    except Exception as e:
        print(f"[wiki-search] {e}"); return None

    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","prop":"extracts","exintro":False,
                    "explaintext":True,"titles":title,"format":"json",
                    "origin":"*","exchars":3000},
            headers=BASE_HEADERS, timeout=8,
        )
        r.raise_for_status()
        for page in r.json().get("query",{}).get("pages",{}).values():
            extract = page.get("extract","")
            if extract and len(extract) > 80:
                url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ','_'))}"
                print(f"[wiki] ✓ '{title}' ({len(extract)} chars)")
                return {"source":"Wikipedia","title":title,"url":url,"content":extract}
    except Exception as e:
        print(f"[wiki-extract] {e}")
    return None


# ── 5. REST COUNTRIES ─────────────────────────────────────────────────
def _detect_country(query: str) -> str | None:
    if not any(w in query.lower() for w in
               {"capital","population","currency","language","country","nation","area"}):
        return None
    m = re.search(
        r'(?:of|in|about|for|capital of|population of|currency of)\s+'
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', query)
    if m: return m.group(1).strip()
    m = re.search(r'\b([A-Z][a-z]{3,})\b', query)
    if m: return m.group(1).strip()
    return None

def search_country(query: str) -> dict | None:
    name = _detect_country(query)
    if not name: return None
    try:
        r = requests.get(
            f"https://restcountries.com/v3.1/name/{requests.utils.quote(name)}",
            headers=BASE_HEADERS, timeout=7,
        )
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict): return None
        c = data[0]
        common    = c.get("name",{}).get("common", name)
        capital   = ", ".join(c.get("capital",["N/A"]))
        pop       = c.get("population", 0)
        region    = c.get("region","N/A"); sub = c.get("subregion","N/A")
        area      = c.get("area","N/A")
        currencies = ", ".join(
            f"{v.get('name','')} ({v.get('symbol','')})"
            for v in c.get("currencies",{}).values()) or "N/A"
        languages  = ", ".join(c.get("languages",{}).values()) or "N/A"
        content = (
            f"Country: {common}\n"
            f"  Capital:    {capital}\n"
            f"  Population: {pop:,}\n"
            f"  Region:     {region} ({sub})\n"
            f"  Area:       {area} km²\n"
            f"  Currencies: {currencies}\n"
            f"  Languages:  {languages}"
        )
        print(f"[country] ✓ {common}")
        return {"source":"REST Countries","title":f"Country: {common}",
                "url":"https://restcountries.com","content":content}
    except Exception as e:
        print(f"[country] failed: {e}"); return None


# ── Master search runner ──────────────────────────────────────────────
def run_search(query: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()

    def add(item: dict | None):
        if not item: return
        key = item.get("url") or item.get("title","")
        if key and key not in seen:
            seen.add(key)
            results.append(item)

    add(search_crypto(query))         # CoinGecko → CryptoCompare
    add(search_weather(query))        # wttr.in
    add(search_country(query))        # REST Countries
    add(search_google_news(query))    # Google News RSS
    add(search_wikipedia(query))      # Wikipedia full extract

    print(f"[search] total: {len(results)} sources")
    return results


def build_search_context(query: str, results: list[dict]) -> str:
    """
    Build the search context block.
    Injected as a SYSTEM message (not user message) so the model
    treats it as authoritative instructions, not user input.
    """
    now_str = time.strftime("%d %b %Y %H:%M UTC", time.gmtime())

    if not results:
        return (
            f"[SEARCH RESULT]\n"
            f"Query: {query}\n"
            f"Time:  {now_str}\n"
            f"Result: No data found from any source.\n"
            f"Instruction: Tell the user you searched but could not find current data. "
            f"Do NOT say 'I don't have real-time access'."
        )

    lines = [
        f"[LIVE SEARCH DATA — fetched at {now_str}]",
        f"Query: {query}",
        f"Sources found: {len(results)}",
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
        "--- END OF LIVE DATA ---",
        "",
        "INSTRUCTION: The above is real live data fetched seconds ago.",
        "READ the content carefully and use the exact figures in your answer.",
        "NEVER say 'I don't have real-time access' or 'my training data'.",
        "NEVER say 'I cannot provide current prices' — the prices are above.",
        "If the data has the answer, state it directly and cite the source.",
        "If the data does not answer the question, say: 'I searched but couldn't find that specific information.'",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────
IMAGE_GEN_RE = re.compile(
    r'\b(generate|create|make|draw|paint|design|produce|render)\b.{0,25}'
    r'\b(image|picture|photo|illustration|artwork|graphic|wallpaper|logo|'
    r'poster|banner|sketch|portrait|thumbnail)\b|^/imagine\b',
    re.IGNORECASE,
)

IMAGE_DECLINE = (
    "I'm text-only — I can't generate images. 🙅\n\n"
    "Try these free tools:\n"
    "• **[Adobe Firefly](https://firefly.adobe.com)** — free, high quality\n"
    "• **[Microsoft Designer](https://designer.microsoft.com)** — free with Microsoft account\n"
    "• **[Ideogram](https://ideogram.ai)** — great for text in images\n"
    "• **[Craiyon](https://www.craiyon.com)** — completely free, no account\n\n"
    "Want me to write a detailed prompt for any of these? ✍️"
)

# ─────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────
SYSTEM_BASE = """You are Zippy, a smart AI assistant made by Arul Vethathiri.

## IDENTITY
- Text-only AI. You cannot generate images.
- Made by Arul Vethathiri, Class 11 student (2026).

## TONE
- Smart, calm, friendly — like a knowledgeable friend.
- Conversational. Not corporate. Not over-excited.
- Short greetings: one sentence only.

## RULES
1. Answer exactly what was asked. Nothing extra.
2. Prices/numbers → give them in the FIRST sentence.
3. Simple questions → 1-3 sentences max.
4. Explanations → short bullets, no long intro.
5. Code → give it directly, brief comments.
6. Math → show steps briefly.
7. Creative → complete the piece, no preamble.
8. FORBIDDEN phrases: "Certainly!", "Great question!", "As an AI",
   "I don't have real-time access", "my training data", "I'd be happy to".
9. Never repeat the question.
10. End with exactly 1 relevant emoji."""

SYSTEM_WITH_SEARCH = SYSTEM_BASE + """

## LIVE DATA INSTRUCTIONS
You have been given LIVE SEARCH DATA in this message.
- That data was fetched from the internet seconds ago. It is real and current.
- READ it carefully. Use the exact figures and facts in your answer.
- State prices/values directly in your first sentence.
- Cite the source naturally: "According to CoinGecko..." or "Google News reports..."
- NEVER say "I don't have real-time access" — you have live data right there.
- NEVER say "I cannot provide current prices" — the prices are in the data above.
- If the data does not answer the question, say: "I searched but couldn't find that specific info." """

# ─────────────────────────────────────────────────────────────────────
#  STATIC REPLIES
# ─────────────────────────────────────────────────────────────────────
IDENTITY = {
    "who are you":       "I'm Zippy, a text-based AI made by Arul Vethathiri! 🤖",
    "what are you":      "I'm Zippy — a text-only AI made by Arul Vethathiri. 🤖",
    "who made you":      "Arul Vethathiri, a Class 11 student. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "who built you":     "Built by Arul Vethathiri. 👨‍💻",
    "what is your name": "I'm Zippy! 😊",
    "what can you do": (
        "I can answer questions, help with code, explain concepts, do maths, "
        "write content, and search the web for live prices, weather, and news. "
        "I can't generate images. 💬"
    ),
    "are you an ai":  "Yes — Zippy AI, made by Arul Vethathiri. 🤖",
    "are you human":  "Nope, I'm Zippy — an AI, but a capable one! 😄",
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
    "okay":       "Sure! 👍",
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
    status = {
        m["name"]: ("available" if model_cooldown.get(m["id"], 0) <= now
                    else f"cooling — {format_wait(model_cooldown[m['id']] - now)} left")
        for m in CHAT_MODELS
    }
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
                "thinking": "Empty.", "searched": False, "sources": []}

    text_lower = user_input.lower().strip()

    # 2. Image → decline
    if IMAGE_GEN_RE.search(user_input):
        return {"reply": IMAGE_DECLINE, "thinking": "Image request.",
                "searched": False, "sources": [], "model_used": "static"}

    # 3. Static replies
    if text_lower in IDENTITY:
        return {"reply": IDENTITY[text_lower], "thinking": "Identity.",
                "searched": False, "sources": [], "model_used": "static"}
    if text_lower in SOCIAL:
        return {"reply": SOCIAL[text_lower], "thinking": "Social.",
                "searched": False, "sources": [], "model_used": "static"}

    # 4. AI thinking — pure model decision, no rules
    think        = run_thinking(user_input)
    needs_search = think.get("needs_search", False)
    search_query = (think.get("search_query") or "").strip() or user_input
    reasoning    = think.get("reasoning", "")

    # 5. Search if needed
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search:
        results        = run_search(search_query)
        search_context = build_search_context(search_query, results)
        if results:
            searched       = True
            search_sources = [
                {"title": r["title"], "url": r["url"], "source": r["source"]}
                for r in results
            ]

    # 6. Build messages
    #    Key fix: inject search data into the SYSTEM message (authoritative)
    #    AND reference it in the user message (double reinforcement)
    system_content = SYSTEM_WITH_SEARCH if search_context else SYSTEM_BASE

    if search_context:
        # Inject live data directly into the system prompt
        system_content = system_content + f"\n\n{search_context}"

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:1500]})

    # User message: reference the data if it exists
    final_user = (
        f"Using the live data in the system context above, answer this:\n{user_input}"
        if search_context else user_input
    )
    messages.append({"role": "user", "content": final_user})

    # 7. Call AI
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
            "searched":   False, "sources": [], "model_used": "none",
        }
