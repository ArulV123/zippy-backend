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
    return f"{math.ceil(seconds / 60)} minute{'s' if math.ceil(seconds/60) != 1 else ''}"


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
#  SEARCH DECISION  (AI model + keyword rules with OR logic)
#
#  The AI model decides first (runs in a thread with 5s timeout).
#  Keyword rules run in parallel as a safety net.
#  Final decision = model_says_yes  OR  rules_say_yes
#  This guarantees search works even on cold start.
# ─────────────────────────────────────────────────────────────────────

THINK_PROMPT = """You are the search decision engine for Zippy AI.

Read the user's message and decide if answering it needs live data from the web.

Default lean: YES — better to search unnecessarily than give stale data.

Always output YES for:
- Any price: crypto, stocks, gold, oil, petrol, forex, currency exchange rates
- Weather / temperature / forecast for any city or region
- Sports: scores, results, match winners, standings, IPL, cricket, football, NBA
- News: anything with "latest", "recent", "breaking", "today", "now", "this week"
- Elections, government decisions, political events
- "Who is the current X" — any current role, position, or title
- Software / app versions: "latest version", "new update", "new release"
- Any question mentioning years 2024, 2025, 2026
- Anything that could have changed in the past year

Output NO only when 100% sure the answer is timeless:
- Pure coding syntax or algorithms
- Math calculations or theory
- Science concepts (gravity, photosynthesis, etc.)
- History before 2023
- Creative writing tasks
- Definitions of stable concepts
- Greetings or casual chat

When in doubt → always search.

Respond with ONLY this JSON (no markdown, no extra text):
{
  "needs_search": true or false,
  "search_query": "short optimised query for search engine, empty string if false",
  "reasoning": "one sentence"
}

Good query examples:
- "bitcoin price today USD INR"
- "weather Chennai today"
- "IPL 2025 final result winner"
- "latest OpenAI news 2025"
- "India prime minister 2025"

JSON only:"""


def run_thinking(user_input: str, timeout: float = 5.0) -> dict | None:
    """Ask AI model to decide. Returns None if it times out."""
    result_box: list[dict] = []

    def _call():
        try:
            raw, used = call_with_fallback(
                [{"role": "system", "content": THINK_PROMPT},
                 {"role": "user",   "content": user_input}],
                THINK_MODELS, max_tokens=100, temperature=0.0, top_p=1.0,
            )
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            m   = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if "needs_search" in parsed:
                    parsed["_via"] = used
                    result_box.append(parsed)
        except Exception as e:
            print(f"[thinking] failed: {e}")

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result_box:
        r = result_box[0]
        print(f"[thinking] model={r.get('_via')} needs_search={r['needs_search']} "
              f"query='{r.get('search_query','')}' | {r.get('reasoning','')}")
        return r
    print(f"[thinking] timed out / failed after {timeout}s")
    return None


# Broad keyword regex — safety net when model times out
_RULE_RE = re.compile(
    r'\b(price|cost|how much|rate|worth|exchange rate'
    r'|bitcoin|btc|ethereum|eth|crypto|dogecoin|doge|solana|sol'
    r'|bnb|xrp|ripple|cardano|ada|litecoin|ltc|shiba|matic|polygon|avax|tron'
    r'|stock|share|market cap|nasdaq|sensex|nifty|nse|bse|dow jones'
    r'|gold|silver|oil|crude|petrol|diesel|fuel'
    r'|dollar|rupee|euro|pound|usd|inr|eur|gbp|forex|currency'
    r'|weather|temperature|forecast|humidity|rainfall|rain today|wind speed'
    r'|score|result|winner|won|champion|final|standings|leaderboard|ranking'
    r'|ipl|cricket|football|soccer|nba|nfl|f1|formula.?1|tennis|wimbledon|wwe'
    r'|news|headline|breaking|latest news|today.?s news|current events'
    r'|election|vote|government|president|prime minister|minister|parliament|cabinet'
    r'|today|right now|currently|at the moment|as of today|this week|this month'
    r'|2024|2025|2026|latest|recent|live|real.?time|new update|new release|just released'
    r'|earthquake|flood|cyclone|storm|disaster|accident|war|conflict|attack)\b',
    re.IGNORECASE,
)

def rules_need_search(user_input: str) -> tuple[bool, str]:
    if _RULE_RE.search(user_input):
        q = user_input.strip().rstrip("?.!").strip()
        return True, q
    return False, ""


def decide_search(user_input: str) -> tuple[bool, str, str]:
    """
    OR logic: search if model says yes OR rules say yes.
    Model runs first in a thread. Rules always run regardless.
    Prefers model's search query (smarter), falls back to rules query.
    """
    model_result              = run_thinking(user_input, timeout=5.0)
    model_search              = model_result["needs_search"]  if model_result else False
    model_query               = (model_result.get("search_query") or "").strip() if model_result else ""
    model_reason              = model_result.get("reasoning", "") if model_result else ""

    rule_search, rule_query   = rules_need_search(user_input)

    needs_search = model_search or rule_search

    if needs_search:
        search_query = model_query or rule_query or user_input.strip().rstrip("?.!")
        if model_search and rule_search:
            reasoning = model_reason or "Both model and rules agree: search needed."
        elif model_search:
            reasoning = model_reason or "Model decided search is needed."
        else:
            reasoning = f"Rules detected live-data keyword (model said no/timed out)."
    else:
        search_query = ""
        reasoning    = model_reason or "No live data needed."

    print(f"[decide] model={model_search} rules={rule_search} "
          f"→ search={needs_search} query='{search_query}'")
    return needs_search, search_query, reasoning


# ─────────────────────────────────────────────────────────────────────
#  SEARCH SOURCES
#  All free, no API keys required, tested to work on server IPs.
# ─────────────────────────────────────────────────────────────────────
BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZippyAI/2.0; +https://github.com/zippy)",
    "Accept":     "application/json",
}

# ── 1. REDDIT  (best free news/current events source, always works) ──
#
#  Reddit's JSON API requires no key and is never blocked on server IPs.
#  Each subreddit has a search endpoint that returns real posts with titles
#  and text content. We search multiple relevant subreddits and merge.

REDDIT_SUBS = [
    "worldnews",       # global news
    "news",            # US/general news
    "technology",      # tech news
    "india",           # India-specific news
    "cricket",         # cricket scores/news
    "soccer",          # football
    "nba",             # basketball
    "stocks",          # stock market
    "CryptoCurrency",  # crypto news
    "science",         # science news
]

def search_reddit(query: str, max_results: int = 5) -> dict | None:
    """
    Search Reddit across relevant subreddits.
    Uses Reddit's public JSON API — no key needed, works on all servers.
    Returns real post titles + selftext content.
    """
    keywords = [w.lower() for w in re.split(r'\W+', query) if len(w) > 2]

    # Pick the most relevant subreddits for this query
    q_lower = query.lower()
    target_subs = []

    if any(w in q_lower for w in ["cricket", "ipl", "bcci", "test match", "odi", "t20"]):
        target_subs = ["cricket", "india", "worldnews"]
    elif any(w in q_lower for w in ["football", "soccer", "premier league", "fifa", "isl"]):
        target_subs = ["soccer", "football", "worldnews"]
    elif any(w in q_lower for w in ["nba", "basketball"]):
        target_subs = ["nba", "sports"]
    elif any(w in q_lower for w in ["crypto", "bitcoin", "ethereum", "btc", "eth", "coin"]):
        target_subs = ["CryptoCurrency", "Bitcoin", "stocks"]
    elif any(w in q_lower for w in ["stock", "share", "market", "sensex", "nifty", "nasdaq"]):
        target_subs = ["stocks", "investing", "IndiaInvestments"]
    elif any(w in q_lower for w in ["india", "modi", "delhi", "mumbai", "bollywood"]):
        target_subs = ["india", "worldnews", "IndiaSpeaks"]
    elif any(w in q_lower for w in ["tech", "ai", "openai", "google", "apple", "software"]):
        target_subs = ["technology", "artificial", "worldnews"]
    else:
        target_subs = ["worldnews", "news", "technology"]

    all_posts: list[dict] = []

    for sub in target_subs[:3]:   # limit to 3 subs to keep it fast
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            r = requests.get(
                url,
                params={"q": query, "sort": "relevance", "t": "month", "limit": 8},
                headers={**BASE_HEADERS, "User-Agent": "ZippyAI/2.0"},
                timeout=8,
            )
            if r.status_code == 200:
                data  = r.json()
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    p = post.get("data", {})
                    title    = p.get("title", "").strip()
                    selftext = p.get("selftext", "").strip()
                    url_post = p.get("url", "")
                    score    = p.get("score", 0)
                    created  = p.get("created_utc", 0)
                    age_days = (time.time() - created) / 86400 if created else 999

                    # Score relevance
                    text_combined = (title + " " + selftext).lower()
                    relevance = sum(1 for kw in keywords if kw in text_combined)

                    if title and relevance > 0:
                        all_posts.append({
                            "title":     title,
                            "body":      selftext[:300] if selftext and selftext != "[removed]" else "",
                            "url":       url_post,
                            "score":     score,
                            "relevance": relevance,
                            "age_days":  age_days,
                            "sub":       sub,
                        })
            print(f"[reddit] r/{sub} → {len(posts) if r.status_code == 200 else 'failed'}")
        except Exception as e:
            print(f"[reddit] r/{sub} failed: {e}")

    if not all_posts:
        # Fallback: just get top posts from r/worldnews
        try:
            r = requests.get(
                "https://www.reddit.com/r/worldnews/hot.json",
                params={"limit": 8},
                headers={**BASE_HEADERS, "User-Agent": "ZippyAI/2.0"},
                timeout=8,
            )
            if r.status_code == 200:
                for post in r.json().get("data", {}).get("children", []):
                    p = post.get("data", {})
                    all_posts.append({
                        "title":     p.get("title", ""),
                        "body":      "",
                        "url":       p.get("url", ""),
                        "score":     p.get("score", 0),
                        "relevance": 1,
                        "age_days":  0,
                        "sub":       "worldnews",
                    })
        except Exception as e:
            print(f"[reddit] fallback failed: {e}")

    if not all_posts:
        return None

    # Sort: high relevance first, then recency, then upvotes
    all_posts.sort(key=lambda x: (-x["relevance"], x["age_days"], -x["score"]))
    top = all_posts[:max_results]

    lines = []
    for i, p in enumerate(top, 1):
        body_str = f"\n   {p['body']}" if p["body"] else ""
        lines.append(
            f"{i}. {p['title']}"
            f"{body_str}\n"
            f"   Source: reddit.com/r/{p['sub']}  |  Score: {p['score']}\n"
            f"   Link: {p['url']}"
        )

    print(f"[reddit] ✓ returning {len(top)} posts")
    return {
        "source":  "Reddit (live community posts)",
        "title":   f"Reddit posts about: {query}",
        "url":     f"https://www.reddit.com/search/?q={requests.utils.quote(query)}&sort=new",
        "content": "\n\n".join(lines),
    }


# ── 2. CRYPTO PRICES  (CoinGecko primary, CryptoCompare fallback) ─────

CRYPTO_MAP = {
    "bitcoin": "bitcoin",      "btc":  "bitcoin",
    "ethereum": "ethereum",    "eth":  "ethereum",
    "dogecoin": "dogecoin",    "doge": "dogecoin",
    "solana": "solana",        "sol":  "solana",
    "bnb": "binancecoin",      "binance coin": "binancecoin",
    "xrp": "ripple",           "ripple": "ripple",
    "cardano": "cardano",      "ada":  "cardano",
    "litecoin": "litecoin",    "ltc":  "litecoin",
    "polkadot": "polkadot",    "dot":  "polkadot",
    "shib": "shiba-inu",       "shiba inu": "shiba-inu",
    "tron": "tron",            "trx":  "tron",
    "avax": "avalanche-2",     "avalanche": "avalanche-2",
    "matic": "matic-network",  "polygon": "matic-network",
    "pepe": "pepe",
}

# CryptoCompare symbol map (different ID format)
CRYPTO_CC_MAP = {
    "bitcoin": "BTC",  "btc":  "BTC",
    "ethereum": "ETH", "eth":  "ETH",
    "dogecoin": "DOGE","doge": "DOGE",
    "solana": "SOL",   "sol":  "SOL",
    "bnb": "BNB",
    "xrp": "XRP",      "ripple": "XRP",
    "cardano": "ADA",  "ada":  "ADA",
    "litecoin": "LTC", "ltc":  "LTC",
    "polkadot": "DOT", "dot":  "DOT",
    "shib": "SHIB",    "shiba": "SHIB",
    "tron": "TRX",     "trx":  "TRX",
    "avax": "AVAX",    "avalanche": "AVAX",
    "matic": "MATIC",  "polygon": "MATIC",
}

def _detect_crypto_names(q: str) -> tuple[list[str], list[str]]:
    """Returns (coingecko_ids, cryptocompare_symbols)."""
    lower = q.lower()
    cg_ids  = list({cg_id  for kw, cg_id  in CRYPTO_MAP.items()    if kw in lower})
    cc_syms = list({sym    for kw, sym     in CRYPTO_CC_MAP.items() if kw in lower})
    return cg_ids, cc_syms

def _coingecko_prices(coin_ids: list[str]) -> dict | None:
    """Primary: CoinGecko free API."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids":                 ",".join(coin_ids),
                "vs_currencies":       "usd,inr",
                "include_24hr_change": "true",
                "include_market_cap":  "true",
                "include_24hr_vol":    "true",
            },
            headers=BASE_HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        lines = []
        for coin_id, info in data.items():
            usd  = info.get("usd",            "N/A")
            inr  = info.get("inr",            "N/A")
            chg  = info.get("usd_24h_change", None)
            mcap = info.get("usd_market_cap", None)
            vol  = info.get("usd_24h_vol",    None)
            usd_str  = f"${usd:,.2f}"  if isinstance(usd,  (int, float)) else str(usd)
            inr_str  = f"₹{inr:,.2f}"  if isinstance(inr,  (int, float)) else str(inr)
            chg_str  = f"{chg:+.2f}%"  if chg  is not None else "N/A"
            mcap_str = f"${mcap:,.0f}" if mcap else "N/A"
            vol_str  = f"${vol:,.0f}"  if vol  else "N/A"
            lines.append(
                f"{coin_id.capitalize()}:\n"
                f"  Price:      {usd_str} USD  /  {inr_str} INR\n"
                f"  24h Change: {chg_str}\n"
                f"  Market Cap: {mcap_str}\n"
                f"  24h Volume: {vol_str}"
            )
        print(f"[crypto] CoinGecko ✓ {coin_ids}")
        return {
            "source":  "CoinGecko (live)",
            "title":   "Live Cryptocurrency Prices",
            "url":     "https://www.coingecko.com",
            "content": "\n\n".join(lines),
        }
    except Exception as e:
        print(f"[crypto] CoinGecko failed: {e}")
        return None

def _cryptocompare_prices(symbols: list[str]) -> dict | None:
    """Fallback: CryptoCompare — no key needed, very generous free tier."""
    try:
        fsyms = ",".join(symbols)
        r = requests.get(
            "https://min-api.cryptocompare.com/data/pricemultifull",
            params={"fsyms": fsyms, "tsyms": "USD,INR"},
            headers=BASE_HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        data = r.json().get("RAW", {})
        if not data:
            return None
        lines = []
        for sym, markets in data.items():
            usd_data = markets.get("USD", {})
            inr_data = markets.get("INR", {})
            price_usd  = usd_data.get("PRICE",          "N/A")
            price_inr  = inr_data.get("PRICE",          "N/A")
            chg_pct    = usd_data.get("CHANGEPCT24HOUR","N/A")
            mcap       = usd_data.get("MKTCAP",         "N/A")
            vol        = usd_data.get("VOLUME24HOURTO",  "N/A")
            usd_str    = f"${price_usd:,.2f}"  if isinstance(price_usd, (int,float)) else str(price_usd)
            inr_str    = f"₹{price_inr:,.2f}"  if isinstance(price_inr, (int,float)) else str(price_inr)
            chg_str    = f"{chg_pct:+.2f}%"    if isinstance(chg_pct,   (int,float)) else str(chg_pct)
            mcap_str   = f"${mcap:,.0f}"        if isinstance(mcap,      (int,float)) else str(mcap)
            vol_str    = f"${vol:,.0f}"          if isinstance(vol,       (int,float)) else str(vol)
            lines.append(
                f"{sym}:\n"
                f"  Price:      {usd_str} USD  /  {inr_str} INR\n"
                f"  24h Change: {chg_str}\n"
                f"  Market Cap: {mcap_str}\n"
                f"  24h Volume: {vol_str}"
            )
        print(f"[crypto] CryptoCompare ✓ {symbols}")
        return {
            "source":  "CryptoCompare (live)",
            "title":   "Live Cryptocurrency Prices",
            "url":     "https://www.cryptocompare.com",
            "content": "\n\n".join(lines),
        }
    except Exception as e:
        print(f"[crypto] CryptoCompare failed: {e}")
        return None

def search_crypto(query: str) -> dict | None:
    """Try CoinGecko first, CryptoCompare as fallback."""
    cg_ids, cc_syms = _detect_crypto_names(query)
    if not cg_ids:
        return None

    result = _coingecko_prices(cg_ids)
    if result:
        return result

    # CoinGecko failed — try CryptoCompare
    if cc_syms:
        return _cryptocompare_prices(cc_syms)
    return None


# ── 3. WEATHER (wttr.in) ──────────────────────────────────────────────
def _detect_city(query: str) -> str | None:
    q = query.lower()
    if not any(w in q for w in {"weather","temperature","forecast","humidity","rain","wind","climate","hot","cold"}):
        return None
    for pat in [
        r'weather\s+(?:in|at|for|of)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'([A-Za-z][A-Za-z\s]{1,24})\s+weather',
        r'(?:temperature|forecast|climate)\s+(?:in|of|at)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'(?:how|what).{0,15}(?:hot|cold|warm|rain).{0,10}in\s+([A-Za-z][A-Za-z\s]{1,24})',
    ]:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            city = m.group(1).strip()
            # Remove trailing question/filler words
            city = re.sub(r'\b(today|now|currently|like|is|the|a|an)\b', '', city, flags=re.IGNORECASE).strip()
            if len(city) > 1:
                return city
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
        vis      = cur.get("visibility",    "N/A")
        forecast = []
        for day in d.get("weather", [])[:3]:
            date  = day.get("date", "")
            maxc  = day.get("maxtempC", "N/A")
            minc  = day.get("mintempC", "N/A")
            desc2 = (day.get("hourly") or [{}])[4].get(
                "weatherDesc", [{}])[0].get("value", "N/A")
            forecast.append(f"  {date}: {desc2}, {minc}°C – {maxc}°C")
        content = (
            f"Current weather in {city}:\n"
            f"  Condition:   {desc}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  Feels like:  {feels}°C\n"
            f"  Humidity:    {humidity}%\n"
            f"  Wind speed:  {wind} km/h\n"
            f"  Visibility:  {vis} km\n"
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


# ── 4. WIKIPEDIA (full article extract) ──────────────────────────────
def search_wikipedia(query: str) -> dict | None:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "srlimit": 1, "format": "json", "origin": "*"},
            headers=BASE_HEADERS, timeout=7,
        )
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]
    except Exception as e:
        print(f"[wiki-search] {e}")
        return None
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "prop": "extracts", "exintro": False,
                    "explaintext": True, "titles": title, "format": "json",
                    "origin": "*", "exchars": 3000},
            headers=BASE_HEADERS, timeout=8,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract and len(extract) > 80:
                url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ','_'))}"
                print(f"[wiki] ✓ '{title}' ({len(extract)} chars)")
                return {"source": "Wikipedia", "title": title,
                        "url": url, "content": extract}
    except Exception as e:
        print(f"[wiki-extract] {e}")
    return None


# ── 5. REST Countries ─────────────────────────────────────────────────
def _detect_country(query: str) -> str | None:
    if not any(w in query.lower() for w in
               {"capital","population","currency","language","country","nation","area","gdp"}):
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
        c         = data[0]
        common    = c.get("name", {}).get("common", name)
        capital   = ", ".join(c.get("capital", ["N/A"]))
        pop       = c.get("population", 0)
        region    = c.get("region", "N/A")
        subregion = c.get("subregion", "N/A")
        area      = c.get("area", "N/A")
        currencies = ", ".join(
            f"{v.get('name','')} ({v.get('symbol','')})"
            for v in c.get("currencies", {}).values()) or "N/A"
        languages  = ", ".join(c.get("languages", {}).values()) or "N/A"
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
        return {"source": "REST Countries", "title": f"Country: {common}",
                "url": "https://restcountries.com", "content": content}
    except Exception as e:
        print(f"[country] failed: {e}")
        return None


# ── Master search runner ──────────────────────────────────────────────
def run_search(query: str) -> list[dict]:
    results:   list[dict] = []
    seen_keys: set[str]   = set()

    def add(item: dict | None):
        if not item:
            return
        key = item.get("url") or item.get("title", "")
        if key and key not in seen_keys:
            seen_keys.add(key)
            results.append(item)

    add(search_crypto(query))    # CoinGecko → CryptoCompare fallback
    add(search_weather(query))   # wttr.in
    add(search_country(query))   # REST Countries
    add(search_reddit(query))    # Reddit JSON — news & current events
    add(search_wikipedia(query)) # Wikipedia full extract

    print(f"[search] total sources: {len(results)}")
    return results


def build_search_context(query: str, results: list[dict]) -> str:
    now_str = time.strftime("%d %b %Y %H:%M UTC", time.gmtime())
    if not results:
        return (
            "<search_results>\n"
            f"SEARCHED: {query}\n"
            f"TIME: {now_str}\n"
            "RESULT: No data found.\n"
            "INSTRUCTION: Tell the user you searched but found nothing. "
            "Do NOT say 'I don't have real-time access'.\n"
            "</search_results>"
        )
    lines = [
        "<search_results>",
        f"SEARCHED:     {query}",
        f"RETRIEVED AT: {now_str}",
        f"SOURCES:      {len(results)}",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"━━━ SOURCE {i}: {r['source']} ━━━",
            f"Title:  {r['title']}",
            f"URL:    {r['url']}",
            "",
            r["content"],
            "",
        ]
    lines += [
        "━━━ END ━━━",
        "",
        "CRITICAL INSTRUCTIONS FOR AI:",
        "- This data is REAL and was fetched LIVE seconds ago.",
        "- Use the exact prices, names, and facts shown above.",
        "- NEVER say 'I don't have real-time access'.",
        "- NEVER say 'I cannot provide current prices' — they are above.",
        "- Cite naturally: 'According to CoinGecko, bitcoin is...'",
        "- If the data doesn't answer, say 'I searched but couldn't find that specific info.'",
        "</search_results>",
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
SYSTEM = """You are Zippy, a smart AI assistant made by Arul Vethathiri.

## IDENTITY
- Text-only AI. You cannot generate images.
- Made by Arul Vethathiri, Class 11 student (2026).

## HOW TO USE SEARCH RESULTS
When you see a <search_results> block:
- That data was fetched LIVE from the web seconds ago. It is real.
- Use the exact numbers, prices, and facts in your answer.
- NEVER say "I don't have real-time access" — you have the data right there.
- NEVER say "I cannot provide current prices" — they are shown above.
- State answers directly: "Bitcoin is currently $X per CoinGecko."
- If data doesn't answer the question: "I searched but couldn't find that."

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
8. FORBIDDEN: "Certainly!", "Great question!", "As an AI",
   "I don't have real-time access", "my training data", "I'd be happy to".
9. Never repeat the question.
10. End with exactly 1 relevant emoji."""

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
                "thinking": "Empty input.", "searched": False, "sources": []}

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

    # 4. Decide search — AI model + rules, OR logic
    needs_search, search_query, reasoning = decide_search(user_input)

    # 5. Search
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search and search_query:
        results        = run_search(search_query)
        search_context = build_search_context(search_query, results)
        if results:
            searched       = True
            search_sources = [
                {"title": r["title"], "url": r["url"], "source": r["source"]}
                for r in results
            ]

    # 6. Build messages
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:1500]})
    final_user = (
        f"{search_context}\n\nAnswer this using the data above:\n{user_input}"
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
