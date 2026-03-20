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
    available_at = time.time() + retry_after
    model_cooldown[model_id] = available_at
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
            text = resp.choices[0].message.content.strip()
            print(f"[model] ✓ {mid}")
            return text, mid
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
#  SEARCH DECISION
#
#  TWO systems run independently, result combined with OR logic:
#
#  System A — AI model (run_thinking):
#    The model reads the message and decides. It uses context, nuance,
#    and understanding. This is the primary decision maker.
#    Runs in a thread with a 5s timeout to handle cold starts.
#
#  System B — Keyword rules (rules_need_search):
#    Fast pattern matching. Acts as a safety net.
#    Catches obvious cases if the model times out or says "no" wrongly.
#
#  Final decision = model_says_search OR rules_say_search
#  This means: search if EITHER system thinks it's needed.
#  Never miss a search because one system failed.
# ─────────────────────────────────────────────────────────────────────

THINK_PROMPT = """You are the search decision engine for Zippy AI.

A user sent a message. Decide: does answering this need live data from the web?

Your default should lean toward YES — it is better to search and find nothing
than to confidently give the user outdated information.

ALWAYS output YES (needs_search: true) for:
- Any price: crypto, stocks, gold, oil, petrol, forex, USD/INR, dollar rate
- Weather, temperature, humidity, rain, forecast for any place
- Sports: match scores, results, winners, standings, IPL, cricket, football, NBA, F1
- News: anything described as "latest", "recent", "breaking", "today", "now", "this week"
- Elections, political events, government decisions
- "Who is the current X" — president, PM, CEO, champion, etc.
- Software/app versions: "latest version of X", "new update"
- Any question with the year 2024, 2025, or 2026
- Anything that could have changed in the last year

Output NO (needs_search: false) ONLY when you are 100% sure the answer is:
- Timeless: coding syntax, algorithms, math, physics theory, definitions
- Historical and settled: events before 2023 with no ongoing relevance
- Creative: writing poems, stories, code, essays
- Personal advice or opinions
- Greetings, thanks, casual chat

When in doubt → search.

Respond with ONLY this JSON (no markdown, no extra text):
{
  "needs_search": true or false,
  "search_query": "short optimised search query, or empty string if false",
  "reasoning": "one sentence"
}

Good search queries — short and direct:
- "bitcoin price today"
- "weather Chennai now"
- "IPL 2025 winner"
- "India vs Australia cricket score"
- "OpenAI latest news"

JSON only:"""


def run_thinking(user_input: str, timeout: float = 5.0) -> dict | None:
    """
    Ask the AI model to decide if search is needed.
    Returns the parsed dict if model responds in time, or None if it times out.
    None means the caller should fall back to rules.
    """
    result_box: list[dict] = []

    def _call():
        try:
            raw, used = call_with_fallback(
                [{"role": "system", "content": THINK_PROMPT},
                 {"role": "user",   "content": user_input}],
                THINK_MODELS,
                max_tokens=100,
                temperature=0.0,
                top_p=1.0,
            )
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            m   = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if "needs_search" in parsed:
                    parsed["_via"] = used
                    result_box.append(parsed)
        except Exception as e:
            print(f"[thinking] model call failed: {e}")

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result_box:
        r = result_box[0]
        print(f"[thinking] model={r.get('_via')} needs_search={r['needs_search']} "
              f"query='{r.get('search_query','')}' | {r.get('reasoning','')}")
        return r

    print(f"[thinking] model timed out / failed — using rules only")
    return None


# Keywords that ALWAYS mean we should search (safety net only)
_RULE_SEARCH = re.compile(
    r'\b(price|cost|how much|rate|worth|value\s+of'
    r'|bitcoin|btc|ethereum|eth|crypto|dogecoin|solana|bnb|xrp|cardano'
    r'|stock|share|nasdaq|sensex|nifty|nse|bse|gold|silver|oil|petrol|diesel|forex'
    r'|dollar|rupee|usd|inr|eur|gbp'
    r'|weather|temperature|forecast|humidity|rainfall'
    r'|score|result|winner|won|champion|standings|leaderboard'
    r'|ipl|cricket|football|soccer|nba|nfl|f1|formula\s*1|tennis|wimbledon'
    r'|news|headline|breaking|latest news|today\'?s news'
    r'|election|vote|government|president|prime minister|minister|parliament'
    r'|today|right now|currently|at the moment|as of today|this week|this month'
    r'|2024|2025|2026|latest|recent|live|real.?time|new update|new release'
    r'|earthquake|flood|cyclone|storm|disaster|accident|attack|war|conflict)\b',
    re.IGNORECASE,
)

def rules_need_search(user_input: str) -> tuple[bool, str]:
    """
    Keyword-based search detection. Safety net only.
    Returns (should_search, query_string).
    """
    if _RULE_SEARCH.search(user_input):
        q = user_input.strip().rstrip("?.!").strip()
        return True, q
    return False, ""


def decide_search(user_input: str) -> tuple[bool, str, str]:
    """
    Combined decision using OR logic:
      search = model_says_yes  OR  rules_say_yes

    Returns (needs_search, search_query, reasoning).

    The AI model is always asked first.
    Rules are a safety net — they can upgrade a "no" to a "yes"
    but cannot downgrade a model "yes" to a "no".
    """
    # Run model thinking (may return None if it timed out)
    model_result = run_thinking(user_input, timeout=5.0)

    model_search = model_result.get("needs_search", False) if model_result else False
    model_query  = (model_result.get("search_query") or "").strip() if model_result else ""
    model_reason = model_result.get("reasoning", "") if model_result else ""

    rule_search, rule_query = rules_need_search(user_input)

    # OR logic: search if either system says yes
    needs_search = model_search or rule_search

    # Prefer model's query (more intelligent), fall back to rule query, then raw input
    if needs_search:
        search_query = model_query or rule_query or user_input.strip().rstrip("?.!")
        if model_search and not rule_search:
            reasoning = model_reason or "Model decided search is needed."
        elif rule_search and not model_search:
            reasoning = f"Rules detected search keyword (model said no / timed out). {model_reason}".strip()
        else:
            reasoning = model_reason or "Both model and rules agree: search needed."
    else:
        search_query = ""
        reasoning = model_reason or "No live data needed for this question."

    print(f"[decide] model={model_search} rules={rule_search} → search={needs_search} "
          f"query='{search_query}'")
    return needs_search, search_query, reasoning


# ─────────────────────────────────────────────────────────────────────
#  SEARCH SOURCES
#  All free, no API keys, no HTML scraping.
#  Using stable JSON/XML APIs that work reliably on server IPs.
# ─────────────────────────────────────────────────────────────────────
BASE_HEADERS = {"User-Agent": "ZippyAI/2.0 (educational project)"}


# ── 1. RSS News (BBC + Reuters + Times of India) ──────────────────────
#  Best source for current events and news. stdlib xml parser — no package.

NEWS_FEEDS = [
    ("BBC News",        "http://feeds.bbci.co.uk/news/rss.xml"),
    ("BBC Technology",  "http://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("BBC Science",     "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
    ("Reuters",         "https://feeds.reuters.com/reuters/topNews"),
    ("Times of India",  "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
]

def _fetch_rss(url: str, max_items: int = 8) -> list[dict]:
    try:
        r = requests.get(url, headers=BASE_HEADERS, timeout=7)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "").strip()
            desc  = item.findtext("description", "").strip()
            link  = item.findtext("link", "").strip()
            pub   = item.findtext("pubDate", "").strip()
            # Strip HTML tags from description
            desc = re.sub(r'<[^>]+>', '', desc).strip()
            if title and len(title) > 5:
                items.append({
                    "title":   title,
                    "desc":    desc[:300] if desc else "",
                    "link":    link,
                    "pubDate": pub,
                })
        return items
    except Exception as e:
        print(f"[rss] {url} failed: {e}")
        return []

def _score_relevance(item: dict, keywords: list[str]) -> int:
    """Simple relevance score: count keyword hits in title + description."""
    text = (item["title"] + " " + item["desc"]).lower()
    return sum(1 for kw in keywords if kw.lower() in text)

def search_news(query: str, max_results: int = 4) -> dict | None:
    """
    Search BBC/Reuters/TOI RSS for news matching the query.
    Returns the top matching articles.
    """
    keywords = [w for w in re.split(r'\W+', query.lower()) if len(w) > 3]
    if not keywords:
        keywords = query.lower().split()[:4]

    all_items: list[dict] = []

    # Try each feed, collect articles
    for feed_name, feed_url in NEWS_FEEDS:
        items = _fetch_rss(feed_url, max_items=15)
        for item in items:
            item["_feed"] = feed_name
            item["_score"] = _score_relevance(item, keywords)
        all_items.extend(items)

    if not all_items:
        return None

    # Sort by relevance, take top results
    all_items.sort(key=lambda x: x["_score"], reverse=True)
    top = [i for i in all_items if i["_score"] > 0][:max_results]

    # If no keyword matches, take the top headlines as general news
    if not top:
        top = all_items[:max_results]

    if not top:
        return None

    lines = []
    for i, item in enumerate(top, 1):
        lines.append(
            f"{i}. [{item['_feed']}] {item['title']}\n"
            f"   {item['desc']}\n"
            f"   Published: {item.get('pubDate', 'N/A')}\n"
            f"   Link: {item['link']}"
        )

    print(f"[news] ✓ {len(top)} articles matched from RSS feeds")
    return {
        "source":  "Live News RSS (BBC/Reuters/TOI)",
        "title":   f"News results for: {query}",
        "url":     "https://www.bbc.com/news",
        "content": "\n\n".join(lines),
    }


# ── 2. CoinGecko: live crypto prices ─────────────────────────────────
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


# ── 3. wttr.in: live weather ──────────────────────────────────────────
def _detect_city(query: str) -> str | None:
    q = query.lower()
    if not any(w in q for w in {"weather", "temperature", "forecast", "humidity", "rain", "wind", "climate"}):
        return None
    m = re.search(r'weather\s+(?:in|at|for|of)?\s*([A-Za-z][A-Za-z\s]{1,24})', query, re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'([A-Za-z][A-Za-z\s]{1,24})\s+weather', query, re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'(?:temperature|forecast|climate)\s+(?:in|of|at)?\s*([A-Za-z][A-Za-z\s]{1,24})', query, re.IGNORECASE)
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
            desc2 = (day.get("hourly") or [{}])[4].get(
                "weatherDesc", [{}])[0].get("value", "N/A")
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


# ── 4. Wikipedia: full article extract ───────────────────────────────
def search_wikipedia(query: str) -> dict | None:
    try:
        # Search for best title
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
                url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"
                print(f"[wiki] ✓ '{title}' ({len(extract)} chars)")
                return {
                    "source":  "Wikipedia",
                    "title":   title,
                    "url":     url,
                    "content": extract,
                }
    except Exception as e:
        print(f"[wiki-extract] {e}")
    return None


# ── 5. REST Countries ─────────────────────────────────────────────────
def _detect_country(query: str) -> str | None:
    if not any(w in query.lower() for w in
               {"capital", "population", "currency", "language", "country", "nation", "area"}):
        return None
    m = re.search(
        r'(?:of|in|about|for|capital of|population of|currency of)\s+'
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
        c         = data[0]
        common    = c.get("name", {}).get("common", name)
        capital   = ", ".join(c.get("capital", ["N/A"]))
        pop       = c.get("population", 0)
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
    """
    Run ALL applicable search sources.
    Every source is tried independently — a failure in one doesn't stop others.
    """
    results:  list[dict] = []
    seen_keys: set[str]  = set()

    def add(item: dict | None):
        if not item:
            return
        key = item.get("url") or item.get("title", "")
        if key and key not in seen_keys:
            seen_keys.add(key)
            results.append(item)

    add(search_crypto(query))    # live crypto prices
    add(search_weather(query))   # live weather
    add(search_country(query))   # country facts
    add(search_news(query))      # RSS news — BBC, Reuters, TOI
    add(search_wikipedia(query)) # full Wikipedia article

    print(f"[search] total sources collected: {len(results)}")
    return results


def build_search_context(query: str, results: list[dict]) -> str:
    now_str = time.strftime("%d %b %Y %H:%M UTC", time.gmtime())

    if not results:
        return (
            "<search_results>\n"
            f"SEARCHED: {query}\n"
            f"TIME:     {now_str}\n"
            "RESULT:   No data found from any source.\n"
            "INSTRUCTION: Tell the user you searched but found no current data. "
            "Do NOT say 'I don't have real-time access'.\n"
            "</search_results>"
        )

    lines = [
        "<search_results>",
        f"SEARCHED:       {query}",
        f"RETRIEVED AT:   {now_str}",
        f"SOURCES FOUND:  {len(results)}",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"━━━ SOURCE {i}: {r['source']} ━━━",
            f"Title: {r['title']}",
            f"URL:   {r['url']}",
            "",
            r["content"],
            "",
        ]
    lines += [
        "━━━ END OF SEARCH RESULTS ━━━",
        "",
        "CRITICAL INSTRUCTIONS:",
        "- This data is REAL. It was fetched live moments ago.",
        "- Use the exact numbers, names, and facts shown above.",
        "- NEVER say 'I don't have real-time access'.",
        "- NEVER say 'I cannot provide current prices' — prices are shown above.",
        "- Cite sources naturally: 'According to CoinGecko...' or 'BBC reports that...'",
        "- If nothing in the results answers the question, say 'I searched but couldn't find that specific information.'",
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
    "Try these free tools instead:\n"
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
- Text-only AI. You CANNOT generate images.
- Made by Arul Vethathiri, Class 11 student (2026).

## HOW TO USE SEARCH RESULTS
When you see a <search_results> block in the message:
- That data was fetched LIVE from the web moments ago. It is real and current.
- Read the content carefully and use the actual figures and facts in your answer.
- NEVER say "I don't have real-time access" — you have live data right there.
- NEVER say "I cannot provide current prices" — the prices are in the data.
- State numbers directly: "Bitcoin is currently $X according to CoinGecko."
- If the data doesn't answer the question, say: "I searched but couldn't find that."

## TONE
- Smart, calm, friendly — like a knowledgeable friend.
- Natural and conversational. Not corporate or stiff.
- Short greetings: one sentence only.

## RULES
1. Answer exactly what was asked. Nothing extra.
2. Prices/numbers → give them in the FIRST sentence.
3. Simple questions → 1-3 sentences max.
4. Explanations → short bullet points, no long intro.
5. Code → give it directly, minimal comments.
6. Math → show steps briefly.
7. Creative → complete the piece, no preamble.
8. FORBIDDEN phrases: "Certainly!", "Great question!", "As an AI",
   "I don't have real-time access", "my training data", "I'd be happy to".
9. Never repeat the question.
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
        "write content, and search the web for live prices, weather, and news. "
        "I can't generate images though. 💬"
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

    # 2. Image request → decline immediately
    if IMAGE_GEN_RE.search(user_input):
        return {"reply": IMAGE_DECLINE, "thinking": "Image request — text-only AI.",
                "searched": False, "sources": [], "model_used": "static"}

    # 3. Static replies
    if text_lower in IDENTITY:
        return {"reply": IDENTITY[text_lower], "thinking": "Identity.",
                "searched": False, "sources": [], "model_used": "static"}
    if text_lower in SOCIAL:
        return {"reply": SOCIAL[text_lower], "thinking": "Social.",
                "searched": False, "sources": [], "model_used": "static"}

    # 4. Search decision — AI model + rules, OR logic
    needs_search, search_query, reasoning = decide_search(user_input)

    # 5. Search
    search_context = ""
    searched       = False
    search_sources: list[dict] = []

    if needs_search and search_query:
        results = run_search(search_query)
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
        f"{search_context}\n\nNow answer this using the data above:\n{user_input}"
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
            "searched":   False, "sources":    [], "model_used": "none",
        }
