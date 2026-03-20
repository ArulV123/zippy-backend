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
#  GROQ CLIENT + WARM-UP
#  Fire a tiny warm-up request at startup so the connection is ready
#  before the first real user request arrives (solves cold-start timeout)
# ─────────────────────────────────────────────────────────────────────
_groq_client: Groq | None = None

def get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

def _warm_up():
    try:
        time.sleep(2)
        get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print("[warmup] ✓ Groq connection ready")
    except Exception as e:
        print(f"[warmup] skipped: {e}")

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
#  DYNAMIC DATE/TIME  — recomputed on every request
#  Injected into every prompt so the AI never assumes wrong year/date
# ─────────────────────────────────────────────────────────────────────
def now_str() -> str:
    return time.strftime("%A, %d %B %Y", time.gmtime())

def now_utc() -> str:
    return time.strftime("%d %b %Y %H:%M UTC", time.gmtime())


# ─────────────────────────────────────────────────────────────────────
#  CORE FALLBACK CALLER
# ─────────────────────────────────────────────────────────────────────
def call_with_fallback(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 700,
    temperature: float = 0.65,
    top_p: float = 0.9,
) -> tuple[str, str]:
    tried: list[str] = []
    for model in models:
        mid = model["id"]
        if not is_available(mid):
            wait = model_cooldown[mid] - time.time()
            tried.append(f"{model['name']} — cooling ({format_wait(wait)} left)")
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


def call_patient(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 700,
    temperature: float = 0.65,
    top_p: float = 0.9,
    max_wait: float = 50.0,
) -> tuple[str, str]:
    """
    Like call_with_fallback but if ALL models are rate-limited and
    the soonest reset is within max_wait seconds, waits then retries.
    This ensures a response is generated at any cost (no silent failure).
    """
    try:
        return call_with_fallback(messages, models, max_tokens, temperature, top_p)
    except RuntimeError as e:
        if "QUOTA_EXCEEDED" not in str(e):
            raise
        soonest = earliest_available_in(models)
        if 0 < soonest <= max_wait:
            print(f"[patient] all models cooling — waiting {soonest:.0f}s for reset...")
            time.sleep(soonest + 1)
            return call_with_fallback(messages, models, max_tokens, temperature, top_p)
        raise


# ─────────────────────────────────────────────────────────────────────
#  THINKING STEP  —  pure AI model, no keyword rules at all
# ─────────────────────────────────────────────────────────────────────
def _make_think_prompt() -> str:
    today = now_str()
    return f"""You are the search decision engine for Zippy AI.
Today's date is {today}.

Read the user's message and decide: does answering it need LIVE data fetched from the internet?

Output ONLY raw JSON — no markdown, no extra text:
{{
  "needs_search": true or false,
  "search_query": "optimised search string, or empty string if false",
  "reasoning": "one sentence"
}}

━━━ SEARCH NEEDED (needs_search: true) ━━━
Anything that changes over time and your training data may be stale:

• Prices: crypto (bitcoin, eth, doge...), stocks, gold, silver, platinum, oil, petrol, diesel
• Currency exchange rates: USD to INR, dollar rate, forex
• Weather or forecast for any city
• Sports: live scores, match results, tournament winners, IPL, cricket, football, NBA, F1
• Recent news: "latest", "breaking", "what happened", "current events", "today's news"
• Elections, government changes, political events
• "Who is the current president/PM/CEO/champion"
• Software/app/product releases: "latest version", "new update", "just released"
• Any question with years 2024, 2025, 2026 about events/results
• Historical events from 2023–2026 that may not be in training data
• Anything described as "today", "right now", "currently", "this week", "this month"
• "What is happening with X" — ongoing situations
• Company news, mergers, launches, controversies

━━━ NO SEARCH NEEDED (needs_search: false) ━━━
Timeless knowledge that does not change:

• Coding: syntax, algorithms, "how to write X in Python", debugging, data structures
• Math problems: calculations, equations, proofs
• Science theory: physics, chemistry, biology concepts
• Stable definitions: "what is recursion", "explain sorting"
• History before 2023 (well-documented events)
• Creative writing: poems, stories, essays, code generation
• Grammar, language, translation
• Personal advice, opinions, recommendations
• Greetings and normal conversation: "hi", "how are you", "tell me a joke"
• "Explain X", "what does X mean", "how does X work" (conceptual questions)

━━━ SEARCH QUERY TIPS ━━━
Short, specific, optimised for search engines:
• "bitcoin price today USD INR" (not "what is the price of bitcoin")
• "silver price India per kg today"
• "Chennai weather today"
• "IPL 2025 final winner result"
• "USD to INR exchange rate today"
• "latest AI news {today}"

JSON only. Nothing else."""


def run_thinking(user_input: str, timeout: float = 9.0) -> dict:
    """Pure AI thinking with hard timeout. No rules anywhere."""
    result_box: list[dict] = []

    def _call():
        try:
            raw, used = call_with_fallback(
                [{"role": "system", "content": _make_think_prompt()},
                 {"role": "user",   "content": user_input}],
                THINK_MODELS, max_tokens=130, temperature=0.0, top_p=1.0,
            )
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            m   = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed.get("needs_search"), bool):
                    parsed["_via"] = used
                    result_box.append(parsed)
        except Exception as e:
            print(f"[think] error: {e}")

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result_box:
        r = result_box[0]
        print(f"[think] via={r.get('_via','?')} search={r['needs_search']} "
              f"q='{r.get('search_query','')}' — {r.get('reasoning','')}")
        return r

    print(f"[think] timed out after {timeout}s — defaulting no-search")
    return {"needs_search": False, "search_query": "", "reasoning": "Thinking timed out."}


# ─────────────────────────────────────────────────────────────────────
#  SEARCH SOURCES
# ─────────────────────────────────────────────────────────────────────
BASE_H = {
    "User-Agent": "Mozilla/5.0 (compatible; ZippyAI/2.0)",
    "Accept":     "application/json, text/html, */*",
}

# ── Shared: fetch USD/INR rate once per request ───────────────────────
_fx_cache: dict = {"rate": None, "ts": 0.0}

def get_usd_inr() -> float:
    """Live USD→INR rate. Cached for 5 minutes."""
    if time.time() - _fx_cache["ts"] < 300 and _fx_cache["rate"]:
        return _fx_cache["rate"]
    try:
        r = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            headers=BASE_H, timeout=6,
        )
        r.raise_for_status()
        rate = float(r.json()["rates"]["INR"])
        _fx_cache.update({"rate": rate, "ts": time.time()})
        print(f"[forex] USD/INR = {rate}")
        return rate
    except Exception as e:
        print(f"[forex] rate fetch failed: {e}")
        return _fx_cache.get("rate") or 83.5  # fallback


def _fmt_usd(val) -> str:
    if not isinstance(val, (int, float)): return "N/A"
    return f"${val:,.2f}" if val >= 1 else f"${val:.6f}"

def _fmt_inr(val) -> str:
    if not isinstance(val, (int, float)): return "N/A"
    return f"₹{val:,.2f}"


# ── 1. CRYPTO (CoinGecko → CryptoCompare) ───────────────────────────
CRYPTO_CG = {
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
    "pepe": "pepe",
}
CRYPTO_CC = {
    "bitcoin": "BTC",  "btc":  "BTC",  "ethereum": "ETH",  "eth":  "ETH",
    "dogecoin": "DOGE","doge": "DOGE", "solana":   "SOL",  "sol":  "SOL",
    "bnb": "BNB",      "xrp":  "XRP",  "ripple":   "XRP",  "cardano": "ADA",
    "ada": "ADA",      "litecoin": "LTC","ltc": "LTC",     "polkadot": "DOT",
    "dot": "DOT",      "shib": "SHIB", "shiba": "SHIB",    "tron": "TRX",
    "trx": "TRX",      "avax": "AVAX", "avalanche": "AVAX","matic": "MATIC",
    "polygon": "MATIC",
}

def _detect_crypto(q: str):
    lo = q.lower()
    cg = list({v for k,v in CRYPTO_CG.items() if k in lo})
    cc = list({v for k,v in CRYPTO_CC.items() if k in lo})
    return cg, cc

def search_crypto(query: str) -> dict | None:
    cg, cc = _detect_crypto(query)
    if not cg: return None

    # CoinGecko primary
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(cg), "vs_currencies": "usd,inr",
                    "include_24hr_change": "true", "include_market_cap": "true",
                    "include_24hr_vol": "true"},
            headers=BASE_H, timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            lines = []
            for cid, info in data.items():
                usd  = info.get("usd");  inr = info.get("inr")
                chg  = info.get("usd_24h_change")
                mcap = info.get("usd_market_cap"); vol = info.get("usd_24h_vol")
                lines.append(
                    f"• **{cid.capitalize()}** — as of {now_utc()}\n"
                    f"  Price:      {_fmt_usd(usd)} USD  /  {_fmt_inr(inr)} INR\n"
                    f"  24h Change: {f'{chg:+.2f}%' if chg is not None else 'N/A'}\n"
                    f"  Market Cap: {_fmt_usd(mcap) if mcap else 'N/A'}\n"
                    f"  24h Volume: {_fmt_usd(vol) if vol else 'N/A'}"
                )
            print(f"[crypto] CoinGecko ✓")
            return {"source": "CoinGecko (live)", "title": "Live Crypto Prices",
                    "url": "https://www.coingecko.com", "content": "\n\n".join(lines)}
    except Exception as e:
        print(f"[crypto] CoinGecko failed: {e}")

    # CryptoCompare fallback
    if cc:
        try:
            r = requests.get(
                "https://min-api.cryptocompare.com/data/pricemultifull",
                params={"fsyms": ",".join(cc), "tsyms": "USD,INR"},
                headers=BASE_H, timeout=8,
            )
            r.raise_for_status()
            raw = r.json().get("RAW", {})
            if raw:
                lines = []
                for sym, markets in raw.items():
                    u = markets.get("USD", {}); n = markets.get("INR", {})
                    pu = u.get("PRICE"); pi = n.get("PRICE")
                    chg = u.get("CHANGEPCT24HOUR")
                    lines.append(
                        f"• **{sym}** — as of {now_utc()}\n"
                        f"  Price:      {_fmt_usd(pu)} USD  /  {_fmt_inr(pi)} INR\n"
                        f"  24h Change: {f'{chg:+.2f}%' if isinstance(chg,(int,float)) else 'N/A'}"
                    )
                print(f"[crypto] CryptoCompare ✓")
                return {"source": "CryptoCompare (live)", "title": "Live Crypto Prices",
                        "url": "https://www.cryptocompare.com", "content": "\n\n".join(lines)}
        except Exception as e:
            print(f"[crypto] CryptoCompare failed: {e}")
    return None


# ── 2. METALS (gold / silver / platinum / palladium) ─────────────────
METALS_WORDS = {"gold","silver","platinum","palladium","metal","bullion"}

def _detect_metals(q: str) -> list[str]:
    lo = q.lower()
    return [m for m in ["gold","silver","platinum","palladium"] if m in lo]

def search_metals(query: str) -> dict | None:
    metals = _detect_metals(query)
    if not metals: return None
    try:
        r = requests.get(
            "https://api.metals.live/v1/spot",
            headers=BASE_H, timeout=8,
        )
        r.raise_for_status()
        spot = r.json()
        # spot is a list of dicts: [{"gold": 2340.5, "silver": 29.8, ...}]
        if isinstance(spot, list) and spot:
            spot = spot[0]
        elif not isinstance(spot, dict):
            return None

        usd_inr = get_usd_inr()
        TROY_OZ_TO_GRAM = 31.1035

        lines = []
        for metal in metals:
            price_usd_per_oz = spot.get(metal)
            if price_usd_per_oz is None:
                continue
            price_usd_per_oz = float(price_usd_per_oz)
            price_inr_per_oz = price_usd_per_oz * usd_inr

            # Conversions (common Indian units)
            per_gram_usd     = price_usd_per_oz / TROY_OZ_TO_GRAM
            per_gram_inr     = price_inr_per_oz / TROY_OZ_TO_GRAM
            per_10g_usd      = per_gram_usd  * 10
            per_10g_inr      = per_gram_inr  * 10
            per_100g_inr     = per_gram_inr  * 100
            per_kg_inr       = per_gram_inr  * 1000

            lines.append(
                f"• **{metal.capitalize()}** (spot price, as of {now_utc()})\n"
                f"  International (per troy oz):  {_fmt_usd(price_usd_per_oz)} USD\n"
                f"  International (per troy oz):  {_fmt_inr(price_inr_per_oz)} INR\n"
                f"  India — per gram:  {_fmt_inr(per_gram_inr)}\n"
                f"  India — per 10g:   {_fmt_inr(per_10g_inr)}\n"
                f"  India — per 100g:  {_fmt_inr(per_100g_inr)}\n"
                f"  India — per kg:    {_fmt_inr(per_kg_inr)}\n"
                f"  (USD/INR rate used: {usd_inr:.2f})"
            )

        if not lines: return None
        print(f"[metals] ✓ {metals}")
        return {
            "source":  "metals.live + open.er-api.com (live spot prices)",
            "title":   "Live Precious Metal Prices",
            "url":     "https://metals.live",
            "content": "\n\n".join(lines),
        }
    except Exception as e:
        print(f"[metals] failed: {e}")
        return None


# ── 3. FOREX / CURRENCY RATES ─────────────────────────────────────────
FOREX_WORDS = {"forex","exchange rate","currency","dollar","rupee","euro","pound",
               "usd","inr","eur","gbp","jpy","cny","aud","cad","sgd","aed",
               "to inr","to usd","to rupee","to dollar"}

def _detect_forex(q: str) -> tuple[str, str] | None:
    """Detect currency pair from query. Returns (from, to) or None."""
    lo = q.lower()
    if not any(w in lo for w in FOREX_WORDS):
        return None

    # Map common names to codes
    name_map = {
        "dollar": "USD", "usd": "USD", "us dollar": "USD",
        "rupee":  "INR", "inr": "INR", "indian rupee": "INR",
        "euro":   "EUR", "eur": "EUR",
        "pound":  "GBP", "gbp": "GBP", "sterling": "GBP",
        "yen":    "JPY", "jpy": "JPY",
        "yuan":   "CNY", "cny": "CNY",
        "aud":    "AUD", "australian": "AUD",
        "cad":    "CAD", "canadian": "CAD",
        "sgd":    "SGD", "singapore": "SGD",
        "aed":    "AED", "dirham": "AED",
    }

    # Try to extract "X to Y" pattern
    m = re.search(r'(\w+)\s+to\s+(\w+)', lo)
    if m:
        frm = name_map.get(m.group(1).lower())
        to  = name_map.get(m.group(2).lower())
        if frm and to:
            return frm, to

    # Default: if INR mentioned with any foreign currency, return that pair
    for name, code in name_map.items():
        if name in lo and code != "INR":
            return code, "INR"

    return None

def search_forex(query: str) -> dict | None:
    pair = _detect_forex(query)
    if not pair: return None
    frm, to = pair
    try:
        r = requests.get(
            f"https://open.er-api.com/v6/latest/{frm}",
            headers=BASE_H, timeout=7,
        )
        r.raise_for_status()
        data  = r.json()
        rates = data.get("rates", {})
        update_time = data.get("time_last_update_utc", now_utc())

        # Show the main pair + a few common ones
        pairs_to_show = [to] + [c for c in ["USD","INR","EUR","GBP","JPY","AED"]
                                 if c != frm and c != to][:4]
        lines = [f"Base currency: {frm} — as of {update_time}"]
        for currency in pairs_to_show:
            rate = rates.get(currency)
            if rate:
                lines.append(f"  1 {frm} = {rate:,.4f} {currency}")

        print(f"[forex] ✓ {frm} rates")
        return {
            "source":  "Open Exchange Rates (live)",
            "title":   f"Exchange Rate: {frm}",
            "url":     "https://open.er-api.com",
            "content": "\n".join(lines),
        }
    except Exception as e:
        print(f"[forex] failed: {e}")
        return None


# ── 4. WEATHER ────────────────────────────────────────────────────────
def _detect_city(query: str) -> str | None:
    q = query.lower()
    if not any(w in q for w in {"weather","temperature","forecast","humidity",
                                 "rain","wind","climate","hot","cold","sunny"}):
        return None
    for pat in [
        r'weather\s+(?:in|at|for|of)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'([A-Za-z][A-Za-z\s]{1,24})\s+weather',
        r'(?:temperature|forecast)\s+(?:in|of|at)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'how.{0,10}(?:hot|cold|warm).{0,10}in\s+([A-Za-z][A-Za-z\s]{1,20})',
    ]:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            city = re.sub(r'\b(today|now|currently|like|is|the|a|an|going|to|be)\b',
                          '', m.group(1), flags=re.IGNORECASE).strip()
            if len(city) > 1: return city
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
            f"Current weather in {city} — {now_utc()}:\n"
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


# ── 5. NEWS  (Google News RSS primary, fallbacks on failure) ─────────
#
#  Strategy: try multiple sources and merge.
#  Google News RSS → Al Jazeera → Times of India → The Hindu
#  All are free XML feeds that work on server IPs.

def _parse_rss(url: str, source_name: str, keywords: list[str],
               max_items: int = 10) -> list[dict]:
    try:
        r = requests.get(
            url,
            headers={**BASE_H, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
            timeout=8,
        )
        r.raise_for_status()
        root  = ET.fromstring(r.content)
        items = root.findall(".//item")[:max_items]
        results = []
        for item in items:
            title   = (item.findtext("title")        or "").strip()
            link    = (item.findtext("link")         or "").strip()
            pub     = (item.findtext("pubDate")      or "").strip()
            source  = (item.findtext("source")       or source_name).strip()
            desc    = re.sub(r'<[^>]+>', '',
                      (item.findtext("description") or "")).strip()

            # Strip publisher suffix from Google News titles: "Title - Publisher"
            title_clean = re.sub(r'\s*-\s*[^-]{3,40}$', '', title).strip() or title

            # Relevance score
            score = sum(1 for kw in keywords
                        if kw.lower() in (title + desc).lower())

            if title_clean:
                results.append({
                    "title":   title_clean,
                    "desc":    desc[:280],
                    "pub":     pub,
                    "source":  source,
                    "link":    link,
                    "score":   score,
                })
        return results
    except Exception as e:
        print(f"[rss] {source_name} failed: {e}")
        return []


NEWS_FEEDS = [
    # Primary — Google News searches across ALL publishers
    ("Google News",
     "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q={query}"),
    # India-specific
    ("Times of India",
     "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("The Hindu",
     "https://www.thehindu.com/news/feeder/default.rss"),
    # International
    ("Al Jazeera",
     "https://www.aljazeera.com/xml/rss/all.xml"),
]


def search_news(query: str, max_results: int = 6) -> dict | None:
    keywords = [w for w in re.split(r'\W+', query.lower()) if len(w) > 2]

    all_articles: list[dict] = []

    for name, url_tmpl in NEWS_FEEDS:
        url = url_tmpl.replace("{query}", requests.utils.quote(query))
        articles = _parse_rss(url, name, keywords, max_items=15)
        all_articles.extend(articles)
        print(f"[news] {name} → {len(articles)} articles")

    if not all_articles:
        return None

    # Deduplicate by title similarity, sort by relevance then recency
    seen_titles: set[str] = set()
    unique: list[dict] = []
    for a in all_articles:
        key = re.sub(r'\W+', '', a["title"].lower())[:40]
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    unique.sort(key=lambda x: (-x["score"], x["title"]))
    top = unique[:max_results]

    # Format as bullet points (AI will present them as bullets)
    lines = [f"News results for '{query}' — fetched {now_utc()}:\n"]
    for a in top:
        bullet = f"• **{a['title']}**"
        if a["desc"]:
            bullet += f"\n  {a['desc']}"
        bullet += f"\n  Source: {a['source']}"
        if a["pub"]:
            bullet += f"  |  {a['pub']}"
        if a["link"]:
            bullet += f"\n  Link: {a['link']}"
        lines.append(bullet)

    print(f"[news] ✓ {len(top)} unique articles returned")
    return {
        "source":  "Google News / Al Jazeera / TOI / The Hindu (live RSS)",
        "title":   f"Latest News: {query}",
        "url":     f"https://news.google.com/search?q={requests.utils.quote(query)}",
        "content": "\n\n".join(lines),
    }


# ── 6. WIKIPEDIA ──────────────────────────────────────────────────────
def search_wikipedia(query: str) -> dict | None:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":query,
                    "srlimit":1,"format":"json","origin":"*"},
            headers=BASE_H, timeout=7,
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
            headers=BASE_H, timeout=8,
        )
        r.raise_for_status()
        for page in r.json().get("query",{}).get("pages",{}).values():
            extract = page.get("extract","")
            if extract and len(extract) > 80:
                url = ("https://en.wikipedia.org/wiki/"
                       + requests.utils.quote(title.replace(' ','_')))
                print(f"[wiki] ✓ '{title}' ({len(extract)} chars)")
                return {"source":"Wikipedia","title":title,
                        "url":url,"content":extract}
    except Exception as e:
        print(f"[wiki-extract] {e}")
    return None


# ── 7. REST COUNTRIES ─────────────────────────────────────────────────
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
            headers=BASE_H, timeout=7,
        )
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict): return None
        c = data[0]
        common = c.get("name",{}).get("common", name)
        content = (
            f"Country: {common}\n"
            f"  Capital:    {', '.join(c.get('capital',['N/A']))}\n"
            f"  Population: {c.get('population',0):,}\n"
            f"  Region:     {c.get('region','N/A')} ({c.get('subregion','N/A')})\n"
            f"  Area:       {c.get('area','N/A')} km²\n"
            f"  Currencies: {', '.join(f'{v.get(\"name\",\"\")} ({v.get(\"symbol\",\"\")})' for v in c.get('currencies',{}).values()) or 'N/A'}\n"
            f"  Languages:  {', '.join(c.get('languages',{}).values()) or 'N/A'}"
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
            seen.add(key); results.append(item)

    add(search_crypto(query))    # CoinGecko → CryptoCompare
    add(search_metals(query))    # metals.live + open.er-api.com
    add(search_forex(query))     # open.er-api.com exchange rates
    add(search_weather(query))   # wttr.in
    add(search_country(query))   # REST Countries
    add(search_news(query))      # Google News + AJ + TOI + Hindu RSS
    add(search_wikipedia(query)) # Wikipedia full extract

    print(f"[search] total sources: {len(results)}")
    return results


def build_search_context(query: str, results: list[dict]) -> str:
    """
    Injected into the system prompt — model treats it as authoritative data.
    """
    ts = now_utc()
    if not results:
        return (
            f"[SEARCH RESULT — {ts}]\n"
            f"Query: {query}\n"
            "Result: No live data found from any source.\n"
            "Instruction: Tell the user you searched but found nothing. "
            "Do NOT say 'I don't have real-time access'."
        )
    lines = [
        f"[LIVE DATA — fetched at {ts}]",
        f"Query:   {query}",
        f"Sources: {len(results)}",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"=== SOURCE {i}: {r['source']} ===",
            f"Title: {r['title']}",
            f"URL:   {r['url']}",
            "",
            r["content"],
            "",
        ]
    lines += [
        "=== END OF LIVE DATA ===",
        "",
        "MANDATORY INSTRUCTIONS:",
        "- This data is REAL. It was fetched live seconds ago.",
        "- Use the exact prices, dates, and facts shown above in your reply.",
        "- State today's date correctly — it is " + now_str() + ".",
        "- NEVER say 'I don't have real-time access'.",
        "- NEVER say 'I cannot provide current prices' — they are above.",
        "- Cite sources naturally: 'According to CoinGecko...' or 'Google News reports...'",
        "- For news: present each item as a bullet point.",
        "- If data doesn't answer the question: say 'I searched but couldn't find that.'",
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
#  SYSTEM PROMPT  (dynamic date always injected)
# ─────────────────────────────────────────────────────────────────────
def make_system(search_context: str = "") -> str:
    today    = now_str()
    time_utc = now_utc()

    base = f"""You are Zippy, a smart AI assistant made by Arul Vethathiri.

## CURRENT DATE & TIME
Today is {today}. Current time: {time_utc}.
ALWAYS use this date when answering. NEVER assume any other year or date.

## IDENTITY
- Text-only AI. You cannot generate images.
- Made by Arul Vethathiri, Class 11 student (2026).

## TONE
- Smart, calm, friendly — like a knowledgeable friend.
- Conversational. Not corporate. Not over-excited.
- Short greetings: one sentence only.

## RESPONSE FORMAT RULES
1. Answer exactly what was asked. Nothing extra.
2. Prices/numbers → give them in the VERY FIRST sentence.
3. Simple questions → 1-3 sentences max.
4. **News / current events → ALWAYS use bullet points** (one bullet per story).
5. Explanations → short bullets or brief paragraphs depending on complexity.
6. Code → give it directly, brief comments.
7. Math → show steps briefly.
8. Creative → complete the piece, no preamble.
9. FORBIDDEN: "Certainly!", "Great question!", "As an AI",
   "I don't have real-time access", "my training data is limited",
   "I cannot provide current prices", "I'd be happy to".
10. Never repeat the question back.
11. End with exactly 1 relevant emoji."""

    if not search_context:
        return base

    return base + f"""

## LIVE DATA — READ THIS CAREFULLY
You have been given live data fetched from the internet right now.

{search_context}

## HOW TO USE THE LIVE DATA ABOVE
- READ the content in the sources carefully.
- Use the EXACT prices, figures, and facts shown.
- Today is {today} — state this if asked about dates.
- NEVER say "I don't have real-time access" — you have live data above.
- NEVER say "I cannot provide current prices" — they are in the data.
- For prices: state the number immediately in your first sentence.
- For news: present each headline as a bullet point.
- Cite sources naturally: "According to CoinGecko..." or "Google News reports..."
- If data is present but doesn't answer the question: say "I searched but couldn't find that specific info." """


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
    return {
        "status":  "Zippy backend is running!",
        "date":    now_str(),
        "models":  status,
    }


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

    # 3. Static replies (no model call)
    if text_lower in IDENTITY:
        return {"reply": IDENTITY[text_lower], "thinking": "Identity.",
                "searched": False, "sources": [], "model_used": "static"}
    if text_lower in SOCIAL:
        return {"reply": SOCIAL[text_lower], "thinking": "Social.",
                "searched": False, "sources": [], "model_used": "static"}

    # 4. Pure AI thinking — model decides search (no rules)
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
    #    Search context is embedded in the system prompt (authoritative placement)
    system_content = make_system(search_context)

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:1500]})

    # User message references the data if present
    final_user = (
        f"Using the live data provided in the system context, answer:\n{user_input}"
        if search_context else user_input
    )
    messages.append({"role": "user", "content": final_user})

    # 7. Call AI — patient mode waits up to 50s if all models are cooling
    try:
        reply, model_used = call_patient(
            messages, CHAT_MODELS,
            max_tokens=700, temperature=0.65, top_p=0.9,
            max_wait=50.0,
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
