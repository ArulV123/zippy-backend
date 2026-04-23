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
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────────────
#  DATE HELPERS
# ─────────────────────────────────────────────────────────────────────
_START_TS   = time.time()
_START_DATE = datetime.fromtimestamp(_START_TS, tz=timezone.utc)

def _utc_now() -> datetime:
    return _START_DATE + timedelta(seconds=time.time() - _START_TS)

def today_str() -> str:
    return _utc_now().strftime("%A, %d %B %Y")

def now_utc_str() -> str:
    return _utc_now().strftime("%d %b %Y %H:%M UTC")

def current_year() -> int:
    return _utc_now().year


# ─────────────────────────────────────────────────────────────────────
#  API KEY POOL
#  Reads keys from environment variables:
#    GROQ_API_KEY  → key slot 0  (your original key)
#    GROQ_KEY_1    → key slot 1
#    GROQ_KEY_2    → key slot 2
#    GROQ_KEY_3    → key slot 3
#    GROQ_KEY_4    → key slot 4
#
#  Logic:
#  • Try every model on key 0 first
#  • If all models on key 0 are rate-limited → silently move to key 1
#  • If all models on key 1 are rate-limited → silently move to key 2
#  • If all models on key 2 are rate-limited → silently move to key 3
#  • If all models on key 3 are rate-limited → silently move to key 4
#  • If all keys exhausted → wait for soonest recovery or return friendly error
# ─────────────────────────────────────────────────────────────────────

def _load_keys() -> list[str]:
    """Load all API keys from environment, deduplicated, no blanks."""
    raw = [
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("GROQ_KEY_1",   ""),
        os.environ.get("GROQ_KEY_2",   ""),
        os.environ.get("GROQ_KEY_3",   ""),
        os.environ.get("GROQ_KEY_4",   ""),
    ]
    seen = set()
    keys = []
    for k in raw:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys

API_KEYS: list[str] = _load_keys()

if not API_KEYS:
    raise RuntimeError(
        "No API keys found. Set GROQ_API_KEY, GROQ_KEY_1, GROQ_KEY_2, "
        "GROQ_KEY_3, GROQ_KEY_4 in Render environment variables."
    )

print(f"[keys] Loaded {len(API_KEYS)} key(s)")

# One Groq client per key
_clients: list[Groq] = [Groq(api_key=k) for k in API_KEYS]


# ─────────────────────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────────────────────
CHAT_MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Model A"},
    {"id": "llama-3.1-8b-instant",    "name": "Model B"},
    {"id": "gemma2-9b-it",            "name": "Model C"},
    {"id": "llama3-70b-8192",         "name": "Model D"},
    {"id": "llama3-8b-8192",          "name": "Model E"},
]

THINK_MODELS = [
    {"id": "llama-3.1-8b-instant",    "name": "Model B"},
    {"id": "gemma2-9b-it",            "name": "Model C"},
    {"id": "llama3-8b-8192",          "name": "Model E"},
    {"id": "llama-3.3-70b-versatile", "name": "Model A"},
]


# ─────────────────────────────────────────────────────────────────────
#  COOLDOWN TRACKER
#  Tracks when each (key_index, model_id) pair becomes available again.
# ─────────────────────────────────────────────────────────────────────

# cooldown[(key_idx, model_id)] = unix timestamp when available again
cooldown: dict[tuple[int, str], float] = {}

def _is_available(key_idx: int, mid: str) -> bool:
    return time.time() >= cooldown.get((key_idx, mid), 0)

def _mark_limited(key_idx: int, mid: str, wait_sec: float):
    cooldown[(key_idx, mid)] = time.time() + wait_sec
    print(f"[rl] key{key_idx} / {mid} blocked {wait_sec:.0f}s")

def _parse_wait(exc: Exception) -> float:
    """Extract retry-after seconds from Groq rate limit error message."""
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

def _soonest_recovery(models: list[dict]) -> float:
    """Seconds until any (key, model) pair becomes available. 0 if any is free now."""
    now = time.time()
    waits = []
    for ki in range(len(API_KEYS)):
        for m in models:
            until = cooldown.get((ki, m["id"]), 0)
            waits.append(max(0.0, until - now))
    return min(waits) if waits else 0.0

def fmt_wait(s: float) -> str:
    if s < 5:  return "a few seconds"
    if s < 90: return f"{math.ceil(s)} seconds"
    return f"{math.ceil(s/60)} minute{'s' if math.ceil(s/60) != 1 else ''}"


# ─────────────────────────────────────────────────────────────────────
#  CORE CALLER
#  For each key (0 → 1 → 2), tries every model in order.
#  Moves to the next key only when ALL models on the current key fail.
#  Errors are logged internally; the caller never sees model/key names.
# ─────────────────────────────────────────────────────────────────────

def call_models(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 700,
    temperature: float = 0.65,
    top_p: float = 0.9,
) -> tuple[str, str]:
    """
    Try every model on every key.
    Returns (reply_text, internal_model_id).
    Raises RuntimeError with QUOTA_EXCEEDED prefix if everything fails.
    """
    for ki, client in enumerate(_clients):
        for m in models:
            mid = m["id"]
            if not _is_available(ki, mid):
                continue   # this (key, model) is cooling — skip silently
            try:
                r = client.chat.completions.create(
                    model=mid,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                print(f"[ok] key{ki} / {mid}")
                return r.choices[0].message.content.strip(), mid

            except RateLimitError as e:
                wait = _parse_wait(e)
                _mark_limited(ki, mid, wait)
                # continue to next model on same key

            except APIStatusError as e:
                _mark_limited(ki, mid, 15)
                print(f"[api-err] key{ki} / {mid} → status {e.status_code}")
                # continue to next model on same key

            except Exception as e:
                print(f"[err] key{ki} / {mid} → {type(e).__name__}: {e}")
                # continue to next model on same key

    # Everything exhausted
    soonest = _soonest_recovery(models)
    raise RuntimeError(f"QUOTA_EXCEEDED|{fmt_wait(soonest)}")


def call_patient(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 700,
    temperature: float = 0.65,
    top_p: float = 0.9,
    max_wait: float = 50.0,
) -> tuple[str, str]:
    """
    Same as call_models but if everything is rate-limited,
    waits up to max_wait seconds for the soonest slot and retries once.
    """
    try:
        return call_models(messages, models, max_tokens, temperature, top_p)
    except RuntimeError as e:
        if "QUOTA_EXCEEDED" not in str(e):
            raise
        soonest = _soonest_recovery(models)
        if 0 < soonest <= max_wait:
            print(f"[patient] waiting {soonest:.0f}s for any slot to free up…")
            time.sleep(soonest + 1)
            return call_models(messages, models, max_tokens, temperature, top_p)
        raise


# ─────────────────────────────────────────────────────────────────────
#  WARM-UP
# ─────────────────────────────────────────────────────────────────────
def _warm_up():
    time.sleep(2)
    for ki, client in enumerate(_clients):
        try:
            client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print(f"[warmup] ✓ key{ki} ready")
            return   # one successful warm-up is enough
        except Exception as e:
            print(f"[warmup] key{ki}: {e}")

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
#  THINKING PROMPT
# ─────────────────────────────────────────────────────────────────────
def make_think_prompt() -> str:
    return f"""You are the search decision engine for Zippy AI.
TODAY IS EXACTLY: {today_str()}
CURRENT YEAR: {current_year()}

Read the user message and decide: does answering it need LIVE internet data?

Output ONLY raw JSON:
{{
  "needs_search": true or false,
  "search_query": "short optimised query, empty if false",
  "reasoning": "one sentence"
}}

SEARCH NEEDED (needs_search: true):
• Any price: crypto, stocks, gold, silver, platinum, oil
• Petrol / diesel / fuel prices anywhere
• Currency exchange rates (USD/INR, dollar, forex)
• Weather or forecast for any city
• Sports scores, match results, tournament winners
• Recent or breaking news, current events
• Elections, government changes, political events
• "Who is the current X" — any active role
• Software releases: "latest version", "new update"
• Questions mentioning today / now / currently / latest / this week / 2024 / 2025 / 2026
• Any ongoing situation or event from 2023–{current_year()}

NO SEARCH NEEDED (needs_search: false):
• Normal conversation: greetings, jokes, small talk
• Pure coding: syntax, algorithms, how to write code
• Math problems and calculations
• Science theory (concepts that don't change)
• History before 2023
• Creative writing: poems, essays, stories
• Definitions of stable concepts
• Grammar and translation

When in doubt → search.

JSON ONLY:"""


def run_thinking(user_input: str, timeout: float = 4.0) -> dict:
    box: list[dict] = []

    def _fallback() -> dict:
        text = user_input.lower()
        news_like = bool(re.search(r"\b(news|latest|recent|today|current|update|breaking|election|result|score|match|weather|forecast|price|prices|crypto|stock|stocks|gold|silver|oil|currency|exchange|who is the current|current\s+\w+)\b", text))
        if news_like:
            return {"needs_search": True, "search_query": user_input, "reasoning": "Keyword fallback to live search."}
        return {"needs_search": False, "search_query": "", "reasoning": "Heuristic fallback."}

    def _call():
        try:
            raw, _ = call_models(
                [{"role": "system", "content": make_think_prompt()},
                 {"role": "user",   "content": user_input}],
                THINK_MODELS, max_tokens=40, temperature=0.0, top_p=1.0,
            )
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                p = json.loads(match.group(0))
                if isinstance(p.get("needs_search"), bool):
                    box.append(p)
        except Exception as e:
            print(f"[think] err: {e}")

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if box:
        r = box[0]
        print(f"[think] search={r['needs_search']} q='{r.get('search_query','')}'")
        return r

    r = _fallback()
    print(f"[think] fallback search={r['needs_search']} q='{r.get('search_query','')}'")
    return r


# ─────────────────────────────────────────────────────────────────────
#  SEARCH SOURCES
# ─────────────────────────────────────────────────────────────────────
BASE_H = {
    "User-Agent": "Mozilla/5.0 (compatible; ZippyAI/2.0)",
    "Accept":     "application/json, text/html, */*",
}

_fx: dict = {"rate": None, "ts": 0.0}

def get_usd_inr() -> float:
    if time.time() - _fx["ts"] < 300 and _fx["rate"]:
        return _fx["rate"]
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD",
                         headers=BASE_H, timeout=6)
        r.raise_for_status()
        rate = float(r.json()["rates"]["INR"])
        _fx.update({"rate": rate, "ts": time.time()})
        return rate
    except Exception as e:
        print(f"[fx] {e}")
        return _fx.get("rate") or 83.5


def _f(val, pre="$", dp=2) -> str:
    if not isinstance(val, (int, float)): return "N/A"
    return f"{pre}{val:,.{dp}f}" if val >= 0.01 else f"{pre}{val:.6f}"


# ── CRYPTO ────────────────────────────────────────────────────────────
CRYPTO_CG = {
    "bitcoin":"bitcoin","btc":"bitcoin","ethereum":"ethereum","eth":"ethereum",
    "dogecoin":"dogecoin","doge":"dogecoin","solana":"solana","sol":"solana",
    "bnb":"binancecoin","xrp":"ripple","ripple":"ripple","cardano":"cardano",
    "ada":"cardano","litecoin":"litecoin","ltc":"litecoin","polkadot":"polkadot",
    "dot":"polkadot","shib":"shiba-inu","shiba":"shiba-inu","tron":"tron",
    "trx":"tron","avax":"avalanche-2","avalanche":"avalanche-2",
    "matic":"matic-network","polygon":"matic-network","pepe":"pepe",
}
CRYPTO_CC = {
    "bitcoin":"BTC","btc":"BTC","ethereum":"ETH","eth":"ETH",
    "dogecoin":"DOGE","doge":"DOGE","solana":"SOL","sol":"SOL",
    "bnb":"BNB","xrp":"XRP","ripple":"XRP","cardano":"ADA","ada":"ADA",
    "litecoin":"LTC","ltc":"LTC","polkadot":"DOT","dot":"DOT",
    "shib":"SHIB","shiba":"SHIB","tron":"TRX","trx":"TRX",
    "avax":"AVAX","avalanche":"AVAX","matic":"MATIC","polygon":"MATIC",
}

def search_crypto(q: str) -> dict | None:
    lo = q.lower()
    cg = list({v for k, v in CRYPTO_CG.items() if k in lo})
    cc = list({v for k, v in CRYPTO_CC.items() if k in lo})
    if not cg: return None
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(cg), "vs_currencies": "usd,inr",
                    "include_24hr_change": "true", "include_market_cap": "true",
                    "include_24hr_vol": "true"},
            headers=BASE_H, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data:
            lines = []
            for cid, info in data.items():
                usd  = info.get("usd"); inr = info.get("inr")
                chg  = info.get("usd_24h_change")
                mcap = info.get("usd_market_cap"); vol = info.get("usd_24h_vol")
                lines.append(
                    f"• {cid.capitalize()} — {now_utc_str()}\n"
                    f"  Price:      {_f(usd)} USD  /  {_f(inr, '₹')} INR\n"
                    f"  24h Change: {f'{chg:+.2f}%' if chg is not None else 'N/A'}\n"
                    f"  Market Cap: {_f(mcap) if mcap else 'N/A'}\n"
                    f"  24h Volume: {_f(vol) if vol else 'N/A'}")
            print("[crypto] CoinGecko ✓")
            return {"source": "CoinGecko (live)", "title": "Live Crypto Prices",
                    "url": "https://www.coingecko.com", "content": "\n\n".join(lines)}
    except Exception as e:
        print(f"[crypto] CoinGecko: {e}")
    if cc:
        try:
            r = requests.get(
                "https://min-api.cryptocompare.com/data/pricemultifull",
                params={"fsyms": ",".join(cc), "tsyms": "USD,INR"},
                headers=BASE_H, timeout=8)
            r.raise_for_status()
            raw = r.json().get("RAW", {})
            if raw:
                lines = []
                for sym, mkt in raw.items():
                    u = mkt.get("USD", {}); n = mkt.get("INR", {})
                    pu = u.get("PRICE"); pi = n.get("PRICE")
                    chg = u.get("CHANGEPCT24HOUR")
                    lines.append(
                        f"• {sym} — {now_utc_str()}\n"
                        f"  Price:      {_f(pu)} USD  /  {_f(pi, '₹')} INR\n"
                        f"  24h Change: {f'{chg:+.2f}%' if isinstance(chg, (int, float)) else 'N/A'}")
                print("[crypto] CryptoCompare ✓")
                return {"source": "CryptoCompare (live)", "title": "Live Crypto Prices",
                        "url": "https://www.cryptocompare.com", "content": "\n\n".join(lines)}
        except Exception as e:
            print(f"[crypto] CryptoCompare: {e}")
    return None


# ── METALS ────────────────────────────────────────────────────────────
def search_metals(q: str) -> dict | None:
    lo = q.lower()
    metals = [m for m in ["gold", "silver", "platinum", "palladium"] if m in lo]
    if not metals: return None
    try:
        r = requests.get("https://api.metals.live/v1/spot", headers=BASE_H, timeout=8)
        r.raise_for_status()
        spot = r.json()
        if isinstance(spot, list) and spot: spot = spot[0]
        if not isinstance(spot, dict): return None
        usd_inr = get_usd_inr()
        TROY = 31.1035
        lines = []
        for metal in metals:
            oz = spot.get(metal)
            if oz is None: continue
            oz = float(oz)
            oz_inr    = oz * usd_inr
            per_g_inr = oz_inr / TROY
            lines.append(
                f"• {metal.capitalize()} — spot price {now_utc_str()}\n"
                f"  International: {_f(oz)} USD / troy oz  |  {_f(oz_inr, '₹')} INR / troy oz\n"
                f"  India — per gram:  {_f(per_g_inr, '₹')}\n"
                f"  India — per 10g:   {_f(per_g_inr * 10, '₹')}\n"
                f"  India — per 100g:  {_f(per_g_inr * 100, '₹')}\n"
                f"  India — per kg:    {_f(per_g_inr * 1000, '₹')}\n"
                f"  (Retail prices include making charges/GST — add 5–15%)\n"
                f"  (USD/INR rate used: {usd_inr:.2f})")
        if not lines: return None
        print(f"[metals] ✓ {metals}")
        return {"source": "metals.live + open.er-api.com (live)", "title": "Live Metal Prices",
                "url": "https://metals.live", "content": "\n\n".join(lines)}
    except Exception as e:
        print(f"[metals] {e}"); return None


# ── FOREX ─────────────────────────────────────────────────────────────
FX_NAMES = {
    "dollar": "USD", "usd": "USD", "us dollar": "USD", "american dollar": "USD",
    "rupee": "INR", "inr": "INR", "indian rupee": "INR",
    "euro": "EUR", "eur": "EUR",
    "pound": "GBP", "gbp": "GBP", "sterling": "GBP",
    "yen": "JPY", "jpy": "JPY", "japanese yen": "JPY",
    "yuan": "CNY", "cny": "CNY", "chinese yuan": "CNY",
    "aud": "AUD", "australian dollar": "AUD",
    "cad": "CAD", "canadian dollar": "CAD",
    "sgd": "SGD", "singapore dollar": "SGD",
    "aed": "AED", "dirham": "AED", "uae": "AED",
    "chf": "CHF", "swiss franc": "CHF",
}
FX_TRIGGER = {"forex", "exchange rate", "currency", "dollar", "rupee", "euro",
              "pound", "usd", "inr", "eur", "gbp", "jpy", "cny", "aud", "cad",
              "sgd", "aed", "to inr", "to usd", "to rupee", "conversion", "convert"}

def search_forex(q: str) -> dict | None:
    lo = q.lower()
    if not any(w in lo for w in FX_TRIGGER): return None
    frm = to = None
    m = re.search(r'(\w[\w\s]*?)\s+to\s+(\w[\w\s]*?)(?:\s+rate|\s+price|\s+exchange|\?|$)', lo)
    if m:
        frm = FX_NAMES.get(m.group(1).strip())
        to  = FX_NAMES.get(m.group(2).strip())
    if not frm:
        for name, code in FX_NAMES.items():
            if name in lo and code != "INR":
                frm = code; to = "INR"; break
    if not frm: frm = "USD"
    if not to:  to  = "INR"
    try:
        r = requests.get(f"https://open.er-api.com/v6/latest/{frm}",
                         headers=BASE_H, timeout=7)
        r.raise_for_status()
        data  = r.json()
        rates = data.get("rates", {})
        upd   = data.get("time_last_update_utc", now_utc_str())
        show  = [to] + [c for c in ["USD", "INR", "EUR", "GBP", "JPY", "AED"]
                        if c not in (frm, to)][:4]
        lines = [f"Currency: {frm} — as of {upd}"]
        for c in show:
            v = rates.get(c)
            if v: lines.append(f"  1 {frm} = {v:,.4f} {c}")
        print(f"[forex] ✓ {frm}")
        return {"source": "Open Exchange Rates (live)", "title": f"Rate: {frm}",
                "url": "https://open.er-api.com", "content": "\n".join(lines)}
    except Exception as e:
        print(f"[forex] {e}"); return None


# ── PETROL / DIESEL ───────────────────────────────────────────────────
FUEL_WORDS = {"petrol", "diesel", "fuel", "lpg", "cng", "gas price", "gasoline"}

def search_fuel(q: str) -> dict | None:
    if not any(w in q.lower() for w in FUEL_WORDS): return None
    content = (
        f"FUEL PRICE DATA — {now_utc_str()}\n\n"
        "IMPORTANT: No real-time API for Indian city-wise petrol/diesel prices.\n\n"
        "Tell the user:\n"
        "• Petrol prices in India vary by city and change on the 1st of each month.\n"
        "• Typical range (early 2026): ₹94–₹106/litre petrol, ₹87–₹95/litre diesel.\n"
        "• For the exact price in their city, check:\n"
        "  - Indian Oil: https://iocl.com\n"
        "  - HP Petrol: https://hindustanpetroleum.com\n"
        "  - Google: 'petrol price [city] today'\n\n"
        "DO NOT state any specific per-litre price as current fact."
    )
    print("[fuel] returning guidance notice")
    return {"source": "Guidance only — no live fuel API", "title": "Petrol/Diesel Prices",
            "url": "https://iocl.com", "content": content}


# ── WEATHER ───────────────────────────────────────────────────────────
def _city(q: str) -> str | None:
    lo = q.lower()
    if not any(w in lo for w in {"weather", "temperature", "forecast", "humidity",
                                  "rain", "wind", "climate", "hot", "cold", "sunny"}):
        return None
    for pat in [
        r'weather\s+(?:in|at|for|of)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'([A-Za-z][A-Za-z\s]{1,24})\s+weather',
        r'(?:temperature|forecast)\s+(?:in|of|at)?\s*([A-Za-z][A-Za-z\s]{1,24})',
    ]:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            c = re.sub(r'\b(today|now|currently|like|is|the|a|an)\b', '',
                       m.group(1), flags=re.IGNORECASE).strip()
            if len(c) > 1: return c
    return None

def search_weather(q: str) -> dict | None:
    city = _city(q)
    if not city: return None
    try:
        r = requests.get(f"https://wttr.in/{requests.utils.quote(city)}",
                         params={"format": "j1"},
                         headers={"User-Agent": "ZippyAI/2.0"}, timeout=8)
        r.raise_for_status()
        d        = r.json()
        cur      = d["current_condition"][0]
        desc     = cur.get("weatherDesc", [{}])[0].get("value", "N/A")
        temp_c   = cur.get("temp_C", "N/A"); temp_f = cur.get("temp_F", "N/A")
        feels    = cur.get("FeelsLikeC", "N/A"); humidity = cur.get("humidity", "N/A")
        wind     = cur.get("windspeedKmph", "N/A"); uv = cur.get("uvIndex", "N/A")
        fc = []
        for day in d.get("weather", [])[:3]:
            date  = day.get("date", "")
            maxc  = day.get("maxtempC", "N/A"); minc = day.get("mintempC", "N/A")
            desc2 = (day.get("hourly") or [{}])[4].get(
                "weatherDesc", [{}])[0].get("value", "N/A")
            fc.append(f"  {date}: {desc2}, {minc}°C – {maxc}°C")
        content = (
            f"Weather in {city} — {now_utc_str()}:\n"
            f"  Condition:   {desc}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  Feels like:  {feels}°C\n"
            f"  Humidity:    {humidity}%\n"
            f"  Wind:        {wind} km/h\n"
            f"  UV Index:    {uv}\n\n"
            f"3-Day Forecast:\n" + "\n".join(fc))
        print(f"[weather] ✓ {city}")
        return {"source": "wttr.in (live)", "title": f"Weather in {city}",
                "url": f"https://wttr.in/{city}", "content": content}
    except Exception as e:
        print(f"[weather] {e}"); return None


# ── NEWS ──────────────────────────────────────────────────────────────
NEWS_MAX_AGE_DAYS = 7

def _article_age_days(pub_date_str: str) -> float:
    if not pub_date_str: return 999
    try:
        dt  = parsedate_to_datetime(pub_date_str).astimezone(timezone.utc)
        age = (_utc_now() - dt).total_seconds() / 86400
        return max(0.0, age)
    except Exception:
        return 999

def _fetch_rss(url: str, src: str, keywords: list[str], max_items: int = 20) -> list[dict]:
    try:
        r = requests.get(
            url,
            headers={**BASE_H, "Accept": "application/rss+xml,text/xml,*/*"},
            timeout=9)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out  = []
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            src_e = (item.findtext("source") or src).strip()
            desc  = re.sub(r'<[^>]+>', '', (item.findtext("description") or "")).strip()
            title_clean = re.sub(r'\s*-\s*[^-]{3,45}$', '', title).strip() or title
            age   = _article_age_days(pub)
            if age > NEWS_MAX_AGE_DAYS: continue
            score = sum(1 for kw in keywords if kw.lower() in (title + desc).lower())
            if title_clean:
                out.append({"title": title_clean, "desc": desc[:280],
                            "pub": pub, "source": src_e, "link": link,
                            "score": score, "age": age})
        return out
    except Exception as e:
        print(f"[rss] {src}: {e}"); return []

NEWS_FEEDS = [
    ("Google News",    "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q={q}"),
    ("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("The Hindu",      "https://www.thehindu.com/news/feeder/default.rss"),
    ("Al Jazeera",     "https://www.aljazeera.com/xml/rss/all.xml"),
    ("BBC Technology", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
]

def search_news(q: str, max_results: int = 6) -> dict | None:
    keywords = [w for w in re.split(r'\W+', q.lower()) if len(w) > 2]
    all_a: list[dict] = []
    for name, tmpl in NEWS_FEEDS:
        url  = tmpl.replace("{q}", requests.utils.quote(q))
        arts = _fetch_rss(url, name, keywords)
        print(f"[news] {name} → {len(arts)} recent articles")
        all_a.extend(arts)

    if not all_a:
        print("[news] no recent articles — retrying without age filter")
        for name, tmpl in NEWS_FEEDS[:2]:
            url = tmpl.replace("{q}", requests.utils.quote(q))
            try:
                r = requests.get(
                    url,
                    headers={**BASE_H, "Accept": "application/rss+xml,text/xml,*/*"},
                    timeout=9)
                r.raise_for_status()
                root = ET.fromstring(r.content)
                for item in root.findall(".//item")[:10]:
                    title = re.sub(r'\s*-\s*[^-]{3,45}$', '',
                                   (item.findtext("title") or "").strip()).strip()
                    link  = (item.findtext("link") or "").strip()
                    pub   = (item.findtext("pubDate") or "").strip()
                    desc  = re.sub(r'<[^>]+>', '',
                                   (item.findtext("description") or "")).strip()
                    if title:
                        all_a.append({"title": title, "desc": desc[:280],
                                      "pub": pub, "source": name, "link": link,
                                      "score": 1, "age": 999})
            except Exception:
                pass

    if not all_a: return None

    seen: set[str] = set(); unique: list[dict] = []
    for a in all_a:
        key = re.sub(r'\W+', '', a["title"].lower())[:40]
        if key not in seen: seen.add(key); unique.append(a)
    unique.sort(key=lambda x: (-x["score"], x["age"]))
    top = unique[:max_results]

    header  = f"News for '{q}' — fetched {now_utc_str()}:\n"
    bullets = []
    for a in top:
        b = f"• **{a['title']}**"
        if a["desc"]: b += f"\n  {a['desc']}"
        b += f"\n  Source: {a['source']}"
        if a["pub"]:  b += f"  |  Published: {a['pub']}"
        if a["link"]: b += f"\n  {a['link']}"
        bullets.append(b)

    print(f"[news] ✓ {len(top)} articles returned")
    return {
        "source":  "Google News / TOI / The Hindu / Al Jazeera (live RSS)",
        "title":   f"News: {q}",
        "url":     f"https://news.google.com/search?q={requests.utils.quote(q)}",
        "content": header + "\n\n".join(bullets),
    }


# ── WIKIPEDIA ─────────────────────────────────────────────────────────
def search_wikipedia(q: str) -> dict | None:
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action": "query", "list": "search", "srsearch": q,
                                 "srlimit": 1, "format": "json", "origin": "*"},
                         headers=BASE_H, timeout=7)
        r.raise_for_status()
        res = r.json().get("query", {}).get("search", [])
        if not res: return None
        title = res[0]["title"]
    except Exception as e:
        print(f"[wiki] search: {e}"); return None
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action": "query", "prop": "extracts",
                                 "exintro": False, "explaintext": True,
                                 "titles": title, "format": "json",
                                 "origin": "*", "exchars": 3000},
                         headers=BASE_H, timeout=8)
        r.raise_for_status()
        for pg in r.json().get("query", {}).get("pages", {}).values():
            txt = pg.get("extract", "")
            if txt and len(txt) > 80:
                url = ("https://en.wikipedia.org/wiki/"
                       + requests.utils.quote(title.replace(' ', '_')))
                print(f"[wiki] ✓ '{title}'")
                return {"source": "Wikipedia", "title": title,
                        "url": url, "content": txt}
    except Exception as e:
        print(f"[wiki] extract: {e}")
    return None


# ── REST COUNTRIES ────────────────────────────────────────────────────
def search_country(q: str) -> dict | None:
    if not any(w in q.lower() for w in
               {"capital", "population", "currency", "language",
                "country", "nation", "area"}):
        return None
    m = re.search(
        r'(?:of|in|about|for|capital of|population of|currency of)\s+'
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', q)
    if not m:
        m = re.search(r'\b([A-Z][a-z]{3,})\b', q)
    if not m: return None
    name = m.group(1).strip()
    try:
        r = requests.get(
            f"https://restcountries.com/v3.1/name/{requests.utils.quote(name)}",
            headers=BASE_H, timeout=7)
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict): return None
        c      = data[0]
        common = c.get("name", {}).get("common", name)
        cur_list = [
            v.get("name", "") + " (" + v.get("symbol", "") + ")"
            for v in c.get("currencies", {}).values()
        ]
        content = (
            f"Country: {common}\n"
            f"  Capital:    {', '.join(c.get('capital', ['N/A']))}\n"
            f"  Population: {c.get('population', 0):,}\n"
            f"  Region:     {c.get('region', 'N/A')} ({c.get('subregion', 'N/A')})\n"
            f"  Area:       {c.get('area', 'N/A')} km\u00b2\n"
            f"  Currencies: {', '.join(cur_list) or 'N/A'}\n"
            f"  Languages:  {', '.join(c.get('languages', {}).values()) or 'N/A'}"
        )
        print(f"[country] ✓ {common}")
        return {"source": "REST Countries", "title": f"Country: {common}",
                "url": "https://restcountries.com", "content": content}
    except Exception as e:
        print(f"[country] {e}"); return None


# ── MASTER SEARCH ─────────────────────────────────────────────────────
def run_search(q: str) -> list[dict]:
    results: list[dict] = []; seen: set[str] = set()

    def add(item):
        if not item: return
        k = item.get("url") or item.get("title", "")
        if k and k not in seen: seen.add(k); results.append(item)

    add(search_crypto(q))
    add(search_metals(q))
    add(search_forex(q))
    add(search_fuel(q))
    add(search_weather(q))
    add(search_country(q))
    add(search_news(q))
    add(search_wikipedia(q))
    print(f"[search] {len(results)} sources")
    return results


def build_context(q: str, results: list[dict]) -> str:
    ts = now_utc_str()
    if not results:
        return (
            f"[SEARCH — {ts}]\nQuery: {q}\n"
            "Result: No live data found.\n"
            "Tell the user you searched but found nothing. "
            "Do NOT say 'I don't have real-time access'.")
    lines = [
        f"╔══ LIVE DATA ═══════════════════════════════",
        f"║  Fetched: {ts}",
        f"║  Query:   {q}",
        f"║  Sources: {len(results)}",
        f"╚═══════════════════════════════════════════",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [f"── SOURCE {i}: {r['source']} ──",
                  f"Title: {r['title']}", f"URL:   {r['url']}", "",
                  r["content"], ""]
    lines += [
        "── END OF LIVE DATA ──", "",
        "⚠ MANDATORY: Use the data above to answer. Do NOT ignore it.",
        "⚠ MANDATORY: Do NOT say 'I don't have real-time access'.",
        "⚠ MANDATORY: Do NOT say 'I cannot provide current prices'.",
        "⚠ MANDATORY: State prices/values in your FIRST sentence.",
        "⚠ MANDATORY: For news, use bullet points.",
        f"⚠ MANDATORY: Today is {today_str()}. Use this date.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────
IMG_RE = re.compile(
    r'\b(generate|create|make|draw|paint|design|produce|render)\b.{0,25}'
    r'\b(image|picture|photo|illustration|artwork|graphic|wallpaper|logo|'
    r'poster|banner|sketch|portrait|thumbnail)\b|^/imagine\b',
    re.IGNORECASE)

IMG_DECLINE = (
    "I'm text-only — I can't generate images. 🙅\n\n"
    "Try these free tools:\n"
    "• **[Adobe Firefly](https://firefly.adobe.com)** — free, high quality\n"
    "• **[Microsoft Designer](https://designer.microsoft.com)** — free with account\n"
    "• **[Ideogram](https://ideogram.ai)** — great for text in images\n"
    "• **[Craiyon](https://www.craiyon.com)** — free, no account needed\n\n"
    "Want me to write a prompt for any of these? ✍️"
)


def _image_prompt_from_user(user_input: str) -> str:
    prompt = re.sub(r'\s+', ' ', user_input or '').strip()
    prompt = re.sub(r'^\s*/imagine\s*', '', prompt, flags=re.IGNORECASE)
    prompt = re.sub(
        r'^\s*(?:please\s+)?(?:generate|create|make|draw|paint|design|produce|render)\s+'
        r'(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|illustration|artwork|graphic|wallpaper|logo|'
        r'poster|banner|sketch|portrait|thumbnail)\s*(?:of|for)?\s*',
        '',
        prompt,
        flags=re.IGNORECASE,
    ).strip()
    prompt = prompt.strip(" \"'`")
    return prompt or (user_input or '').strip()


# ─────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────
def make_system(ctx: str = "") -> str:
    td = today_str(); tu = now_utc_str(); yr = current_year()
    date_block = (
        f"╔═══════════════════════════════════════╗\n"
        f"║  TODAY:  {td:<31}║\n"
        f"║  TIME:   {tu:<31}║\n"
        f"║  YEAR:   {yr:<31}║\n"
        f"╚═══════════════════════════════════════╝\n"
        f"USE THIS DATE. NEVER assume any other year or date."
    )
    base = f"""{date_block}

You are Zippy, a smart AI assistant made by Arul Vethathiri.

## IDENTITY
- Text-only AI. You cannot generate images.
- Made by Arul Vethathiri, Class 11 student ({yr}).

## TONE
- Smart, calm, friendly — like a knowledgeable friend.
- Conversational. Not corporate. Not over-excited.
- No greeting or intro for news/current-event answers.

## RESPONSE RULES
1. Answer exactly what was asked. Nothing extra.
2. Prices/numbers → state them in the VERY FIRST sentence.
3. Simple questions → 1-3 sentences max.
4. News / current events → start directly with bullet points, one per story, with no preamble or self-introduction.
5. Explanations → short bullets or brief paragraphs.
6. Code → give directly with brief comments.
7. Math → show steps briefly.
8. Creative → complete the piece, no preamble.
9. For live news/source replies, never include raw source URLs in the text; source cards are shown separately in the UI.
10. FORBIDDEN phrases: "Certainly!", "Great question!", "As an AI",
   "I don't have real-time access", "my training data is limited",
   "I cannot provide current prices", "I'd be happy to".
11. Never repeat the question.
12. End with exactly 1 relevant emoji.

## PETROL / DIESEL PRICES
No real-time API exists for Indian fuel prices.
Tell the user the typical range and direct them to iocl.com or hpcl.com.
NEVER guess or invent a specific per-litre price.
"""
    if not ctx:
        return base
    return base + f"""

## ⚠ LIVE DATA BELOW — YOU MUST USE THIS ⚠

{ctx}

## HOW TO USE THE LIVE DATA
- READ every source carefully. The exact figures are there.
- Use those exact numbers — do not use training-data estimates.
- State today's date as {td} if asked.
- NEVER say 'I don't have real-time access'.
- NEVER say 'I cannot provide current prices'.
- If fuel data says "no live data", tell the user honestly and give the URL.
- For news: present as bullet points.
- Cite sources naturally: "According to CoinGecko..." or "Google News reports..."
"""


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
#  REQUEST / ROUTES
# ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.get("/")
def root():
    now  = time.time()
    avail = sum(
        1 for ki in range(len(API_KEYS))
        for m in CHAT_MODELS
        if cooldown.get((ki, m["id"]), 0) <= now
    )
    return {
        "status":          "Zippy running",
        "date":            today_str(),
        "time":            now_utc_str(),
        "keys_loaded":     len(API_KEYS),
        "slots_available": avail,
        "slots_total":     len(API_KEYS) * len(CHAT_MODELS),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message, flags=re.IGNORECASE
    ).strip()

    if not user_input:
        return {"reply": "Didn't catch that — say again? 😊",
                "thinking": "", "searched": False, "sources": []}

    tl = user_input.lower().strip()

    if IMG_RE.search(user_input):
        prompt = _image_prompt_from_user(user_input)
        return {"reply": f"[[IMAGE: {prompt}]]", "thinking": "",
                "searched": False, "sources": []}

    if tl in IDENTITY:
        return {"reply": IDENTITY[tl], "thinking": "",
                "searched": False, "sources": []}
    if tl in SOCIAL:
        return {"reply": SOCIAL[tl], "thinking": "",
                "searched": False, "sources": []}

    # Decide whether to search
    think        = run_thinking(user_input)
    needs_search = think.get("needs_search", False)
    sq           = (think.get("search_query") or "").strip() or user_input
    reasoning    = think.get("reasoning", "")

    ctx = ""; sources = []; found = False
    if needs_search:
        results = run_search(sq)
        ctx     = build_context(sq, results)
        if results:
            found   = True
            sources = [{"title": r["title"], "url": r["url"], "source": r["source"]}
                       for r in results]

    # Build messages
    system = make_system(ctx)
    msgs   = [{"role": "system", "content": system}]

    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            msgs.append({"role": h["role"], "content": str(h["content"])[:1500]})

    user_msg = (
        f"[Today is {today_str()}. Use live data from system context.]\n\n{user_input}"
        if ctx else
        f"[Today is {today_str()}]\n\n{user_input}"
    )
    msgs.append({"role": "user", "content": user_msg})

    try:
        reply, _ = call_patient(
            msgs, CHAT_MODELS,
            max_tokens=700, temperature=0.65, top_p=0.9, max_wait=50.0,
        )
        return {"reply": reply, "thinking": reasoning,
                "searched": found, "sources": sources}

    except RuntimeError as exc:
        # Extract wait time from error — never expose internal names
        parts    = str(exc).split("|", 1)
        wait_str = parts[1].strip() if len(parts) == 2 else "a few minutes"
        return {
            "reply": (
                f"Zippy is taking a short break due to high demand. 🔄\n\n"
                f"Please wait **{wait_str}** and try again — "
                f"Zippy will be back shortly!"
            ),
            "thinking": "",
            "searched": False,
            "sources":  [],
        }
