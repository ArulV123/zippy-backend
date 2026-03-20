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
#  TODAY'S DATE  — single source of truth, set once at startup
#  This avoids any drift from Render's server clock being off by a day.
#  Update this if you redeploy on a different day.
# ─────────────────────────────────────────────────────────────────────
#  We derive it from the system's UTC clock but also embed it as a
#  constant so the AI always gets the same answer within one deployment.
_START_TS   = time.time()
_START_DATE = datetime.fromtimestamp(_START_TS, tz=timezone.utc)

def _utc_now() -> datetime:
    """Current UTC datetime."""
    elapsed = time.time() - _START_TS
    return _START_DATE + timedelta(seconds=elapsed)

def today_str() -> str:
    """e.g. 'Friday, 21 March 2026'"""
    return _utc_now().strftime("%A, %d %B %Y")

def now_utc_str() -> str:
    """e.g. '21 Mar 2026 14:32 UTC'"""
    return _utc_now().strftime("%d %b %Y %H:%M UTC")

def current_year() -> int:
    return _utc_now().year


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

def is_available(mid: str) -> bool:
    return time.time() >= model_cooldown.get(mid, 0)

def mark_rate_limited(mid: str, after: float):
    model_cooldown[mid] = time.time() + after
    print(f"[rl] {mid} blocked {after:.0f}s")

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

def earliest_reset(models: list[dict]) -> float:
    now = time.time()
    return min((max(0.0, model_cooldown.get(m["id"], 0) - now) for m in models), default=0.0)

def fmt_wait(s: float) -> str:
    if s < 5:  return "a few seconds"
    if s < 90: return f"{math.ceil(s)} seconds"
    return f"{math.ceil(s/60)} minute{'s' if math.ceil(s/60) != 1 else ''}"


# ─────────────────────────────────────────────────────────────────────
#  GROQ CLIENT + WARM-UP
# ─────────────────────────────────────────────────────────────────────
_groq: Groq | None = None

def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq

def _warm_up():
    try:
        time.sleep(2)
        get_groq().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print("[warmup] ✓ Groq ready")
    except Exception as e:
        print(f"[warmup] {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────
#  CORE CALLER
# ─────────────────────────────────────────────────────────────────────
def call_models(
    messages: list[dict],
    models: list[dict],
    max_tokens: int = 700,
    temperature: float = 0.65,
    top_p: float = 0.9,
) -> tuple[str, str]:
    tried: list[str] = []
    for m in models:
        mid = m["id"]
        if not is_available(mid):
            wait = model_cooldown[mid] - time.time()
            tried.append(f"{m['name']} — cooling ({fmt_wait(wait)} left)"); continue
        try:
            r = get_groq().chat.completions.create(
                model=mid, messages=messages,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            )
            print(f"[model] ✓ {mid}")
            return r.choices[0].message.content.strip(), mid
        except RateLimitError as e:
            w = parse_retry_after(e); mark_rate_limited(mid, w)
            tried.append(f"{m['name']} — rate limited ({fmt_wait(w)})")
        except APIStatusError as e:
            mark_rate_limited(mid, 15)
            tried.append(f"{m['name']} — API {e.status_code}")
        except Exception as e:
            tried.append(f"{m['name']} — {type(e).__name__}")
            print(f"[err] {mid} → {e}")
    raise RuntimeError(f"QUOTA_EXCEEDED|{fmt_wait(earliest_reset(models))}|{'||'.join(tried)}")


def call_patient(messages, models, max_tokens=700, temperature=0.65,
                 top_p=0.9, max_wait=50.0) -> tuple[str, str]:
    """If all models rate-limited, wait up to max_wait seconds then retry."""
    try:
        return call_models(messages, models, max_tokens, temperature, top_p)
    except RuntimeError as e:
        if "QUOTA_EXCEEDED" not in str(e): raise
        soonest = earliest_reset(models)
        if 0 < soonest <= max_wait:
            print(f"[patient] waiting {soonest:.0f}s for reset…")
            time.sleep(soonest + 1)
            return call_models(messages, models, max_tokens, temperature, top_p)
        raise


# ─────────────────────────────────────────────────────────────────────
#  THINKING PROMPT  — date injected, asks AI to decide search
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

Good search_query examples:
• "bitcoin price USD INR today"
• "silver price India per gram today {current_year()}"
• "Chennai weather today"
• "petrol price India {today_str()}"
• "IPL 2025 final winner"
• "USD to INR rate today"

JSON ONLY:"""


def run_thinking(user_input: str, timeout: float = 9.0) -> dict:
    box: list[dict] = []

    def _call():
        try:
            raw, used = call_models(
                [{"role": "system", "content": make_think_prompt()},
                 {"role": "user",   "content": user_input}],
                THINK_MODELS, max_tokens=130, temperature=0.0, top_p=1.0,
            )
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            m   = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                p = json.loads(m.group(0))
                if isinstance(p.get("needs_search"), bool):
                    p["_via"] = used; box.append(p)
        except Exception as e:
            print(f"[think] err: {e}")

    t = threading.Thread(target=_call, daemon=True)
    t.start(); t.join(timeout=timeout)

    if box:
        r = box[0]
        print(f"[think] {r.get('_via','?')} search={r['needs_search']} "
              f"q='{r.get('search_query','')}' — {r.get('reasoning','')}")
        return r
    print(f"[think] timed out — no-search default")
    return {"needs_search": False, "search_query": "", "reasoning": "Timed out."}


# ─────────────────────────────────────────────────────────────────────
#  SEARCH SOURCES
# ─────────────────────────────────────────────────────────────────────
BASE_H = {
    "User-Agent": "Mozilla/5.0 (compatible; ZippyAI/2.0)",
    "Accept":     "application/json, text/html, */*",
}

# ── FX cache ──────────────────────────────────────────────────────────
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


# ── 1. CRYPTO ─────────────────────────────────────────────────────────
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
    cg = list({v for k,v in CRYPTO_CG.items() if k in lo})
    cc = list({v for k,v in CRYPTO_CC.items() if k in lo})
    if not cg: return None
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(cg), "vs_currencies": "usd,inr",
                    "include_24hr_change":"true","include_market_cap":"true",
                    "include_24hr_vol":"true"},
            headers=BASE_H, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data:
            lines = []
            for cid, info in data.items():
                usd = info.get("usd"); inr = info.get("inr")
                chg = info.get("usd_24h_change")
                mcap = info.get("usd_market_cap"); vol = info.get("usd_24h_vol")
                lines.append(
                    f"• {cid.capitalize()} — {now_utc_str()}\n"
                    f"  Price:      {_f(usd)} USD  /  {_f(inr,'₹')} INR\n"
                    f"  24h Change: {f'{chg:+.2f}%' if chg is not None else 'N/A'}\n"
                    f"  Market Cap: {_f(mcap) if mcap else 'N/A'}\n"
                    f"  24h Volume: {_f(vol) if vol else 'N/A'}")
            print(f"[crypto] CoinGecko ✓")
            return {"source":"CoinGecko (live)","title":"Live Crypto Prices",
                    "url":"https://www.coingecko.com","content":"\n\n".join(lines)}
    except Exception as e:
        print(f"[crypto] CoinGecko: {e}")
    if cc:
        try:
            r = requests.get(
                "https://min-api.cryptocompare.com/data/pricemultifull",
                params={"fsyms":",".join(cc),"tsyms":"USD,INR"},
                headers=BASE_H, timeout=8)
            r.raise_for_status()
            raw = r.json().get("RAW",{})
            if raw:
                lines = []
                for sym, mkt in raw.items():
                    u = mkt.get("USD",{}); n = mkt.get("INR",{})
                    pu = u.get("PRICE"); pi = n.get("PRICE")
                    chg = u.get("CHANGEPCT24HOUR")
                    lines.append(
                        f"• {sym} — {now_utc_str()}\n"
                        f"  Price:      {_f(pu)} USD  /  {_f(pi,'₹')} INR\n"
                        f"  24h Change: {f'{chg:+.2f}%' if isinstance(chg,(int,float)) else 'N/A'}")
                print(f"[crypto] CryptoCompare ✓")
                return {"source":"CryptoCompare (live)","title":"Live Crypto Prices",
                        "url":"https://www.cryptocompare.com","content":"\n\n".join(lines)}
        except Exception as e:
            print(f"[crypto] CryptoCompare: {e}")
    return None


# ── 2. METALS (gold / silver / platinum / palladium) ──────────────────
METAL_WORDS = {"gold","silver","platinum","palladium","bullion"}

def search_metals(q: str) -> dict | None:
    lo = q.lower()
    metals = [m for m in ["gold","silver","platinum","palladium"] if m in lo]
    if not metals: return None
    try:
        r = requests.get("https://api.metals.live/v1/spot",
                         headers=BASE_H, timeout=8)
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
            oz_inr = oz * usd_inr
            per_g   = oz / TROY
            per_g_inr = oz_inr / TROY
            lines.append(
                f"• {metal.capitalize()} — spot price {now_utc_str()}\n"
                f"  International price:  {_f(oz)} USD / troy oz\n"
                f"  International price:  {_f(oz_inr,'₹')} INR / troy oz\n"
                f"  India — per gram:   {_f(per_g_inr,'₹')}\n"
                f"  India — per 10g:    {_f(per_g_inr*10,'₹')}\n"
                f"  India — per 100g:   {_f(per_g_inr*100,'₹')}\n"
                f"  India — per kg:     {_f(per_g_inr*1000,'₹')}\n"
                f"  (Note: actual retail prices include making charges/GST — "
                f"add 5–15% over spot for jewellery)\n"
                f"  (USD/INR rate used: {usd_inr:.2f})")
        if not lines: return None
        print(f"[metals] ✓ {metals}")
        return {"source":"metals.live + open.er-api.com (live)","title":"Live Metal Prices",
                "url":"https://metals.live","content":"\n\n".join(lines)}
    except Exception as e:
        print(f"[metals] {e}")
        return None


# ── 3. FOREX ──────────────────────────────────────────────────────────
FX_NAMES = {
    "dollar":"USD","usd":"USD","us dollar":"USD","american dollar":"USD",
    "rupee":"INR","inr":"INR","indian rupee":"INR",
    "euro":"EUR","eur":"EUR",
    "pound":"GBP","gbp":"GBP","sterling":"GBP",
    "yen":"JPY","jpy":"JPY","japanese yen":"JPY",
    "yuan":"CNY","cny":"CNY","chinese yuan":"CNY",
    "aud":"AUD","australian dollar":"AUD",
    "cad":"CAD","canadian dollar":"CAD",
    "sgd":"SGD","singapore dollar":"SGD",
    "aed":"AED","dirham":"AED","uae":"AED",
    "chf":"CHF","swiss franc":"CHF",
}
FX_TRIGGER = {"forex","exchange rate","currency","dollar","rupee","euro","pound",
              "usd","inr","eur","gbp","jpy","cny","aud","cad","sgd","aed",
              "to inr","to usd","to rupee","conversion","convert"}

def search_forex(q: str) -> dict | None:
    lo = q.lower()
    if not any(w in lo for w in FX_TRIGGER): return None
    frm = to = None
    m = re.search(r'(\w[\w\s]*?)\s+to\s+(\w[\w\s]*?)(?:\s+rate|\s+price|\s+exchange|\?|$)',
                  lo)
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
        data   = r.json()
        rates  = data.get("rates",{})
        upd    = data.get("time_last_update_utc", now_utc_str())
        show   = [to] + [c for c in ["USD","INR","EUR","GBP","JPY","AED"]
                         if c not in (frm, to)][:4]
        lines  = [f"Currency: {frm} — as of {upd}"]
        for c in show:
            v = rates.get(c)
            if v: lines.append(f"  1 {frm} = {v:,.4f} {c}")
        print(f"[forex] ✓ {frm}")
        return {"source":"Open Exchange Rates (live)","title":f"Rate: {frm}",
                "url":"https://open.er-api.com","content":"\n".join(lines)}
    except Exception as e:
        print(f"[forex] {e}"); return None


# ── 4. PETROL / DIESEL ────────────────────────────────────────────────
#  No reliable free API exists for Indian city-wise fuel prices.
#  We return an explicit "no live data" block so the AI tells the truth
#  instead of hallucinating ₹1000/litre or any wrong number.
FUEL_WORDS = {"petrol","diesel","fuel","lpg","cng","gas price","gasoline"}

def search_fuel(q: str) -> dict | None:
    if not any(w in q.lower() for w in FUEL_WORDS): return None
    content = (
        f"FUEL PRICE DATA — {now_utc_str()}\n\n"
        "IMPORTANT: No real-time API is available for Indian city-wise petrol/diesel "
        "prices. Do NOT guess or make up a price.\n\n"
        "Tell the user:\n"
        "• Petrol prices in India vary by city and change on the 1st of each month.\n"
        "• Typical range (as of early 2026): ₹94–₹106 per litre for petrol, "
        "₹87–₹95 per litre for diesel (major metro cities).\n"
        "• For the exact current price in their city, they should check:\n"
        "  - Indian Oil Corporation: https://iocl.com\n"
        "  - HP Petrol Pump app or https://hindustanpetroleum.com\n"
        "  - Google: search 'petrol price [city name] today'\n"
        "  - SMS: HPSMSPRICE to 9222201122 (HPCL)\n\n"
        "DO NOT state any specific per-litre price as current fact."
    )
    print("[fuel] returning no-live-data notice")
    return {"source":"No live fuel API — guidance only","title":"Petrol/Diesel Prices",
            "url":"https://iocl.com","content":content}


# ── 5. WEATHER ────────────────────────────────────────────────────────
def _city(q: str) -> str | None:
    lo = q.lower()
    if not any(w in lo for w in {"weather","temperature","forecast","humidity",
                                  "rain","wind","climate","hot","cold","sunny"}):
        return None
    for pat in [
        r'weather\s+(?:in|at|for|of)?\s*([A-Za-z][A-Za-z\s]{1,24})',
        r'([A-Za-z][A-Za-z\s]{1,24})\s+weather',
        r'(?:temperature|forecast)\s+(?:in|of|at)?\s*([A-Za-z][A-Za-z\s]{1,24})',
    ]:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            c = re.sub(r'\b(today|now|currently|like|is|the|a|an)\b','',
                       m.group(1), flags=re.IGNORECASE).strip()
            if len(c) > 1: return c
    return None

def search_weather(q: str) -> dict | None:
    city = _city(q)
    if not city: return None
    try:
        r = requests.get(f"https://wttr.in/{requests.utils.quote(city)}",
                         params={"format":"j1"},
                         headers={"User-Agent":"ZippyAI/2.0"}, timeout=8)
        r.raise_for_status()
        d   = r.json()
        cur = d["current_condition"][0]
        desc     = cur.get("weatherDesc",[{}])[0].get("value","N/A")
        temp_c   = cur.get("temp_C","N/A"); temp_f = cur.get("temp_F","N/A")
        feels    = cur.get("FeelsLikeC","N/A"); humidity = cur.get("humidity","N/A")
        wind     = cur.get("windspeedKmph","N/A"); uv = cur.get("uvIndex","N/A")
        fc = []
        for day in d.get("weather",[])[:3]:
            date  = day.get("date","")
            maxc  = day.get("maxtempC","N/A"); minc = day.get("mintempC","N/A")
            desc2 = (day.get("hourly") or [{}])[4].get(
                "weatherDesc",[{}])[0].get("value","N/A")
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
        return {"source":"wttr.in (live)","title":f"Weather in {city}",
                "url":f"https://wttr.in/{city}","content":content}
    except Exception as e:
        print(f"[weather] {e}"); return None


# ── 6. NEWS (Google News + 3 fallbacks, RECENT ONLY) ─────────────────
NEWS_MAX_AGE_DAYS = 7   # only show articles published in last 7 days

def _article_age_days(pub_date_str: str) -> float:
    """Return how many days ago an RSS pubDate was published. 999 if unparseable."""
    if not pub_date_str: return 999
    try:
        dt = parsedate_to_datetime(pub_date_str)
        dt = dt.astimezone(timezone.utc)
        age = (_utc_now() - dt).total_seconds() / 86400
        return max(0.0, age)
    except Exception:
        return 999

def _fetch_rss(url: str, src: str, keywords: list[str],
               max_items: int = 20) -> list[dict]:
    try:
        r = requests.get(url,
                         headers={**BASE_H,"Accept":"application/rss+xml,text/xml,*/*"},
                         timeout=9)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out  = []
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            src_e = (item.findtext("source") or src).strip()
            desc  = re.sub(r'<[^>]+>','',(item.findtext("description") or "")).strip()
            title_clean = re.sub(r'\s*-\s*[^-]{3,45}$','',title).strip() or title

            age = _article_age_days(pub)
            # Skip articles older than NEWS_MAX_AGE_DAYS
            if age > NEWS_MAX_AGE_DAYS:
                continue

            score = sum(1 for kw in keywords if kw.lower() in (title+desc).lower())
            if title_clean:
                out.append({"title":title_clean,"desc":desc[:280],
                            "pub":pub,"source":src_e,"link":link,
                            "score":score,"age":age})
        return out
    except Exception as e:
        print(f"[rss] {src}: {e}"); return []

NEWS_FEEDS = [
    # Primary: Google News searches all publishers
    ("Google News",
     "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q={q}"),
    # India-specific reliable feeds
    ("Times of India",
     "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("The Hindu",
     "https://www.thehindu.com/news/feeder/default.rss"),
    # International
    ("Al Jazeera",
     "https://www.aljazeera.com/xml/rss/all.xml"),
    # Tech
    ("BBC Technology",
     "https://feeds.bbci.co.uk/news/technology/rss.xml"),
]

def search_news(q: str, max_results: int = 6) -> dict | None:
    keywords = [w for w in re.split(r'\W+', q.lower()) if len(w) > 2]
    all_a: list[dict] = []
    for name, tmpl in NEWS_FEEDS:
        url = tmpl.replace("{q}", requests.utils.quote(q))
        arts = _fetch_rss(url, name, keywords, max_items=20)
        print(f"[news] {name} → {len(arts)} recent articles")
        all_a.extend(arts)

    if not all_a:
        # If ALL feeds returned nothing recent, try without age filter
        print("[news] no recent articles — retrying without age filter")
        for name, tmpl in NEWS_FEEDS[:2]:
            url = tmpl.replace("{q}", requests.utils.quote(q))
            try:
                r = requests.get(url,
                    headers={**BASE_H,"Accept":"application/rss+xml,text/xml,*/*"},
                    timeout=9)
                r.raise_for_status()
                root = ET.fromstring(r.content)
                for item in root.findall(".//item")[:10]:
                    title = re.sub(r'\s*-\s*[^-]{3,45}$','',
                                   (item.findtext("title") or "").strip()).strip()
                    link  = (item.findtext("link") or "").strip()
                    pub   = (item.findtext("pubDate") or "").strip()
                    desc  = re.sub(r'<[^>]+>','',
                                   (item.findtext("description") or "")).strip()
                    if title:
                        all_a.append({"title":title,"desc":desc[:280],
                                      "pub":pub,"source":name,"link":link,
                                      "score":1,"age":999})
            except Exception:
                pass

    if not all_a: return None

    # Deduplicate by title
    seen: set[str] = set()
    unique: list[dict] = []
    for a in all_a:
        key = re.sub(r'\W+','',a["title"].lower())[:40]
        if key not in seen:
            seen.add(key); unique.append(a)

    # Sort: highest relevance first, then most recent
    unique.sort(key=lambda x: (-x["score"], x["age"]))
    top = unique[:max_results]

    header = f"News for '{q}' — fetched {now_utc_str()}:\n"
    bullets = []
    for a in top:
        b = f"• **{a['title']}**"
        if a["desc"]: b += f"\n  {a['desc']}"
        b += f"\n  Source: {a['source']}"
        if a["pub"]: b += f"  |  Published: {a['pub']}"
        if a["link"]: b += f"\n  {a['link']}"
        bullets.append(b)

    print(f"[news] ✓ {len(top)} articles returned")
    return {"source":"Google News / TOI / The Hindu / Al Jazeera (live RSS)",
            "title":f"News: {q}",
            "url":f"https://news.google.com/search?q={requests.utils.quote(q)}",
            "content": header + "\n\n".join(bullets)}


# ── 7. WIKIPEDIA ──────────────────────────────────────────────────────
def search_wikipedia(q: str) -> dict | None:
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action":"query","list":"search","srsearch":q,
                                 "srlimit":1,"format":"json","origin":"*"},
                         headers=BASE_H, timeout=7)
        r.raise_for_status()
        res = r.json().get("query",{}).get("search",[])
        if not res: return None
        title = res[0]["title"]
    except Exception as e:
        print(f"[wiki] search: {e}"); return None
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action":"query","prop":"extracts","exintro":False,
                                 "explaintext":True,"titles":title,"format":"json",
                                 "origin":"*","exchars":3000},
                         headers=BASE_H, timeout=8)
        r.raise_for_status()
        for pg in r.json().get("query",{}).get("pages",{}).values():
            txt = pg.get("extract","")
            if txt and len(txt) > 80:
                url = ("https://en.wikipedia.org/wiki/"
                       + requests.utils.quote(title.replace(' ','_')))
                print(f"[wiki] ✓ '{title}'")
                return {"source":"Wikipedia","title":title,"url":url,"content":txt}
    except Exception as e:
        print(f"[wiki] extract: {e}")
    return None


# ── 8. REST COUNTRIES ─────────────────────────────────────────────────
def search_country(q: str) -> dict | None:
    if not any(w in q.lower() for w in
               {"capital","population","currency","language","country","nation","area"}):
        return None
    m = re.search(
        r'(?:of|in|about|for|capital of|population of|currency of)\s+'
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', q)
    if not m:
        m = re.search(r'\b([A-Z][a-z]{3,})\b', q)
    if not m: return None
    name = m.group(1).strip()
    try:
        r = requests.get(f"https://restcountries.com/v3.1/name/{requests.utils.quote(name)}",
                         headers=BASE_H, timeout=7)
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
            f"  Languages:  {', '.join(c.get('languages',{}).values()) or 'N/A'}")
        print(f"[country] ✓ {common}")
        return {"source":"REST Countries","title":f"Country: {common}",
                "url":"https://restcountries.com","content":content}
    except Exception as e:
        print(f"[country] {e}"); return None


# ── Master search ─────────────────────────────────────────────────────
def run_search(q: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    def add(item):
        if not item: return
        k = item.get("url") or item.get("title","")
        if k and k not in seen: seen.add(k); results.append(item)
    add(search_crypto(q))
    add(search_metals(q))
    add(search_forex(q))
    add(search_fuel(q))        # returns guidance, never a fake price
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
        lines += [
            f"── SOURCE {i}: {r['source']} ──",
            f"Title: {r['title']}",
            f"URL:   {r['url']}",
            "",
            r["content"],
            "",
        ]
    lines += [
        "── END OF LIVE DATA ──",
        "",
        "⚠ MANDATORY: Use the data above to answer. Do NOT ignore it.",
        "⚠ MANDATORY: Do NOT say 'I don't have real-time access'.",
        "⚠ MANDATORY: Do NOT say 'I cannot provide current prices'.",
        "⚠ MANDATORY: State prices/values in your FIRST sentence.",
        "⚠ MANDATORY: For news, use bullet points.",
        f"⚠ MANDATORY: Today is {today_str()}. Use this date. Never assume another year.",
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

# ─────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  — date always at the top, impossible to miss
# ─────────────────────────────────────────────────────────────────────
def make_system(ctx: str = "") -> str:
    td = today_str(); tu = now_utc_str(); yr = current_year()

    # Date block — first thing the model reads
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
- Greetings: one short sentence only.

## RESPONSE RULES
1. Answer exactly what was asked. Nothing extra.
2. Prices/numbers → state them in the VERY FIRST sentence.
3. Simple questions → 1-3 sentences max.
4. News / current events → ALWAYS bullet points, one per story.
5. Explanations → short bullets or brief paragraphs.
6. Code → give directly with brief comments.
7. Math → show steps briefly.
8. Creative → complete the piece, no preamble.
9. FORBIDDEN phrases: "Certainly!", "Great question!", "As an AI",
   "I don't have real-time access", "my training data is limited",
   "I cannot provide current prices", "I'd be happy to".
10. Never repeat the question.
11. End with exactly 1 relevant emoji.

## PETROL / DIESEL PRICES
No real-time API exists for Indian fuel prices. If search data says "no live data",
tell the user the typical range and direct them to iocl.com or hpcl.com.
NEVER guess or invent a specific per-litre price.
"""

    if not ctx:
        return base

    return base + f"""

## ⚠ LIVE DATA BELOW — YOU MUST USE THIS ⚠

{ctx}

## HOW TO USE THE LIVE DATA
- READ every source carefully. The exact figures are there.
- Use those exact numbers in your reply — do not use training-data estimates.
- State today's date as {td} if asked.
- NEVER say 'I don't have real-time access' — you have live data above.
- NEVER say 'I cannot provide current prices' — they are shown above.
- If a price source says "no live data" for fuel, tell the user honestly and give the official URL.
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
#  REQUEST
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
    return {
        "status": "Zippy running",
        "date":   today_str(),
        "time":   now_utc_str(),
        "models": {
            m["name"]: ("available" if model_cooldown.get(m["id"],0) <= now
                        else f"cooling {fmt_wait(model_cooldown[m['id']]-now)}")
            for m in CHAT_MODELS
        },
    }


@app.post("/chat")
def chat(req: ChatRequest):

    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message, flags=re.IGNORECASE
    ).strip()

    if not user_input:
        return {"reply":"Didn't catch that — say again? 😊",
                "thinking":"Empty.","searched":False,"sources":[]}

    tl = user_input.lower().strip()

    if IMG_RE.search(user_input):
        return {"reply":IMG_DECLINE,"thinking":"Image request.",
                "searched":False,"sources":[],"model_used":"static"}

    if tl in IDENTITY:
        return {"reply":IDENTITY[tl],"thinking":"Identity.",
                "searched":False,"sources":[],"model_used":"static"}
    if tl in SOCIAL:
        return {"reply":SOCIAL[tl],"thinking":"Social.",
                "searched":False,"sources":[],"model_used":"static"}

    # Pure AI thinking
    think        = run_thinking(user_input)
    needs_search = think.get("needs_search", False)
    sq           = (think.get("search_query") or "").strip() or user_input
    reasoning    = think.get("reasoning","")

    # Search
    ctx     = ""
    sources = []
    found   = False

    if needs_search:
        results = run_search(sq)
        ctx     = build_context(sq, results)
        if results:
            found   = True
            sources = [{"title":r["title"],"url":r["url"],"source":r["source"]}
                       for r in results]

    # System prompt with date + optional live data
    system = make_system(ctx)
    msgs   = [{"role":"system","content":system}]

    for h in req.history[-20:]:
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            msgs.append({"role":h["role"],"content":str(h["content"])[:1500]})

    # User message — double-anchor with date and data reminder
    if ctx:
        user_msg = (
            f"[Today is {today_str()}. "
            f"You have live data in the system context above. "
            f"Use those exact figures in your answer.]\n\n"
            f"{user_input}"
        )
    else:
        user_msg = f"[Today is {today_str()}]\n\n{user_input}"

    msgs.append({"role":"user","content":user_msg})

    try:
        reply, model = call_patient(msgs, CHAT_MODELS,
                                    max_tokens=700, temperature=0.65,
                                    top_p=0.9, max_wait=50.0)
        return {"reply":reply,"thinking":reasoning,
                "searched":found,"sources":sources,"model_used":model}

    except RuntimeError as exc:
        parts = str(exc).split("|",2)
        wait  = parts[1] if len(parts)==3 else "a few minutes"
        tried = "\n".join(f"• {l}" for l in parts[2].split("||") if l.strip()) \
                if len(parts)==3 else str(exc)
        return {
            "reply":(
                f"⚠️ **All AI models are rate-limited right now.**\n\n"
                f"Please wait at least **{wait}** and try again.\n\n"
                f"*Groq free tier: 30 req/min · 1,000 req/day per model. "
                f"All 5 models temporarily exhausted.*\n\n"
                f"**Models tried:**\n{tried}"
            ),
            "thinking":"All models rate-limited.",
            "searched":False,"sources":[],"model_used":"none",
        }
