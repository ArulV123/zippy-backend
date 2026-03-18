# main.py - diagnostic + hardened Zippy backend
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import re
import time
import random
import base64
import asyncio
from collections import defaultdict
from urllib.parse import quote
from contextlib import asynccontextmanager
import httpx
import json

# Entities used once in this reply:
# :contentReference[oaicite:2]{index=2}
# :contentReference[oaicite:3]{index=3}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting...")
    yield
    print("🔴 Server shutting down")

app = FastAPI(lifespan=lifespan)

origins = ["https://arulv123.github.io", "http://localhost:8000", "http://127.0.0.1:8000", "*"]
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
]

model_usage = defaultdict(list)
model_blocked_until = {}
MAX_RPM = 18

def get_best_model():
    now = time.time()
    for model in MODELS:
        if model in model_blocked_until:
            if now < model_blocked_until[model]:
                continue
            else:
                del model_blocked_until[model]
        model_usage[model] = [t for t in model_usage[model] if now - t < 60]
        if len(model_usage[model]) < MAX_RPM:
            return model
    return None

SYSTEM = """You are Zippy, a smart AI assistant.
"""

IDENTITY = {
    "who are you": "I'm Zippy, a smart AI assistant! 🤖",
}
SOCIAL = {"thanks":"You're welcome! 😊"}

HARMFUL_KEYWORDS = [
    "how to make a bomb", "how to build a bomb", "build a nuke",
    "make a nuke", "how to make a weapon", "how to kill",
    "how to make drugs", "synthesize drugs", "make explosives",
    "child porn", "csam",
]
def is_harmful(text: str) -> bool:
    return any(k in text.lower() for k in HARMFUL_KEYWORDS)

HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "").strip()
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "").strip()

def to_data_url_png(content_bytes: bytes) -> str:
    b64 = base64.b64encode(content_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ImageRequest(BaseModel):
    prompt: str

class ImageAnalysisRequest(BaseModel):
    image: str
    question: str = "What's in this image?"

# -------------------------
# HEALTH endpoint - test external services and return status & small debug text
# -------------------------
@app.get("/health")
async def health():
    results = {"pollinations": None, "huggingface": None, "notes": []}
    async with httpx.AsyncClient() as client:
        # Pollinations quick GET
        poll_urls = [
            "https://image.pollinations.ai/prompt/test",
            "https://gen.pollinations.ai/prompt/test"
        ]
        for u in poll_urls:
            try:
                r = await client.get(u, timeout=10, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0","Accept":"image/png"})
                results["pollinations"] = {"url": u, "status": r.status_code, "content_type": r.headers.get("content-type",""), "len": len(r.content)}
                # include first bytes (text) if content-type isn't image
                if r.content and len(r.content) < 2000 and "image" not in r.headers.get("content-type",""):
                    results["pollinations"]["body_snippet"] = (r.text[:1000])
                break
            except Exception as e:
                results["notes"].append(f"pollinations {u} error: {str(e)}")
                results["pollinations"] = {"url": u, "status": "error", "error": str(e)}

        # Hugging Face model status (only HEAD/GET), but avoid exposing keys
        if HUGGINGFACE_API_KEY:
            try:
                hf_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
                r = await client.get(hf_url, headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}, timeout=10)
                results["huggingface"] = {"url": hf_url, "status": r.status_code, "content_type": r.headers.get("content-type","")}
                if r.status_code != 200:
                    # include snippet (if not binary)
                    try:
                        results["huggingface"]["body_snippet"] = r.text[:1000]
                    except Exception:
                        pass
            except Exception as e:
                results["huggingface"] = {"error": str(e)}
        else:
            results["huggingface"] = {"error": "HUGGINGFACE_API_KEY not set (set HUGGINGFACE_API_KEY env var)"}
    return results

# -------------------------
# DEBUG endpoint - runs a single generation + analysis and returns detailed logs (no secrets)
# -------------------------
@app.post("/debug")
async def debug_endpoint(body: ImageRequest):
    prompt = (body.prompt or "").strip()
    log = {"prompt": prompt, "pollinations_attempts": [], "hf_attempts": [], "analysis_attempts": []}
    # try generate (Pollinations)
    async with httpx.AsyncClient() as client:
        poll_url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        try:
            r = await client.get(poll_url, headers={"User-Agent":"Mozilla/5.0","Accept":"image/png"}, timeout=20, follow_redirects=True)
            log["pollinations_attempts"].append({"url": poll_url, "status": r.status_code, "ctype": r.headers.get("content-type",""), "len": len(r.content), "text_snippet": (r.text[:400] if r.text and "image" not in r.headers.get("content-type","") else "")})
        except Exception as e:
            log["pollinations_attempts"].append({"url": poll_url, "error": str(e)})
        # HuggingFace quick test if key present
        if HUGGINGFACE_API_KEY:
            try:
                hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
                r = await client.post(hf_url, headers={"Authorization":f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type":"application/json"}, json={"inputs":prompt}, timeout=30)
                log["hf_attempts"].append({"url": hf_url, "status": r.status_code, "ctype": r.headers.get("content-type",""), "len": len(r.content), "text_snippet": (r.text[:400] if r.text and "image" not in r.headers.get("content-type","") else "")})
            except Exception as e:
                log["hf_attempts"].append({"url": hf_url, "error": str(e)})
        else:
            log["hf_attempts"].append({"error": "HUGGINGFACE_API_KEY not set"})
    return log

# -------------------------
# IMAGE GENERATION (internal helper with verbose logging)
# -------------------------
async def generate_image_internal(prompt: str):
    if not prompt or not prompt.strip():
        return {"success": False, "error": "empty prompt", "service": "none"}
    prompt = prompt.strip()
    encoded_prompt = quote(prompt)
    seed = random.randint(1, 999999)
    hf_headers = {}
    if HUGGINGFACE_API_KEY:
        hf_headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"

    attempts_log = []
    async with httpx.AsyncClient(timeout=None) as client:
        # Try multiple pollinations-like hosts (some hosts may be down or changed)
        pollinations_hosts = [
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width=1024&height=1024",
            f"https://gen.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width=1024&height=1024",
        ]
        for url in pollinations_hosts:
            try:
                r = await client.get(url, headers={"User-Agent":"Mozilla/5.0","Accept":"image/png"}, timeout=30, follow_redirects=True)
                attempts_log.append({"service":"pollinations", "url":url, "status": r.status_code, "ctype": r.headers.get("content-type",""), "len": len(r.content), "text_snippet": (r.text[:800] if r.text and "image" not in r.headers.get("content-type","") else "")})
                if r.status_code == 200 and r.content and len(r.content) > 500:
                    return {"success": True, "imageUrl": to_data_url_png(r.content), "service": "pollinations", "attempts": attempts_log}
            except Exception as e:
                attempts_log.append({"service":"pollinations", "url":url, "error": str(e)})

        # HF fallbacks
        hf_models = [
            "black-forest-labs/FLUX.1-schnell",
            "ByteDance/SDXL-Lightning",
            "stabilityai/stable-diffusion-2-1"
        ]
        for model_name in hf_models:
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            payload = {"inputs": prompt, "options": {"wait_for_model": True}}
            try:
                r = await client.post(api_url, headers={**hf_headers, "Content-Type":"application/json", "Accept":"application/octet-stream"}, json=payload, timeout=90)
                ctype = r.headers.get("content-type","")
                entry = {"service":"huggingface", "model":model_name, "status": r.status_code, "ctype": ctype, "len": len(r.content)}
                if r.status_code != 200:
                    # include snippet for debugging (not secrets)
                    try:
                        entry["text_snippet"] = r.text[:1000]
                    except Exception:
                        pass
                    attempts_log.append(entry)
                    continue
                # if binary bytes returned
                if r.content and len(r.content) > 100:
                    return {"success": True, "imageUrl": to_data_url_png(r.content), "service": model_name, "attempts": attempts_log + [entry]}
                # if JSON returned, maybe b64 inside
                if "application/json" in ctype:
                    try:
                        data = r.json()
                        for key in ("image", "image_base64", "b64_json", "base64", "generated_image"):
                            if isinstance(data, dict) and key in data and isinstance(data[key], str) and len(data[key])>100:
                                try:
                                    b = base64.b64decode(data[key])
                                    return {"success": True, "imageUrl": to_data_url_png(b), "service": model_name, "attempts": attempts_log + [entry]}
                                except Exception:
                                    pass
                    except Exception:
                        pass
                attempts_log.append(entry)
            except Exception as e:
                attempts_log.append({"service":"huggingface","model":model_name,"error":str(e)})
        # nothing worked
        return {"success": False, "error": "All image services failed or warming up.", "attempts": attempts_log}

# public endpoint that uses internal helper
@app.post("/generate-image")
async def generate_image(body: ImageRequest):
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    result = await generate_image_internal(prompt)
    return result

# -------------------------
# IMAGE ANALYSIS (resilient, returns clear errors)
# -------------------------
@app.post("/analyze-image")
async def analyze_image(req: ImageAnalysisRequest):
    try:
        image_data = req.image or ""
        if "," in image_data:
            image_data = image_data.split(",",1)[1]
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        if not HUGGINGFACE_API_KEY:
            return {"success": False, "error": "HUGGINGFACE_API_KEY not set. Set env var and restart."}

        models_to_try = [
            ("Salesforce/blip-image-captioning-base", 30),
            ("nlpconnect/vit-gpt2-image-captioning", 30),
            ("Salesforce/blip-image-captioning-large", 60),
        ]
        attempts = []
        async with httpx.AsyncClient(timeout=None) as client:
            for model_name, timeout in models_to_try:
                api_url = f"https://api-inference.huggingface.co/models/{model_name}"
                try:
                    resp = await client.post(api_url, headers={"Authorization":f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type":"application/octet-stream"}, content=image_bytes, params={"wait_for_model":"true"}, timeout=timeout)
                except Exception as e:
                    attempts.append({"model": model_name, "error": str(e)})
                    continue
                attempts.append({"model": model_name, "status": resp.status_code, "ctype": resp.headers.get("content-type",""), "len": len(resp.content), "text_snippet": (resp.text[:800] if resp.text and "application/json" in resp.headers.get("content-type","") else "")})
                if resp.status_code == 200:
                    ctype = resp.headers.get("content-type","")
                    if "application/json" in ctype:
                        try:
                            data = resp.json()
                        except Exception:
                            data = None
                        # try common keys
                        if isinstance(data, list) and len(data)>0 and isinstance(data[0], dict) and "generated_text" in data[0]:
                            desc = data[0]["generated_text"]
                            return {"success": True, "description": desc, "attempts": attempts}
                        if isinstance(data, dict):
                            for key in ("generated_text","caption","result","description","text"):
                                if key in data and isinstance(data[key], str):
                                    return {"success": True, "description": data[key], "attempts": attempts}
                    # fallback to plain text
                    text = resp.text.strip()
                    if text:
                        return {"success": True, "description": text[:4000], "attempts": attempts}
                elif resp.status_code == 503:
                    # model warming, try next
                    continue
                else:
                    continue
        return {"success": False, "error": "All analysis models failed or warming up.", "attempts": attempts}
    except Exception as e:
        return {"success": False, "error": f"Unexpected: {str(e)}"}

# -------------------------
# CHAT (keeps your logic; auto image decision uses the LLM)
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    user_input = re.sub(r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+','', req.message or "", flags=re.IGNORECASE).strip()
    if not user_input:
        return {"reply":"Didn't catch that, say again? 😊"}
    text = user_input.lower().strip()
    if text in IDENTITY:
        return {"reply": IDENTITY[text]}
    if text in SOCIAL:
        return {"reply": SOCIAL[text]}
    if is_harmful(text):
        return {"reply":"That's not something I can help with. Ask me something else! 😊"}

    clean_history = []
    for msg in req.history or []:
        try:
            role = msg.get("role","") if isinstance(msg, dict) else ""
            content = msg.get("content","") if isinstance(msg, dict) else ""
            if role in ("user","assistant") and isinstance(content,str) and content.strip():
                clean_history.append({"role": role, "content": content.strip()})
        except:
            continue
    clean_history = clean_history[-20:]
    filtered=[]
    last_role="assistant"
    for msg in clean_history:
        if msg["role"] != last_role:
            filtered.append(msg)
            last_role=msg["role"]

    messages=[{"role":"system","content":SYSTEM}]
    messages += filtered
    messages.append({"role":"user","content":user_input})

    for attempt in range(len(MODELS)):
        model = get_best_model()
        if model is None:
            time.sleep(5)
            model = get_best_model()
        if model is None:
            return {"reply":"Couldn't connect right now, try again in a moment! 🙏"}
        try:
            model_usage[model].append(time.time())
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=0.85,
                top_p=0.92,
            )
            reply = response.choices[0].message.content.strip()
            # ask the LLM whether to generate an image (tiny prompt, deterministic)
            wants_image = False
            try:
                sg_resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":"Answer with just YES or NO. Does the user's request require generating a new illustrative image?"},{"role":"user","content":user_input}],
                    max_tokens=5,
                    temperature=0.0
                )
                sg_text = sg_resp.choices[0].message.content.strip().lower()
                wants_image = "yes" in sg_text
            except Exception as e:
                print("image-decision failed:", e)
                wants_image = False

            if wants_image:
                # call internal image generator (async) and wait
                try:
                    result = asyncio.run(generate_image_internal(user_input))
                    if result.get("success"):
                        return {"reply": reply, "image": result.get("imageUrl"), "service": result.get("service")}
                    else:
                        # include attempts for debugging
                        return {"reply": reply, "image_generation_failed": result}
                except Exception as e:
                    print("image generation error from chat:", e)
            return {"reply": reply}
        except Exception as e:
            err = str(e)
            print(f"❌ {model} failed: {err}")
            if "429" in err or "rate_limit" in err:
                model_blocked_until[model] = time.time() + 60
                model_usage[model] = []
                continue
            elif "not found" in err.lower() or "invalid" in err.lower():
                model_blocked_until[model] = time.time() + 3600
                continue
            else:
                break
    return {"reply":"Couldn't connect right now, try again in a moment! 🙏"}
