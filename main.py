# main.py - Full Zippy backend (Pollinations primary, HF fallbacks, image analysis, AI-driven image generation)
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

# ----------------------------------------------------------------------
# Basic app + lifecycle
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Model rotation / usage tracking
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# System prompt, identity, helpers
# ----------------------------------------------------------------------
SYSTEM = """You are Zippy, a smart AI assistant.

**Your Capabilities:**
- Image generation (when users ask to generate/create/draw images)
- Image analysis (when users upload images)
- Web search for current information
- Coding, math, explanations, creative tasks
"""

IDENTITY = {
    "who are you":       "I'm Zippy, a smart AI assistant! 🤖",
    "what are you":      "I'm Zippy, an AI assistant. 🤖",
    "who made you":      "I was made by Arul Vethathiri. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "what is your name": "My name is Zippy! 😊",
    "are you an ai":     "Yes! I'm Zippy, an AI assistant. 🤖",
    "are you human":     "Nope! I'm an AI but great at conversation! 😄",
}

SOCIAL = {
    "thanks":     "You're welcome! 😊",
    "thank you":  "Happy to help! 🌟",
    "bye":        "Take care! 👋",
    "goodbye":    "See you! 👋",
    "good night": "Good night! 🌙",
}

HARMFUL_KEYWORDS = [
    "how to make a bomb", "how to build a bomb", "build a nuke",
    "make a nuke", "how to make a weapon", "how to kill",
    "how to make drugs", "synthesize drugs", "make explosives",
    "child porn", "csam",
]

def is_harmful(text: str) -> bool:
    return any(k in text.lower() for k in HARMFUL_KEYWORDS)

# Environment keys
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")

# ----------------------------------------------------------------------
# Simple SerpAPI helper (optional) - synchronous but small
# ----------------------------------------------------------------------
def web_search_sync(query: str, max_results: int = 3):
    results = []
    if not SERPAPI_API_KEY:
        return results
    try:
        import requests as _requests
        url = "https://serpapi.com/search.json"
        params = {"q": query, "api_key": SERPAPI_API_KEY, "num": max_results}
        r = _requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("organic_results", [])[:max_results]:
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet") or item.get("snippet_text") or "",
                    "link": item.get("link") or item.get("url")
                })
    except Exception as e:
        print("web_search_sync error:", e)
    return results

# ----------------------------------------------------------------------
# Request/response models
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: list = []

class ImageAnalysisRequest(BaseModel):
    image: str
    question: str = "What's in this image?"

class ImageRequest(BaseModel):
    prompt: str

# ----------------------------------------------------------------------
# Helper: convert content to data-url base64
# ----------------------------------------------------------------------
def to_data_url_png(content_bytes: bytes) -> str:
    b64 = base64.b64encode(content_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ----------------------------------------------------------------------
# IMAGE ANALYSIS (async, httpx, retries, HF models)
# ----------------------------------------------------------------------
@app.post("/analyze-image")
async def analyze_image(req: ImageAnalysisRequest):
    try:
        image_data = req.image or ""
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        hf_headers = {}
        if HUGGINGFACE_API_KEY:
            hf_headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"

        models_to_try = [
            ("Salesforce/blip-image-captioning-base", 30),
            ("nlpconnect/vit-gpt2-image-captioning", 30),
            ("Salesforce/blip-image-captioning-large", 60),
        ]

        async with httpx.AsyncClient(timeout=None) as client:
            for model_name, timeout in models_to_try:
                api_url = f"https://api-inference.huggingface.co/models/{model_name}"
                print(f"Trying {model_name} for analysis...")
                try:
                    resp = await client.post(
                        api_url,
                        headers={**hf_headers, "Content-Type": "application/octet-stream"},
                        content=image_bytes,
                        params={"wait_for_model": "true"},
                        timeout=timeout
                    )
                except httpx.TimeoutException:
                    print(f"{model_name} timeout")
                    continue
                except Exception as e:
                    print(f"{model_name} request failed: {e}")
                    continue

                if resp.status_code == 200:
                    ctype = resp.headers.get("content-type", "")
                    if "application/json" in ctype:
                        try:
                            data = resp.json()
                        except Exception:
                            data = None
                        description = None
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            if "generated_text" in data[0]:
                                description = data[0]["generated_text"]
                        elif isinstance(data, dict):
                            if "generated_text" in data:
                                description = data["generated_text"]
                            if not description:
                                for key in ("caption", "result", "description", "text"):
                                    if key in data and isinstance(data[key], str):
                                        description = data[key]
                                        break

                        if description:
                            print(f"✅ Image analyzed by {model_name}: {description}")
                            return {"description": description, "response": f"I can see: {description}", "success": True}
                        else:
                            text = resp.text.strip()
                            if text:
                                print(f"⚠️ Parsed text fallback from {model_name}")
                                return {"description": text[:4000], "response": f"I can see: {text[:4000]}", "success": True}
                    else:
                        text = resp.text.strip()
                        if text:
                            return {"description": text[:4000], "response": f"I can see: {text[:4000]}", "success": True}

                elif resp.status_code == 503:
                    print(f"{model_name} loading (503), trying next")
                    continue
                else:
                    print(f"{model_name} returned {resp.status_code}: {resp.text[:300]}")
                    continue

        error_msg = "Image analysis is warming up or unavailable. Try again in a few seconds."
        return {"description": error_msg, "response": error_msg, "success": False}

    except Exception as e:
        print("Image analysis error:", e)
        return {"description": "Having trouble analyzing this image. Please try again!", "response": "Having trouble analyzing this image. Please try again!", "success": False}

# ----------------------------------------------------------------------
# Root & chat (keeps your existing chat logic, with optional web search injection)
# ----------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "Zippy backend is running!", "version": "5.0"}

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message or "", flags=re.IGNORECASE
    ).strip()

    if not user_input:
        return {"reply": "Didn't catch that, say again? 😊"}

    text = user_input.lower().strip()

    if text in IDENTITY:
        return {"reply": IDENTITY[text]}
    if text in SOCIAL:
        return {"reply": SOCIAL[text]}
    if is_harmful(text):
        return {"reply": "That's not something I can help with. Ask me something else! 😊"}

    clean_history = []
    for msg in req.history or []:
        try:
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                clean_history.append({"role": role, "content": content.strip()})
        except:
            continue

    clean_history = clean_history[-20:]

    filtered = []
    last_role = "assistant"
    for msg in clean_history:
        if msg["role"] != last_role:
            filtered.append(msg)
            last_role = msg["role"]

    messages = [{"role": "system", "content": SYSTEM}]

    # Heuristic: if query likely needs current info, do a web search (SerpAPI required)
    try:
        if re.search(r'\b(latest|today|now|news|who is|who was|president|current|score|price)\b', user_input, flags=re.I):
            search_results = web_search_sync(user_input, max_results=3)
            if search_results:
                sr_text = "Web search results (top):\n"
                for i, r in enumerate(search_results, 1):
                    sr_text += f"{i}. {r.get('title') or ''}\n{r.get('snippet') or ''}\nSource: {r.get('link') or ''}\n\n"
                messages.append({"role": "system", "content": sr_text})
    except Exception as e:
        print("web search injection error:", e)

    messages += filtered
    messages.append({"role": "user", "content": user_input})

    for attempt in range(len(MODELS)):
        model = get_best_model()

        if model is None:
            time.sleep(5)
            model = get_best_model()

        if model is None:
            return {"reply": "Couldn't connect right now, try again in a moment! 🙏"}

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

            refusal_hints = [
                "cannot assist", "can't assist", "unable to assist",
                "i'm not able", "i cannot", "not able to help",
                "against my", "not appropriate", "harmful"
            ]
            if any(hint in reply.lower() for hint in refusal_hints):
                return {"reply": "That's not something I can help with. Ask me something else! 😊"}

            print(f"✅ Served by: {model}")

            # Decide whether to auto-generate an image (AI decides)
            try:
                # Ask the LLM (via groq client) a tiny yes/no question
                should_msg = [
                    {"role": "system", "content": "Answer with just YES or NO. Does the user's request require generating a new illustrative image?"},
                    {"role": "user", "content": user_input}
                ]
                sg_resp = client.chat.completions.create(
                    model=model,
                    messages=should_msg,
                    max_tokens=5,
                    temperature=0.0
                )
                sg_text = sg_resp.choices[0].message.content.strip().lower()
                wants_image = "yes" in sg_text
            except Exception as e:
                print("image-decision step failed:", e)
                wants_image = False

            if wants_image:
                # call internal async image generator from sync endpoint
                try:
                    generated = asyncio.run(generate_image_internal(user_input))
                    if generated and generated.get("success"):
                        return {"reply": reply, "image": generated.get("imageUrl")}
                except Exception as e:
                    print("image generation from chat failed:", e)

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

    return {"reply": "Couldn't connect right now, try again in a moment! 🙏"}

# ----------------------------------------------------------------------
# IMAGE GENERATION - Pollinations primary, HF fallbacks (async, robust)
# ----------------------------------------------------------------------
async def generate_image_internal(prompt: str):
    """
    Internal helper that tries Pollinations first, then Hugging Face models.
    Returns dict: {"success": bool, "imageUrl": str, "service": str, "error": str (optional)}
    """
    if not prompt or not prompt.strip():
        return {"success": False, "imageUrl": "", "service": "none", "error": "empty prompt"}

    prompt = prompt.strip()
    encoded_prompt = quote(prompt)
    seed = random.randint(1, 999999)

    hf_headers = {}
    if HUGGINGFACE_API_KEY:
        hf_headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"

    async with httpx.AsyncClient(timeout=None) as client:
        # ---------------------------
        # 1) Pollinations (primary, free, fast)
        # ---------------------------
        try:
            poll_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width=1024&height=1024"
            for attempt in range(3):
                try:
                    resp = await client.get(poll_url, timeout=60)
                except Exception as e:
                    print("Pollinations request error:", e)
                    resp = None

                if resp and resp.status_code == 200 and resp.content and len(resp.content) > 1000:
                    image_url = to_data_url_png(resp.content)
                    print("✅ Generated with Pollinations")
                    return {"imageUrl": image_url, "service": "pollinations", "success": True}
                else:
                    print(f"Pollinations attempt {attempt+1} failed: status={(resp.status_code if resp else 'no resp')}")
                    await asyncio.sleep(0.8)
        except Exception as e:
            print("Pollinations top-level error:", e)

        # Helper for HF fallbacks
        async def hf_generate(api_model: str, timeout_s: int = 90):
            api_url = f"https://api-inference.huggingface.co/models/{api_model}"
            payload = {"inputs": prompt, "options": {"wait_for_model": True}}
            try:
                resp = await client.post(
                    api_url,
                    headers={**hf_headers, "Content-Type": "application/json", "Accept": "application/octet-stream"},
                    json=payload,
                    timeout=timeout_s
                )
            except Exception as e:
                print(f"{api_model} request error:", e)
                return None

            if resp.status_code == 200:
                ctype = resp.headers.get("content-type", "")
                if "image" in ctype or "application/octet-stream" in ctype or (resp.content and len(resp.content) > 100):
                    return resp.content
                if "application/json" in ctype:
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                    for key in ("image", "image_base64", "b64_json", "base64", "generated_image"):
                        if isinstance(data, dict) and key in data and isinstance(data[key], str) and len(data[key]) > 100:
                            try:
                                return base64.b64decode(data[key])
                            except Exception:
                                pass
                if resp.content and len(resp.content) > 100:
                    return resp.content
                return None
            elif resp.status_code == 503:
                print(f"{api_model} reported model loading (503).")
                return None
            else:
                print(f"{api_model} returned {resp.status_code}: {resp.text[:200]}")
                return None

        # ---------------------------
        # 2) HuggingFace: FLUX
        # ---------------------------
        try:
            flux_bytes = await hf_generate("black-forest-labs/FLUX.1-schnell", timeout_s=90)
            if flux_bytes:
                return {"imageUrl": to_data_url_png(flux_bytes), "service": "flux-schnell", "success": True}
        except Exception as e:
            print("FLUX error:", e)

        # ---------------------------
        # 3) HuggingFace: SDXL Lightning
        # ---------------------------
        try:
            sdxl_bytes = await hf_generate("ByteDance/SDXL-Lightning", timeout_s=90)
            if sdxl_bytes:
                return {"imageUrl": to_data_url_png(sdxl_bytes), "service": "sdxl-lightning", "success": True}
        except Exception as e:
            print("SDXL error:", e)

        # ---------------------------
        # 4) HuggingFace: Stable Diffusion 2.1 fallback
        # ---------------------------
        try:
            sd_bytes = await hf_generate("stabilityai/stable-diffusion-2-1", timeout_s=120)
            if sd_bytes:
                return {"imageUrl": to_data_url_png(sd_bytes), "service": "stable-diffusion", "success": True}
        except Exception as e:
            print("SD 2.1 error:", e)

    # All services failed
    return {"imageUrl": "", "service": "none", "success": False, "error": "All image services failed or are warming up."}

@app.post("/generate-image")
async def generate_image(body: ImageRequest):
    """
    Public endpoint to generate image from a prompt.
    Uses the shared generate_image_internal helper.
    """
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    result = await generate_image_internal(prompt)
    return result

# ----------------------------------------------------------------------
# End of file
# ----------------------------------------------------------------------
