from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import re
import time
import random
import base64
import requests
from collections import defaultdict
from urllib.parse import quote
import asyncio
from contextlib import asynccontextmanager

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

# IMPROVED SYSTEM PROMPT - Better search behavior
SYSTEM = """You are Zippy, a smart AI assistant.

**Your Capabilities:**
- Image generation (when users ask to generate/create/draw images)
- Image analysis (when users upload images)
- Web search for current information
- Coding, math, explanations, creative tasks

**When to Search the Web (CRITICAL):**
You MUST search for:
- Current events, news, recent happenings
- Stock prices, sports scores, weather
- Information that changes over time (who holds a position, what's currently happening, latest releases)
- Facts you're uncertain about
- Recent developments in any field
- When the user asks "what's happening", "latest", "current", "recent", "now", "today"

NEVER ask the user to search themselves. If you need current info, YOU search.

**Tone:**
- Smart, calm, helpful friend
- Natural and conversational
- Short greetings: "Hey! What's up?" or "Hi!"

**Response Length:**
- Greetings: 1-2 sentences
- Simple questions: 2-4 sentences
- Explanations: thorough with bullet points
- Code: COMPLETE working code
- Math: show all steps

**Rules:**
- NEVER say: "Certainly!", "Great question!", "Of course!", "Absolutely!", "As an AI"
- Only mention your creator when directly asked
- Don't ask the user for information you can search for
- Always end with 1 emoji

**Image Features:**
- You CAN generate images (tell users this when relevant)
- You CAN analyze uploaded images (tell users this when relevant)
"""

IDENTITY = {
    "who are you":       "I'm Zippy, a smart AI assistant! 🤖",
    "what are you":      "I'm Zippy, an AI assistant. 🤖",
    "who made you":      "I was made by Arul Vethathiri. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "who built you":     "I was built by Arul Vethathiri. 👨‍💻",
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

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ImageAnalysisRequest(BaseModel):
    image: str
    question: str = "What's in this image?"

@app.post("/analyze-image")
async def analyze_image(req: ImageAnalysisRequest):
    """Image analysis with better timeout handling"""
    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Try different models with longer timeout
        models_to_try = [
            ("Salesforce/blip-image-captioning-base", 45),  # Base model first (faster)
            ("nlpconnect/vit-gpt2-image-captioning", 45),
            ("Salesforce/blip-image-captioning-large", 60),
        ]
        
        for model_name, timeout in models_to_try:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
                
                print(f"Trying {model_name}...")
                
                response = requests.post(
                    API_URL,
                    headers={"Content-Type": "application/octet-stream"},
                    data=image_bytes,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    description = None
                    
                    if isinstance(result, list) and len(result) > 0:
                        if "generated_text" in result[0]:
                            description = result[0]["generated_text"]
                    elif isinstance(result, dict) and "generated_text" in result:
                        description = result["generated_text"]
                    
                    if description:
                        print(f"✅ Image analyzed: {description}")
                        response_text = f"I can see: {description}"
                        
                        return {
                            "description": description,
                            "response": response_text,
                            "success": True
                        }
                elif response.status_code == 503:
                    print(f"Model {model_name} loading...")
                    continue
                        
            except requests.Timeout:
                print(f"Model {model_name} timeout")
                continue
            except Exception as e:
                print(f"Model {model_name} failed: {str(e)}")
                continue
        
        # All models failed
        error_msg = "Image analysis is warming up (20-30 seconds first time). Please try again!"
        return {
            "description": error_msg,
            "response": error_msg,
            "success": False
        }
            
    except Exception as e:
        print(f"Image analysis error: {str(e)}")
        error_msg = "Having trouble analyzing this image. Please try again!"
        return {
            "description": error_msg,
            "response": error_msg,
            "success": False
        }

@app.get("/")
def root():
    return {"status": "Zippy backend is running!", "version": "5.0"}

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '', req.message, flags=re.IGNORECASE
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
    for msg in req.history:
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

# ===================================
# IMAGE GENERATION - MULTIPLE FREE PROVIDERS
# ===================================
class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
async def generate_image(body: ImageRequest):
    """
    Tries multiple free image generation services:
    1. Flux-schnell (fast and reliable)
    2. Stable Diffusion XL Lightning (fast)
    3. Stable Diffusion 2.1 (fallback)
    """
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    print(f"🎨 Generating: {prompt[:60]}...")

    # Service 1: Try flux-schnell (very fast, good quality)
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
            headers={"Content-Type": "application/json"},
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200 and len(response.content) > 1000:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"
            print("✅ Generated with flux-schnell")
            return {
                "imageUrl": image_url,
                "service": "flux-schnell",
                "success": True
            }
    except Exception as e:
        print(f"flux-schnell failed: {str(e)}")

    # Service 2: Try SDXL Lightning (very fast)
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/ByteDance/SDXL-Lightning",
            headers={"Content-Type": "application/json"},
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200 and len(response.content) > 1000:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"
            print("✅ Generated with SDXL-Lightning")
            return {
                "imageUrl": image_url,
                "service": "sdxl-lightning",
                "success": True
            }
    except Exception as e:
        print(f"SDXL-Lightning failed: {str(e)}")

    # Service 3: Try Stable Diffusion 2.1 (reliable fallback)
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
            headers={"Content-Type": "application/json"},
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200 and len(response.content) > 1000:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"
            print("✅ Generated with SD 2.1")
            return {
                "imageUrl": image_url,
                "service": "stable-diffusion",
                "success": True
            }
        elif response.status_code == 503:
            return {
                "imageUrl": "",
                "service": "loading",
                "success": False,
                "error": "Image generator warming up (20-30 seconds first time). Try again in a moment!"
            }
    except Exception as e:
        print(f"SD 2.1 failed: {str(e)}")

    # All failed
    return {
        "imageUrl": "",
        "service": "none",
        "success": False,
        "error": "Image generation temporarily unavailable. Try again in a moment!"
    }
