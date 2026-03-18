from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import re
import time
from collections import defaultdict
from urllib.parse import quote
from contextlib import asynccontextmanager

# ----------------------------
# App setup
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting...")
    yield
    print("🔴 Server shutting down")

app = FastAPI(lifespan=lifespan)

origins = [
    "https://arulv123.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ----------------------------
# Models and rate limiting
# ----------------------------
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
            del model_blocked_until[model]

        model_usage[model] = [t for t in model_usage[model] if now - t < 60]
        if len(model_usage[model]) < MAX_RPM:
            return model
    return None

# ----------------------------
# Prompts / rules
# ----------------------------
SYSTEM = """You are Zippy, a smart AI assistant.

You can:
- answer questions
- write code
- solve math
- generate images when a visual would help

Be natural, helpful, and clear.
"""

IDENTITY = {
    "who are you": "I'm Zippy, a smart AI assistant! 🤖",
    "what are you": "I'm Zippy, an AI assistant. 🤖",
    "who made you": "I was made by Arul Vethathiri. 👨‍💻",
    "who created you": "I was created by Arul Vethathiri. 👨‍💻",
    "who built you": "I was built by Arul Vethathiri. 👨‍💻",
    "what is your name": "My name is Zippy! 😊",
    "are you an ai": "Yes! I'm Zippy, an AI assistant. 🤖",
    "are you human": "Nope! I'm an AI but great at conversation! 😄",
}

SOCIAL = {
    "thanks": "You're welcome! 😊",
    "thank you": "Happy to help! 🌟",
    "bye": "Take care! 👋",
    "goodbye": "See you! 👋",
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

# ----------------------------
# Env helpers
# ----------------------------
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "").strip()

# ----------------------------
# Request models
# ----------------------------
class ChatRequest(BaseModel):
    message: str
    history: list = []

class ImageRequest(BaseModel):
    prompt: str

class ImageAnalysisRequest(BaseModel):
    image: str
    question: str = "What's in this image?"

# ----------------------------
# Image generation
# ----------------------------
def generate_image_pollinations(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return {
            "success": False,
            "service": "pollinations",
            "imageUrl": "",
            "error": "prompt required",
        }

    encoded = quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}"
    return {
        "success": True,
        "service": "pollinations",
        "imageUrl": url,
    }

@app.post("/generate-image")
async def generate_image(body: ImageRequest):
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    return generate_image_pollinations(prompt)

# ----------------------------
# Image analysis
# ----------------------------
@app.post("/analyze-image")
async def analyze_image(req: ImageAnalysisRequest):
    return {
        "success": False,
        "error": "Image analysis is not available with only a Groq API key. Add a vision-capable API key to enable it.",
        "response": "I cannot analyze images yet with this setup."
    }

# ----------------------------
# AI decision: should an image be generated?
# ----------------------------
def should_generate_image(user_input: str, assistant_reply: str) -> bool:
    try:
        decision = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only YES or NO. "
                        "Does the user request need a new illustrative image? "
                        "Say YES only when a visual would genuinely help."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User request: {user_input}\nAssistant reply: {assistant_reply}",
                },
            ],
            max_tokens=3,
            temperature=0.0,
        )
        ans = (decision.choices[0].message.content or "").strip().lower()
        return ans.startswith("yes")
    except Exception as e:
        print("should_generate_image error:", e)
        return False

# ----------------------------
# Root
# ----------------------------
@app.get("/")
def root():
    return {
        "status": "Zippy backend is running!",
        "version": "5.0",
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY", "").strip()),
    }

# ----------------------------
# Chat
# ----------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+|err+|ah+|oh+)\s+',
        '',
        req.message or '',
        flags=re.IGNORECASE
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
        except Exception:
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

    for _ in range(len(MODELS)):
        model = get_best_model()

        if model is None:
            time.sleep(5)
            model = get_best_model()

        if model is None:
            return {"reply": "Couldn't connect right now, try again in a moment! 🙏"}

        try:
            model_usage[model].append(time.time())

            response = groq_client.chat.completions.create(
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

            if should_generate_image(user_input, reply):
                img = generate_image_pollinations(user_input)
                if img["success"]:
                    return {
                        "reply": reply,
                        "image": img["imageUrl"],
                        "image_service": img["service"],
                    }

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
