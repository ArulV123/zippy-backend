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
    # Startup: Warm up image model
    try:
        tiny_image = base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82').decode()
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        await asyncio.to_thread(
            requests.post, API_URL, 
            headers={"Content-Type": "application/octet-stream"},
            data=base64.b64decode(tiny_image),
            timeout=30
        )
        print("✅ Image model warmed up")
    except:
        print("⚠️ Model warmup failed (will work on first real use)")
    
    yield
    print("🔴 Server shutting down")

app = FastAPI(lifespan=lifespan)

origins = [
    "https://arulv123.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*"
]

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

SYSTEM = """You are Zippy, a smart AI assistant made by Arul Vethathiri.
Tone:
- Talk like a smart, calm, helpful friend. Not overly excited. Not poetic. Not dramatic.
- Natural like a real conversation.
- Greetings: short and casual like "Hey! What's up?" or "Hi! What can I help with?"
Response Length — VERY IMPORTANT:
- Greetings / small talk / simple yes-no: 1-2 sentences MAX.
- Simple factual questions: 2-4 sentences.
- Explanations and how-things-work: use bullet points, be thorough, cover ALL important points fully.
- Code requests: give the COMPLETE working code, never cut it short, never truncate.
- Comparisons and lists: be comprehensive, use tables or bullets.
- Creative writing: complete the FULL piece, never cut short.
- Math: show every step clearly.
- If a topic genuinely needs a long detailed answer, write a long detailed answer. NEVER artificially shorten important information just to be brief.
Rules:
- NEVER say: Certainly!, Great question!, Of course!, Absolutely!, As an AI, traveler, delightful.
- Never repeat the question back.
- Always end with 1 relevant emoji."""

IDENTITY = {
    "who are you":       "I'm Zippy, a powerful AI chatbot made by Arul Vethathiri! 🤖",
    "what are you":      "I'm Zippy, an AI made by Arul Vethathiri. 🤖",
    "who made you":      "I was made by Arul Vethathiri. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "who built you":     "I was built by Arul Vethathiri. 👨‍💻",
    "what is your name": "My name is Zippy! 😊",
    "are you an ai":     "Yes! I'm Zippy, an AI made by Arul Vethathiri. 🤖",
    "are you human":     "Nope! I'm Zippy — an AI but great at conversation! 😄",
}

SOCIAL = {
    "thanks":     "You're welcome! 😊",
    "thank you":  "Happy to help anytime! 🌟",
    "bye":        "Goodbye! Take care! 👋",
    "goodbye":    "See you later! 👋",
    "good night": "Good night! Sleep well! 🌙",
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
    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        
        models_to_try = [
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip-image-captioning-base",
            "nlpconnect/vit-gpt2-image-captioning"
        ]
        
        description = None
        
        for model_name in models_to_try:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
                headers = {"Content-Type": "application/octet-stream"}
                
                response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        if "generated_text" in result[0]:
                            description = result[0]["generated_text"]
                    elif isinstance(result, dict) and "generated_text" in result:
                        description = result["generated_text"]
                    
                    if description:
                        break
                        
            except Exception as e:
                print(f"Model {model_name} failed: {str(e)}")
                continue
        
        if description:
            response_text = f"I can see: {description}\n\n**About your question:** \"{req.question}\"\n\nBased on the image: {description}"
            
            return {
                "description": description,
                "response": response_text,
                "success": True
            }
        else:
            error_msg = "The image analysis service is starting up (takes 20-30 seconds on first use). Please try again in a moment!"
            return {
                "description": error_msg,
                "response": error_msg,
                "success": False
            }
            
    except Exception as e:
        print(f"Image analysis error: {str(e)}")
        error_msg = f"I can see you've uploaded an image, but I'm having trouble analyzing it right now. Please try again! Error: {str(e)[:100]}"
        return {
            "description": error_msg,
            "response": error_msg,
            "success": False
        }

@app.get("/")
def root():
    return {"status": "Zippy backend is running!", "version": "3.1"}

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
        return {"reply": "That's not something I can help with. Let's keep things positive — ask me anything else! 😊"}

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
            return {"reply": "Oops! Couldn't connect to Zippy, please wait a moment and try again! 🙏"}

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

            print(f"✅ Served by: {model} ({len(model_usage[model])}/min)")
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

    return {"reply": "Oops! Couldn't connect to Zippy, please wait a moment and try again! 🙏"}

# ===================================
# SIMPLIFIED IMAGE GENERATION - NO URL TESTING
# ===================================
class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
async def generate_image(body: ImageRequest):
    """
    Generates image URL using Pollinations AI.
    Returns URL directly without testing - browser will load it.
    """
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="`prompt` is required")

    # Generate unique URL with random seed
    seed = random.randint(0, 999999)
    encoded = quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&seed={seed}"
    
    # Return URL directly - no testing needed!
    return {
        "imageUrl": image_url,
        "service": "pollinations",
        "success": True
    }
