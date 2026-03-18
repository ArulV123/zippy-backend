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
        print("⚠️ Model warmup failed")
    
    yield
    print("🔴 Shutting down")

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

# IMPROVED SYSTEM PROMPT - No unnecessary name mentions, knows about image capabilities
SYSTEM = """You are Zippy, a smart AI assistant.

**Your Capabilities:**
- You can generate images when users ask (e.g., "generate image of...", "create image of...", "draw me...")
- You can analyze images that users upload
- You have web search to find current information when needed
- You're great at coding, math, explanations, and creative tasks

**Tone:**
- Talk like a smart, calm, helpful friend
- Natural and conversational
- Not overly excited, poetic, or dramatic
- Greetings: short and casual like "Hey! What's up?" or "Hi! What can I help with?"

**Response Length:**
- Greetings/small talk: 1-2 sentences MAX
- Simple questions: 2-4 sentences
- Explanations: thorough with bullet points covering ALL important points
- Code: COMPLETE working code, never truncated
- Comparisons/lists: comprehensive, use tables or bullets
- Creative writing: FULL piece, never cut short
- Math: show every step clearly

**Important Rules:**
- NEVER say: "Certainly!", "Great question!", "Of course!", "Absolutely!", "As an AI", "traveler", "delightful"
- Never repeat the question back
- Only mention your creator (Arul Vethathiri) if directly asked "who made you" or similar
- Don't unnecessarily mention your creator in normal conversation
- Always end with 1 relevant emoji

**When to Use Your Capabilities:**
- Image generation: When user asks to generate/create/draw an image
- Image analysis: When user uploads an image
- Web search: When you need current info, recent events, or verification of facts you're uncertain about
"""

IDENTITY = {
    "who are you":       "I'm Zippy, a smart AI assistant! 🤖",
    "what are you":      "I'm Zippy, an AI assistant. 🤖",
    "who made you":      "I was made by Arul Vethathiri. 👨‍💻",
    "who created you":   "I was created by Arul Vethathiri. 👨‍💻",
    "who built you":     "I was built by Arul Vethathiri. 👨‍💻",
    "what is your name": "My name is Zippy! 😊",
    "are you an ai":     "Yes! I'm Zippy, an AI assistant. 🤖",
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
    """Analyzes images using Hugging Face BLIP models with fallbacks"""
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
                        print(f"✅ Image analyzed with {model_name}")
                        break
                        
            except Exception as e:
                print(f"Model {model_name} failed: {str(e)}")
                continue
        
        if description:
            response_text = f"I can see: {description}\n\nRegarding your question \"{req.question}\" - based on the image, {description}"
            
            return {
                "description": description,
                "response": response_text,
                "success": True
            }
        else:
            error_msg = "The image analysis service is warming up (takes 20-30 seconds on first use). Please try again in a moment!"
            return {
                "description": error_msg,
                "response": error_msg,
                "success": False
            }
            
    except Exception as e:
        print(f"Image analysis error: {str(e)}")
        error_msg = "I'm having trouble analyzing this image right now. Please try again!"
        return {
            "description": error_msg,
            "response": error_msg,
            "success": False
        }

@app.get("/")
def root():
    return {"status": "Zippy backend is running!", "version": "4.0"}

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
            return {"reply": "Oops! Couldn't connect right now, please try again in a moment! 🙏"}

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

    return {"reply": "Oops! Couldn't connect right now, please try again in a moment! 🙏"}

# ===================================
# IMAGE GENERATION - FIXED TO ACTUALLY WORK
# ===================================
class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
async def generate_image(body: ImageRequest):
    """
    Generates images using Hugging Face Stable Diffusion.
    Returns actual image data as base64, not URLs that can fail.
    """
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="`prompt` is required")

    try:
        # Use Hugging Face Stable Diffusion
        HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        
        print(f"🎨 Generating image: {prompt[:50]}...")
        
        response = requests.post(
            HF_API_URL,
            headers={"Content-Type": "application/json"},
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200 and response.content:
            # Convert to base64
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_base64}"
            
            print(f"✅ Image generated successfully!")
            
            return {
                "imageUrl": image_data_url,
                "service": "huggingface",
                "success": True
            }
        elif response.status_code == 503:
            # Model is loading
            return {
                "imageUrl": "",
                "service": "huggingface",
                "success": False,
                "error": "The image generation model is warming up (takes 20-30 seconds on first use). Please try again in a moment!"
            }
        else:
            raise Exception(f"HuggingFace returned {response.status_code}")
            
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return {
            "imageUrl": "",
            "service": "none",
            "success": False,
            "error": "Image generation is temporarily unavailable. Please try again in a moment!"
        }
