from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import re
import time
from collections import defaultdict

app = FastAPI()

origins = ["https://arulv123.github.io"]

client = Groq(api_key=os.environ["GROQ_API_KEY"])

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

# Track requests per model per minute
model_usage = defaultdict(list)  # model -> list of timestamps
model_blocked_until = {}         # model -> timestamp when it can be used again
MAX_RPM = 18  # stay under 20/min limit with buffer

def get_best_model():
    now = time.time()
    for model in MODELS:
        # Skip if blocked due to 429
        if model in model_blocked_until:
            if now < model_blocked_until[model]:
                continue
            else:
                del model_blocked_until[model]

        # Clean old timestamps outside 1 min window
        model_usage[model] = [t for t in model_usage[model] if now - t < 60]

        # If under limit, use this model
        if len(model_usage[model]) < MAX_RPM:
            return model

    return None  # all models busy

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

@app.get("/")
def root():
    return {"status": "Zippy backend is running!"}

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

    # Clean history
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

    # Ensure alternating roles
    filtered = []
    last_role = "assistant"
    for msg in clean_history:
        if msg["role"] != last_role:
            filtered.append(msg)
            last_role = msg["role"]

    messages = [{"role": "system", "content": SYSTEM}]
    messages += filtered
    messages.append({"role": "user", "content": user_input})

    # Try up to 6 times across models
    for attempt in range(len(MODELS)):
        model = get_best_model()

        if model is None:
            # All models busy right now, wait 5 seconds and retry once
            time.sleep(5)
            model = get_best_model()

        if model is None:
            return {"reply": "Oops! Couldn't connect to Zippy, please wait a moment and try again! 🙏"}

        try:
            # Log this request
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
                # Block this model for 60 seconds
                model_blocked_until[model] = time.time() + 60
                model_usage[model] = []
                continue
            elif "not found" in err.lower() or "invalid" in err.lower():
                model_blocked_until[model] = time.time() + 3600  # block bad model for 1hr
                continue
            else:
                break

    return {"reply": "Oops! Couldn't connect to Zippy, please wait a moment and try again! 🙏"}
