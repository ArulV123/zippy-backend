from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import re

app = FastAPI()

origins = [
    "https://arulv123.github.io"
]

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
    "llama3-8b-8192",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
]

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
    "how to hack", "how to steal", "child porn", "csam",
]

def is_harmful(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in HARMFUL_KEYWORDS)

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

    messages = [{"role": "system", "content": SYSTEM}]
    messages += req.history[-20:]
    messages.append({"role": "user", "content": user_input})

    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=0.85,
                top_p=0.92,
            )
            reply = response.choices[0].message.content.strip()

            # If model refused with a safety message, give friendly response
            refusal_hints = ["cannot assist", "can't assist", "unable to assist",
                             "i'm not able", "i cannot", "not able to help",
                             "against my", "not appropriate", "harmful"]
            if any(hint in reply.lower() for hint in refusal_hints):
                return {"reply": "That's not something I can help with. Let's keep things positive — ask me anything else! 😊"}

            return {"reply": reply}

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err:
                # This model hit its limit, try the next one
                continue
            else:
                # Unknown error, stop trying
                break

    # All models exhausted or unknown error
    return {"reply": "Oops! Couldn't connect to Zippy, please wait a moment and try again! 🙏"}
