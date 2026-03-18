from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import re
from urllib.parse import quote

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GROQ
# =========================
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# =========================
# REQUEST MODEL
# =========================
class ChatRequest(BaseModel):
    message: str
    history: list = []

# =========================
# IMAGE GENERATION
# =========================
def generate_image(prompt: str):
    encoded = quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}"

    print("🎨 IMAGE GENERATED:", url)

    return url

# =========================
# IMAGE DECISION (FIXED)
# =========================
def should_generate_image(user_input: str):
    keywords = [
        "draw", "image", "picture", "show",
        "generate", "create", "design",
        "illustration", "sketch", "art"
    ]

    text = user_input.lower()

    return any(word in text for word in keywords)

# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: ChatRequest):
    user_input = re.sub(
        r'^(mm+|um+|uh+|hmm+|hm+)\s+',
        '',
        req.message
    ).strip()

    if not user_input:
        return {"reply": "Say something 😊"}

    # =========================
    # CALL GROQ
    # =========================
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are Zippy, a helpful AI."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=1024,
        temperature=0.7,
    )

    reply = response.choices[0].message.content.strip()

    print("💬 USER:", user_input)
    print("🤖 REPLY:", reply)

    # =========================
    # IMAGE LOGIC (FIXED)
    # =========================
    if should_generate_image(user_input):
        image_url = generate_image(user_input)

        return {
            "reply": reply,
            "image": image_url
        }

    return {"reply": reply}

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "Zippy running 🚀"}
