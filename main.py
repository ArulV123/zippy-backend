from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from urllib.parse import quote

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

class Req(BaseModel):
    message: str

def generate_image(prompt):
    # Pollinations AI endpoint - encode the prompt properly
    encoded_prompt = quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"

@app.post("/chat")
def chat(req: Req):
    msg = req.message.strip()
    
    try:
        # Get AI response
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Zippy AI, a helpful assistant. When users ask you to generate images, describe what you'll create in detail."},
                {"role": "user", "content": msg}
            ],
            max_tokens=500
        )
        reply = res.choices[0].message.content
        
        # Check if user wants an image
        image_keywords = ["draw", "image", "generate", "show", "picture", "create", "sketch", "paint"]
        wants_image = any(keyword in msg.lower() for keyword in image_keywords)
        
        if wants_image:
            image_url = generate_image(msg)
            return {"reply": reply, "image": image_url}
        
        return {"reply": reply}
    
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "error": True}

@app.get("/")
def root():
    return {"status": "Zippy AI is running"}
