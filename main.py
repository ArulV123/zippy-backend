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

def generate_image_pollinations(prompt):
    """Generate image using Pollinations AI"""
    encoded_prompt = quote(prompt)
    # Simplified URL - Pollinations will use defaults
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}"

def generate_image_flux(prompt):
    """Alternative: Generate image using Flux via Pollinations"""
    encoded_prompt = quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux&width=512&height=512"

def generate_image(prompt):
    """Try Pollinations AI with Flux model for better results"""
    return generate_image_flux(prompt)

@app.post("/chat")
def chat(req: Req):
    msg = req.message.strip()
    
    try:
        # Get AI response
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Zippy AI, a helpful assistant. When users ask you to generate images, acknowledge their request warmly and describe what kind of image you'll create."},
                {"role": "user", "content": msg}
            ],
            max_tokens=500
        )
        reply = res.choices[0].message.content
        
        # Check if user wants an image
        image_keywords = ["draw", "image", "generate", "show", "picture", "create", "sketch", "paint", "photo"]
        wants_image = any(keyword in msg.lower() for keyword in image_keywords)
        
        if wants_image:
            image_url = generate_image(msg)
            return {"reply": reply, "image": image_url}
        
        return {"reply": reply}
    
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "error": True}

@app.get("/")
def root():
    return {"status": "Zippy AI is running", "version": "2.0"}

@app.get("/test-image/{prompt}")
def test_image(prompt: str):
    """Test endpoint to verify image URL generation"""
    return {
        "prompt": prompt,
        "image_url": generate_image(prompt),
        "message": "Try opening this URL in your browser"
    }
