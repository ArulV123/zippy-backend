from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from urllib.parse import quote
import requests
import time

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

def wait_for_image(url, max_retries=10, timeout=30):
    """
    Wait for the image to be generated and ready.
    Returns True if image is ready, False otherwise.
    """
    start_time = time.time()
    
    for attempt in range(max_retries):
        # Check if we've exceeded timeout
        if time.time() - start_time > timeout:
            print(f"Timeout waiting for image after {timeout} seconds")
            return False
        
        try:
            # Make a HEAD request to check if image exists without downloading it
            response = requests.head(url, timeout=5, allow_redirects=True)
            
            if response.status_code == 200:
                print(f"Image ready after {attempt + 1} attempts ({time.time() - start_time:.2f}s)")
                return True
            
            # If not ready, wait with exponential backoff
            wait_time = min(2 ** attempt * 0.5, 5)  # Max 5 seconds between retries
            print(f"Attempt {attempt + 1}: Image not ready (status {response.status_code}), waiting {wait_time}s...")
            time.sleep(wait_time)
            
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1}: Request failed - {e}")
            time.sleep(2)
    
    print(f"Image failed to load after {max_retries} attempts")
    return False

def generate_image(prompt):
    """
    Generate image URL and wait for it to be ready.
    Returns tuple: (url, success)
    """
    # Encode prompt for URL
    encoded_prompt = quote(prompt)
    
    # Try multiple image generation services/formats
    urls_to_try = [
        # Pollinations with Flux model (usually more reliable)
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux&width=512&height=512&enhance=true",
        # Pollinations default
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512",
        # Pollinations simple
        f"https://image.pollinations.ai/prompt/{encoded_prompt}",
    ]
    
    for i, url in enumerate(urls_to_try):
        print(f"\nTrying image generation method {i + 1}/{len(urls_to_try)}...")
        print(f"URL: {url}")
        
        # Wait for this URL to be ready
        if wait_for_image(url, max_retries=8, timeout=25):
            return url, True
    
    # If all methods failed, return the first URL anyway
    # (better than returning nothing)
    print("All methods failed, returning first URL anyway")
    return urls_to_try[0], False

@app.post("/chat")
def chat(req: Req):
    msg = req.message.strip()
    
    try:
        # Get AI response
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Zippy AI, a helpful assistant. When users ask you to generate images, acknowledge warmly and briefly describe what you'll create."},
                {"role": "user", "content": msg}
            ],
            max_tokens=500
        )
        reply = res.choices[0].message.content
        
        # Check if user wants an image
        image_keywords = ["draw", "image", "generate", "show", "picture", "create", "sketch", "paint", "photo", "make"]
        wants_image = any(keyword in msg.lower() for keyword in image_keywords)
        
        if wants_image:
            print(f"\n{'='*60}")
            print(f"IMAGE GENERATION REQUESTED")
            print(f"Prompt: {msg}")
            print(f"{'='*60}")
            
            # Generate image and wait for it to be ready
            image_url, success = generate_image(msg)
            
            if success:
                print(f"✅ Image successfully generated and ready!")
            else:
                print(f"⚠️ Image may take additional time to load")
            
            return {
                "reply": reply, 
                "image": image_url,
                "image_ready": success
            }
        
        return {"reply": reply}
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"reply": f"Error: {str(e)}", "error": True}

@app.get("/")
def root():
    return {"status": "Zippy AI is running", "version": "3.0 - Image Pre-loading"}

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "groq_api_key_set": bool(os.environ.get("GROQ_API_KEY"))}

@app.get("/test-image/{prompt}")
def test_image(prompt: str):
    """Test endpoint to generate and verify an image"""
    url, success = generate_image(prompt)
    return {
        "prompt": prompt,
        "image_url": url,
        "ready": success,
        "message": "Image is ready!" if success else "Image may still be loading..."
    }
