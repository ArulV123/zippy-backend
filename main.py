import zipfile, os

base = "/mnt/data/zippy_working"
os.makedirs(base, exist_ok=True)

backend = """from fastapi import FastAPI
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

client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))

class Req(BaseModel):
    message: str

def generate_image(prompt):
    return "https://image.pollinations.ai/prompt/" + quote(prompt)

@app.post("/chat")
def chat(req: Req):
    msg = req.message.strip()

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"You are Zippy AI."},
            {"role":"user","content":msg}
        ]
    )

    reply = res.choices[0].message.content

    if any(k in msg.lower() for k in ["draw","image","generate","show","picture"]):
        return {"reply": reply, "image": generate_image(msg)}

    return {"reply": reply}
"""

frontend = """<!DOCTYPE html>
<html>
<head>
<title>Zippy</title>
</head>
<body>

<h2>Zippy AI</h2>

<input id="inp">
<button onclick="send()">Send</button>

<p id="out"></p>
<img id="img" width="400"/>

<script>
async function send(){
 let text=document.getElementById("inp").value;
 let res=await fetch("http://127.0.0.1:8000/chat",{
  method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify({message:text})
 });

 let data=await res.json();

 document.getElementById("out").innerText=data.reply;

 if(data.image){
  let img=document.getElementById("img");
  img.src=data.image;
 }
}
</script>

</body>
</html>
"""

with open(base+"/main.py","w") as f:
    f.write(backend)

with open(base+"/index.html","w") as f:
    f.write(frontend)

zip_path="/mnt/data/zippy_working.zip"
with zipfile.ZipFile(zip_path,'w') as z:
    z.write(base+"/main.py","main.py")
    z.write(base+"/index.html","index.html")

zip_path
