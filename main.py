from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from duckduckgo_search import DDGS
import os
import re
import json

app = FastAPI()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM = """You are Zippy, a smart AI assistant made by Arul Vethathiri.

Tone:
- Talk like a smart, calm, helpful friend. Not overly excited. Not poetic. Not dramatic.
- Short and natural like a real conversation.
- Greetings: just say hi back naturally. Short and casual.

Rules:
- Simple questions: 1-2 sentences. No fluff.
- Explanations: short bullet points, no long intros.
- Code: give the code directly, one line comment if needed.
- Creative tasks: complete the full piece, no preamble.
- Math: step by step, brief.
- NEVER say: Certainly!, Great question!, Of course!, Absolutely!, As an AI, traveler, delightful.
- Never repeat the question back.
- Always end with 1 relevant emoji."""

THINK_PROMPT = """You are a reasoning engine for an AI assistant called Zippy.

Given a user's message, output ONLY a valid JSON object (no markdown, no backticks) with these exact keys:
- "needs_search": boolean — true ONLY if the question needs real-time or very recent information
- "search_query": string — best web search query if needs_search is true, otherwise ""
- "reasoning": string — 1 sentence explaining your decision

Search IS needed for:
  current events, breaking news, live prices/stocks, sports scores, weather,
  recent product releases, "latest", "today", "this week", "right now" questions,
  specific recent facts that may have changed.

Search is NOT needed for:
  coding help, math, general knowledge, history, concepts, creative writing,
  opinions, greetings, identity questions, anything timeless.

Respond ONLY with raw JSON. Example:
{"needs_search": true, "search_query": "India vs Australia cricket score today", "reasoning": "User asked for a live sports score which requires real-time data."}"""

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

class ChatRequest(BaseModel):
    message: str
    history: list = []


def run_thinking(user_input: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": THINK_PROMPT},
                {"role": "user",   "content": user_input},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "needs_search": False,
            "search_query": "",
            "reasoning": f"Thinking step completed, no search needed.",
        }


def web_search(query: str, max_results: int = 4) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def format_search_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body  = r.get("body", "")[:300]
        href  = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
    return "\n\n".join(lines)


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
        return {
            "reply":    "Didn't catch that, say again? 😊",
            "thinking": "Empty input.",
            "searched": False,
            "sources":  [],
        }

    text = user_input.lower().strip()

    if text in IDENTITY:
        return {
            "reply":    IDENTITY[text],
            "thinking": "Identity question — hardcoded reply used.",
            "searched": False,
            "sources":  [],
        }
    if text in SOCIAL:
        return {
            "reply":    SOCIAL[text],
            "thinking": "Social phrase — hardcoded reply used.",
            "searched": False,
            "sources":  [],
        }

    # Thinking step
    think        = run_thinking(user_input)
    needs_search = think.get("needs_search", False)
    search_query = think.get("search_query", "")
    reasoning    = think.get("reasoning", "")

    # Web search
    search_context = ""
    searched       = False
    search_sources = []

    if needs_search and search_query:
        results = web_search(search_query)
        if results:
            search_context = format_search_results(results)
            searched = True
            search_sources = [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", "")[:200],
                }
                for r in results[:4]
            ]

    # Build messages
    messages = [{"role": "system", "content": SYSTEM}]
    messages += req.history[-20:]

    final_user_content = (
        f"{search_context}\n\nUsing the above search results, answer this:\n{user_input}"
        if search_context else user_input
    )
    messages.append({"role": "user", "content": final_user_content})

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        max_tokens=512,
        temperature=0.85,
        top_p=0.92,
    )

    reply = response.choices[0].message.content.strip()

    return {
        "reply":    reply,
        "thinking": reasoning,
        "searched": searched,
        "sources":  search_sources,
    }
