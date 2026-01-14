from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
import os

# ------------------------
# FastAPI App Initialization
# ------------------------
app = FastAPI(title="Atul Rao – AI Career Assistant")

# Serve frontend UI
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------
# SYSTEM PROMPT with honesty & recruiter-friendly instructions
# ------------------------
SYSTEM_PROMPT = """
You are an AI assistant representing Atul Rao to recruiters, engineers, and managers.

Rules:
- Answer ONLY using the information provided below.
- Be professional, friendly, and concise.
- If information is missing, respond politely: "I do not have that information currently."
- Do NOT assume, exaggerate, or add facts.
- Detect the type of user asking the question:
    - Recruiter: Focus on impact, skills, outcomes.
    - Engineer: Focus on technical details, tools, processes.
    - Manager: Focus on ownership, responsibility, reliability.
- Default to general professional tone if unsure.

Information about Atul Rao:
"""

# Load profile
with open("profile.md", "r", encoding="utf-8") as f:
    PROFILE_TEXT = f.read()

# ------------------------
# Chat Request Schema
# ------------------------
class ChatRequest(BaseModel):
    question: str

# ------------------------
# Intent / Mode Detection Function
# ------------------------
def detect_mode(question: str) -> str:
    q_lower = question.lower()
    if any(k in q_lower for k in ["hire", "role", "why should we hire", "job"]):
        return "Recruiter"
    elif any(k in q_lower for k in ["how", "tool", "spark", "airflow", "api", "flask"]):
        return "Engineer"
    elif any(k in q_lower for k in ["lead", "responsibility", "manage", "ownership"]):
        return "Manager"
    else:
        return "General"

# ------------------------
# Chat Endpoint
# ------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    print("👉 Received question:", request.question)

    # Detect user mode
    mode = detect_mode(request.question)
    print("Detected mode:", mode)

    try:
        # Compose system prompt with mode
        prompt = SYSTEM_PROMPT + PROFILE_TEXT + f"\n\nUser Type: {mode}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": request.question}
            ],
            timeout=30  # seconds
        )
        print("✅ OpenAI responded")

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print("❌ OpenAI error:", str(e))
        return {"error": str(e)}
