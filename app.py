from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
import os


app = FastAPI(title="Atul Rao – AI Career Assistant")
# Add at the end, before uvicorn runs
app.mount("/", StaticFiles(directory=".", html=True), name="static")


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an AI assistant representing Atul Rao to recruiters and hiring managers.

Rules:
- Answer ONLY using the information provided below
- Be friendly, professional, and concise
- If information is missing, say you do not have that information
- Do NOT make assumptions or add facts

Information about Atul Rao:
"""

with open("profile.md", "r", encoding="utf-8") as f:
    PROFILE_TEXT = f.read()

print("PROFILE CONTENT LENGTH:", len(PROFILE_TEXT))
print("FIRST 300 CHARACTERS:")
print(PROFILE_TEXT[:300])

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    print("👉 Received question:", request.question)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + PROFILE_TEXT},
                {"role": "user", "content": request.question}
            ],
            timeout=30  # IMPORTANT
        )
        print("✅ OpenAI responded")

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print("❌ OpenAI error:", str(e))
        return {"error": str(e)}
    


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    






    
