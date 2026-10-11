"""Interview Preparation Engine - backend.

Run:  python server.py   then open http://localhost:8000 in Chrome or Edge.
Needs ANTHROPIC_API_KEY in the environment or in a .env file next to this script.
"""

import base64
import json
import uuid
from pathlib import Path

import anthropic
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

client = anthropic.Anthropic()
app = FastAPI()

# In-memory sessions: fine for a single-user local tool.
SESSIONS: dict[str, dict] = {}


INTERVIEWER_SYSTEM = """You are a seasoned hiring manager running a realistic spoken mock interview.

Target role: {target_role}
Candidate's current role: {current_role}
Interview style: {style}

Job description:
<job_description>
{jd}
</job_description>

The candidate's resume is attached in the first message.

How to run the interview:
- Ask exactly one question per turn. Your reply is converted to speech, so write only what you would say out loud: no markdown, bullet points, headings, or stage directions. Keep it to one to three sentences.
- Ground questions in the JD's key requirements and in specific items from the resume (projects, transitions, gaps, claims worth probing). Consider the jump from the candidate's current role to the target role.
- Mix question types according to the interview style. When an answer is vague or skips something important, you may use your next question as a short follow-up that digs deeper.
- Do not give feedback, hints, or evaluations during the interview. A brief, natural acknowledgement ("Thanks, that's helpful.") before the next question is fine.
- On the first turn, greet the candidate in one sentence, then ask your first question.
- Latency-sensitive; begin your visible answer immediately."""


FEEDBACK_SYSTEM = """You are an expert interview coach. You will receive the job description, the candidate's resume, and the transcript of a spoken mock interview (answers were captured by speech-to-text, so ignore minor transcription errors and filler words).

Give honest, specific, actionable feedback. For each answer, judge relevance to the question, structure (e.g. STAR for behavioral questions), depth and specificity, evidence of impact, and alignment with what the JD asks for. Suggest a stronger answer the candidate could realistically give using details from their own resume. Score each answer 1-10 where 5 is an average candidate and 8+ would clearly advance to the next round. If an answer was blank or skipped, say so and score it 1."""


FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer"},
        "hire_signal": {"type": "string", "enum": ["Strong no", "Lean no", "Lean yes", "Strong yes"]},
        "summary": {"type": "string"},
        "top_strengths": {"type": "array", "items": {"type": "string"}},
        "top_improvements": {"type": "array", "items": {"type": "string"}},
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "score": {"type": "integer"},
                    "what_went_well": {"type": "array", "items": {"type": "string"}},
                    "what_to_improve": {"type": "array", "items": {"type": "string"}},
                    "stronger_answer": {"type": "string"},
                },
                "required": ["question", "score", "what_went_well", "what_to_improve", "stronger_answer"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overall_score", "hire_signal", "summary", "top_strengths", "top_improvements", "answers"],
    "additionalProperties": False,
}


def call_claude(**kwargs) -> str:
    """Call Claude with server-side refusal fallback; return the text of the reply."""
    try:
        response = client.beta.messages.create(
            model=MODEL,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            **kwargs,
        )
    except (anthropic.AuthenticationError, TypeError) as e:
        if isinstance(e, TypeError) and "authentication" not in str(e):
            raise
        raise HTTPException(401, "Anthropic API key missing or invalid. Put ANTHROPIC_API_KEY=... in the .env file and restart the server.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate limited by the Anthropic API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Anthropic API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the Anthropic API. Check your internet connection.")

    if response.stop_reason == "refusal":
        raise HTTPException(502, "The model declined this request. Try rephrasing the inputs.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise HTTPException(502, f"Empty response from model (stop_reason={response.stop_reason}).")
    return text


def resume_blocks(session: dict) -> list[dict]:
    if session["resume_pdf_b64"]:
        return [{
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": session["resume_pdf_b64"]},
            "title": "Candidate resume",
        }]
    return [{"type": "text", "text": f"<resume>\n{session['resume_text']}\n</resume>"}]


def next_question(session: dict) -> str:
    asked = sum(1 for m in session["history"] if m["role"] == "assistant")
    total = session["num_questions"]
    note = f"You are now asking question {asked + 1} of {total}."
    if asked + 1 == total:
        note += " This is the final question; make it count."
    question = call_claude(
        max_tokens=4000,
        output_config={"effort": "low"},
        system=INTERVIEWER_SYSTEM.format(**session["profile"]),
        messages=session["history"] + [{"role": "system", "content": note}],
    )
    session["history"].append({"role": "assistant", "content": question})
    return question


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/start")
async def start(
    target_role: str = Form(...),
    current_role: str = Form(...),
    jd: str = Form(...),
    style: str = Form("Mixed behavioral and technical"),
    num_questions: int = Form(6),
    resume_text: str = Form(""),
    resume_file: UploadFile | None = File(None),
):
    resume_pdf_b64 = ""
    if resume_file and resume_file.filename:
        data = await resume_file.read()
        name = resume_file.filename.lower()
        if name.endswith(".pdf"):
            resume_pdf_b64 = base64.standard_b64encode(data).decode()
        elif name.endswith((".txt", ".md")):
            resume_text = data.decode("utf-8", errors="replace")
        else:
            raise HTTPException(400, "Upload the resume as PDF or .txt, or paste it as text.")
    if not resume_pdf_b64 and not resume_text.strip():
        raise HTTPException(400, "Please provide your resume (PDF upload or pasted text).")

    session = {
        "profile": {"target_role": target_role, "current_role": current_role, "jd": jd, "style": style},
        "num_questions": max(1, min(num_questions, 15)),
        "resume_pdf_b64": resume_pdf_b64,
        "resume_text": resume_text,
        "history": [],
    }
    session["history"].append({
        "role": "user",
        "content": resume_blocks(session) + [{"type": "text", "text": "(The candidate has joined the call and is ready.)"}],
    })
    question = next_question(session)
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = session
    return {"session_id": session_id, "question": question, "number": 1, "total": session["num_questions"]}


class AnswerIn(BaseModel):
    session_id: str
    answer: str


@app.post("/api/answer")
def answer(body: AnswerIn):
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Start a new interview.")
    if session["history"][-1]["role"] != "assistant":
        raise HTTPException(409, "This question was already answered.")
    session["history"].append({"role": "user", "content": body.answer.strip() or "(no answer given)"})

    asked = sum(1 for m in session["history"] if m["role"] == "assistant")
    if asked >= session["num_questions"]:
        return {"done": True}
    try:
        question = next_question(session)
    except HTTPException:
        session["history"].pop()  # let the client retry this answer
        raise
    return {"done": False, "question": question, "number": asked + 1, "total": session["num_questions"]}


class FeedbackIn(BaseModel):
    session_id: str


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Start a new interview.")

    history = session["history"]
    # Pair each question (assistant turn) with the answer that followed it.
    qa = []
    for i, m in enumerate(history):
        if m["role"] == "assistant" and i + 1 < len(history):
            qa.append((m["content"], history[i + 1]["content"]))
    if not qa:
        raise HTTPException(400, "No answers recorded yet.")
    transcript = "\n\n".join(
        f"Q{n}: {q}\nCandidate: {a}" for n, (q, a) in enumerate(qa, 1)
    )
    p = session["profile"]
    prompt = (
        f"Target role: {p['target_role']}\nCandidate's current role: {p['current_role']}\n\n"
        f"<job_description>\n{p['jd']}\n</job_description>\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"Give feedback on all {len(qa)} answers, in order."
    )
    text = call_claude(
        max_tokens=16000,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": FEEDBACK_SCHEMA}},
        system=FEEDBACK_SYSTEM,
        messages=[{"role": "user", "content": resume_blocks(session) + [{"type": "text", "text": prompt}]}],
    )
    return json.loads(text)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
