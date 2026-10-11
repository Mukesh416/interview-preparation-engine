# Interview Preparation Engine

Spoken mock interviews tailored to a JD and your resume, with per-answer feedback at the end.

## Setup (once)

1. Copy `.env.example` to `.env` and paste your Anthropic API key (https://console.anthropic.com/settings/keys).
2. `python -m pip install --user -r requirements.txt`

## Run

```
python server.py
```

Open http://localhost:8000 in **Chrome or Edge** (voice input uses the browser's speech recognition) and allow the microphone.

## How it works

- `server.py` (FastAPI) keeps the interview conversation and calls Claude (`claude-opus-5`):
  - interviewer turns run at low effort so questions come back quickly;
  - the final review runs at high effort and returns structured JSON (scores, strengths, improvements, a stronger answer per question).
- `static/index.html` speaks each question (browser text-to-speech) and transcribes your spoken answer live; you can edit the transcript before submitting.
- Resumes: PDF is sent to Claude directly; .txt or pasted text also works.
- Sessions live in memory, so restarting the server ends any interview in progress.
