import os
import io
import csv
import json
import time
import uuid
import math
import requests
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

# Minimal, single-file core with offline-first logic
# Embeddings & Generation via local Ollama
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")

DATA_PATH = os.environ.get("DATA_PATH", os.path.join(os.path.dirname(__file__), "employees.json"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

class Employee(BaseModel):
    id: int
    name: str
    title: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    domain_experience: Optional[List[str]] = None
    location: Optional[str] = None
    availability: str

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    EMPLOYEES: List[Employee] = [Employee(**e) for e in json.load(f)]

# In-memory index
EMB_DIM: Optional[int] = None
EMP_EMB: Optional[np.ndarray] = None  # shape: (N, D)

app = FastAPI(title="HR Resource Query Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def employee_text(e: Employee) -> str:
    parts = [
        e.name,
        e.title,
        f"experience {e.experience_years} years",
        ", ".join(e.skills),
        ", ".join(e.projects),
        ", ".join(e.domain_experience or []),
        e.location or "",
        e.availability,
    ]
    return " | ".join([p for p in parts if p])


def ollama_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    try:
        payload = {"model": EMBED_MODEL, "input": texts}
        r = requests.post(f"{OLLAMA_BASE}/api/embeddings", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Ollama returns { embeddings: [[...], [...], ...] }
        embs = np.array(data.get("embeddings", []), dtype=np.float32)
        return embs
    except Exception:
        return None


def build_index():
    global EMP_EMB, EMB_DIM
    texts = [employee_text(e) for e in EMPLOYEES]
    embs = ollama_embeddings(texts)
    if embs is not None and len(embs) == len(EMPLOYEES):
        EMP_EMB = embs
        EMB_DIM = embs.shape[1]
    else:
        EMP_EMB = None
        EMB_DIM = None


def keyword_score(text: str, query: str) -> float:
    t = text.lower()
    tokens = [tok for tok in ''.join([c if c.isalnum() or c in ['+', '#', '.'] else ' ' for c in query.lower()]).split() if tok]
    if not tokens:
        return 0.0
    hit = sum(1 for tok in tokens if tok in t)
    return hit / len(tokens)


def top_k(query: str, k: int = 5) -> List[Employee]:
    if EMP_EMB is not None and EMB_DIM is not None:
        q_embs = ollama_embeddings([query])
        if q_embs is not None and len(q_embs) == 1:
            q = q_embs[0]
            # cosine similarity
            a = EMP_EMB
            dot = (a @ q)
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(q) + 1e-8
            sim = dot / (na * nb + 1e-8)
            # bonus for exp + availability
            bonus = np.array([
                min(2, math.floor(e.experience_years/2)) * 0.05 + (0.1 if e.availability == 'available' else 0.0)
                for e in EMPLOYEES
            ])
            scores = sim + bonus
            idx = np.argsort(-scores)[:k]
            return [EMPLOYEES[int(i)] for i in idx]

    # Fallback: keyword
    scored = []
    for e in EMPLOYEES:
        s = keyword_score(employee_text(e), query)
        s += min(2, math.floor(e.experience_years/2)) * 0.05
        if e.availability == 'available':
            s += 0.1
        scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:k]]


def llama_reply(query: str, candidates: List[Employee]) -> str:
    system = (
        "You are a helpful, warm HR talent advisor. Answer succinctly but human, conversational, and encouraging.\n"
        "- Input: the user's staffing query and up to 5 candidate employee profiles (JSON).\n"
        "- Task: Recommend the best 2-3 matches with rationale: skills, years of experience, domain projects, availability. Mention names first, then reasoning.\n"
        "- Tone: supportive, clear, and professional. Avoid bullet overload; use short paragraphs and compact bullets only when helpful.\n"
        "- If the query is vague, suggest clarifying questions.\n"
        "- End with a friendly, action-oriented question."
    )
    context = f"User query: {query}\n\nCandidates JSON:\n" + json.dumps([c.model_dump() for c in candidates], indent=2)

    try:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
            "stream": False,
            "temperature": 0.3,
        }
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    except Exception:
        # Fallback: humanized template
        bullets = "\n".join([
            f"• {c.name} — {c.title}, {c.experience_years} yrs. Notable: {c.projects[0] if c.projects else '—'}. Skills: {', '.join(c.skills[:4])}. Availability: {c.availability}."
            for c in candidates[:3]
        ])
        return (
            "Here are strong matches I found based on your request:\n\n"
            f"{bullets}\n\n"
            "Would you like me to confirm their availability for a kickoff or share more on their past projects?"
        )


@app.on_event("startup")
async def _startup():
    build_index()


@app.get("/health")
def health():
    # also check ollama quickly
    ok = True
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    except Exception:
        ok = False
    return {"status": "ok", "ollama": ok}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html><head><meta charset='utf-8'><title>HR Chatbot API</title></head><body><h1>HR Resource Chatbot — Offline API</h1><p>UI: <a href='http://localhost:8501'>Streamlit UI</a></p><ul><li><a href='/health'>/health</a></li><li><a href='/employees/search?q=python&k=5'>/employees/search</a></li><li><a href='/docs'>/docs</a></li></ul></body></html>"""

@app.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)


@app.get("/employees/search")
def employees_search(q: str = "", k: int = 5):
    res = top_k(q, k)
    return {"query": q, "results": [e.model_dump() for e in res]}


class ChatBody(BaseModel):
    query: str
    k: int = 5

@app.post("/chat")
def chat(body: ChatBody):
    res = top_k(body.query, body.k)
    answer = llama_reply(body.query, res)
    return {"content": answer, "candidates": [e.model_dump() for e in res]}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Accept CSV or JSON of employees and store under uploads/. Merge into memory and rebuild index."""
    fname = f"{int(time.time())}_{uuid.uuid4().hex}_{file.filename}"
    dest = os.path.join(UPLOAD_DIR, fname)
    raw = file.file.read()
    with open(dest, "wb") as f:
        f.write(raw)

    # parse
    added = []
    try:
        if file.filename.lower().endswith(".json"):
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict) and "employees" in data:
                data = data["employees"]
            for item in data:
                added.append(Employee(**item))
        elif file.filename.lower().endswith(".csv"):
            text = raw.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                emp = Employee(
                    id=int(row.get("id") or 0),
                    name=row.get("name") or "",
                    title=row.get("title") or "",
                    skills=[s.strip() for s in (row.get("skills") or "").split(";") if s.strip()],
                    experience_years=int(row.get("experience_years") or 0),
                    projects=[p.strip() for p in (row.get("projects") or "").split(";") if p.strip()],
                    domain_experience=[d.strip() for d in (row.get("domain_experience") or "").split(";") if d.strip()],
                    location=row.get("location") or None,
                    availability=row.get("availability") or "available",
                )
                added.append(emp)
        else:
            return {"error": "Unsupported file type. Use .json or .csv"}
    except Exception as e:
        return {"error": f"Failed to parse: {e}"}

    # merge in-memory (id conflict: replace)
    by_id = {e.id: e for e in EMPLOYEES}
    for e in added:
        by_id[e.id] = e
    EMPLOYEES.clear()
    EMPLOYEES.extend(by_id.values())

    build_index()
    return {"stored": fname, "added": len(added), "total": len(EMPLOYEES)}