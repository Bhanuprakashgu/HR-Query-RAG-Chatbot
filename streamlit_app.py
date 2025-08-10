import json
import os
import sys
import time
import subprocess
import atexit
import requests
import streamlit as st
from typing import List

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="HR Resource Chatbot", page_icon="🤝", layout="wide")

st.title("HR Resource Query Chatbot")
st.caption("Local Llama 3 via Ollama • Drag-and-drop uploads • No cloud required")


def _wait_for(url: str, timeout: float = 60.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _ensure_services():
    if st.session_state.get("_services_started"):
        return

    st.session_state.setdefault("_procs", [])
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Start FastAPI if not healthy
    healthy = False
    try:
        healthy = requests.get(f"{API_BASE}/health", timeout=2).ok
    except Exception:
        healthy = False
    if not healthy:
        try:
            p_api = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
                cwd=script_dir,
            )
            st.session_state["_procs"].append(p_api)
            _wait_for(f"{API_BASE}/health", timeout=60)
        except Exception:
            pass

    # Start MCP server (best-effort)
    try:
        p_mcp = subprocess.Popen([sys.executable, "mcp_server.py"], cwd=script_dir)
        st.session_state["_procs"].append(p_mcp)
    except Exception:
        pass

    def _cleanup():
        for p in st.session_state.get("_procs", []):
            try:
                if p and p.poll() is None:
                    p.terminate()
            except Exception:
                pass

    atexit.register(_cleanup)
    st.session_state["_services_started"] = True


_ensure_services()

# Health
try:
    health = requests.get(f"{API_BASE}/health", timeout=3).json()
    st.sidebar.success("API: up")
    st.sidebar.write(f"Ollama: {'running' if health.get('ollama') else 'not detected'}")
except Exception:
    st.sidebar.error("API: down")

with st.sidebar:
    st.subheader("Upload Employees (JSON)")
    file = st.file_uploader("Upload employees.json", type=["json"])
    if file is not None:
        files = {"file": (file.name, file.getvalue())}
        try:
            resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
            if resp.ok:
                st.success(f"Uploaded: {resp.json().get('added', 0)} added")
            else:
                st.error(resp.text)
        except Exception as e:
            st.error(str(e))

SUGGESTIONS = [
    "Find Python developers with 3+ years experience",
    "Who has worked on healthcare projects?",
    "Suggest people for a React Native project",
    "Find developers who know both AWS and Docker",
]

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "content": "Hi! Tell me what you need — skills, years, domain, availability — and I’ll recommend the best people."}
    ]


def build_profiles_list(query: str) -> str:
    """Get matching profiles from backend and filter locally for relevance."""
    try:
        data = requests.get(f"{API_BASE}/employees/search", params={"q": query, "k": 5}, timeout=60).json()
    except Exception as e:
        return f"(fallback) Could not fetch profiles: {e}"

    results = data.get("results", [])
    if not results:
        return "_No matching profiles found._"

    # Local filter
    q_tokens = [t.strip().lower() for t in query.split() if t.strip()]
    filtered = []
    for e in results:
        haystack_parts = [
            e.get("name", ""),
            e.get("title", ""),
            " ".join(e.get("domain_experience", []) or []),
            str(e.get("experience_years", "")),
            " ".join(e.get("projects", []) or []),
            " ".join(e.get("skills", []) or []),
        ]
        haystack = " ".join(haystack_parts).lower()
        if any(tok in haystack for tok in q_tokens):
            filtered.append(e)

    if not filtered:
        return "_No matching profiles found._"

    # Build bullet list
    lines = ["**Suitable Profiles:**\n"]
    for e in filtered:
        name = e.get("name", "N/A")
        role = e.get("title", "N/A")
        dept = ", ".join(e.get("domain_experience", [])) if e.get("domain_experience") else "N/A"
        exp = e.get("experience_years", "N/A")
        notable = ", ".join(e.get("projects", [])) if e.get("projects") else "—"
        skills = ", ".join(e.get("skills", [])) if e.get("skills") else "—"
        avail = e.get("availability", "unknown")
        lines.append(f"- **{name}** — {role}, {exp} yrs. Dept: {dept}. Notable: {notable}. Skills: {skills}. Availability: {avail}.")
    return "\n".join(lines)


with st.container():
    for msg in st.session_state.history:
        align = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(align):
            st.write(msg["content"])

    prompt = st.chat_input("Ask for skills, experience, domain, or availability…")
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})

        try:
            res = requests.post(f"{API_BASE}/chat", json={"query": prompt, "k": 5}, timeout=120).json()
            answer = res.get("content", "")
        except Exception as e:
            answer = f"(fallback) I encountered an issue. Could you try again?\n\nError: {e}"

        profiles_text = build_profiles_list(prompt)
        if profiles_text.startswith("**Suitable Profiles:**"):
            st.session_state.history.append({"role": "assistant", "content": profiles_text})
        else:
            st.session_state.history.append({"role": "assistant", "content": answer})

        st.rerun()
