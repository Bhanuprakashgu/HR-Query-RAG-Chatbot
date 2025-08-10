import os
import requests
from mcp.server.fastmcp import FastMCP

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

mcp = FastMCP("hr-resource-chatbot-offline")

@mcp.tool()
def employees_search(q: str, k: int = 5) -> dict:
    """Search employees by natural language query. Returns JSON with results."""
    r = requests.get(f"{API_BASE}/employees/search", params={"q": q, "k": k}, timeout=60)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def chat(query: str, k: int = 5) -> dict:
    """Chat with the HR assistant. Returns natural-language answer and candidates."""
    r = requests.post(f"{API_BASE}/chat", json={"query": query, "k": k}, timeout=120)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def health() -> dict:
    """Check API and Ollama health."""
    r = requests.get(f"{API_BASE}/health", timeout=5)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    # Run with: python offline/mcp_server.py
    # Then connect via an MCP-compatible client (e.g., Claude Desktop) using a command config
    mcp.run()