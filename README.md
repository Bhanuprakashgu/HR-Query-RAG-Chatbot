# HR Resource Query Chatbot

## Overview
The HR Resource Query Chatbot is a smart AI tool that helps HR teams quickly find employees based on skills, years of experience, and availability.  
It understands plain language questions and gives accurate, ranked results in seconds — all running locally for privacy.

## Features
- Upload employee data in JSON format.
- Search using natural language (e.g., “Having 4+ years of experience in Python”).
- Uses AI embeddings and cosine similarity for semantic matching.
- Runs fully offline with a local AI model.
- Simple web interface for quick access.

## Architecture
- **Frontend:** Streamlit app for file upload, query entry, and results display.
- **Backend Core:** Handles embeddings, similarity search, and ranking.
- **Middleware:** Connects frontend to backend logic.
- **Dataset:** Employee profiles stored in `employees.json`.
- **Local AI Model:** Powered by Ollama for embeddings.

**Flow:**  
1. Upload the JSON file.  
2. Query is converted to embeddings.  
3. Compared with employee embeddings using cosine similarity.  
4. Results are ranked and shown.

## Setup & Installation
1. Clone the repository and navigate into it:
```bash
git clone <repo_url>
cd HR-RAG-chatbot
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Use CMD // Install and start [Ollama](https://ollama.ai/), then pull models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```
5. Start the application:
```bash
python start.py
```

## API Documentation
- **POST /search** – Search employees by query.
- Request body:
```json
{
  "query": "Having 4+ years of experience in Python",
  "top_k": 5
}
```
- Response: List of top matching employees with details.

## AI Development Process
- **Tools used:** ChatGPT, GitHub Copilot (as supportive assistants).
- **AI usage:** Helped with structuring some functions, refining code efficiency, and quick debugging suggestions.
- **Manual work:** All major architecture, backend logic, frontend design, and integration were implemented manually.
- **AI contribution:** ~30% assisted, ~70% manually written.
- **Reason:** AI was used strategically to save time on repetitive coding tasks while ensuring the main logic and design remained fully under my control.

## Technical Decisions
- **Local AI (Ollama):** Chosen for privacy and zero API cost.
- **Embedding Model:** `nomic-embed-text` for fast and accurate semantic search.
- **Frontend:** Streamlit for rapid UI building.

## Future Improvements
- Add authentication.
- Support CSV/Excel uploads.
- More filters for refined search.
- UI enhancements with sorting and pagination.

## Demo
[Loop Video](https://drive.google.com/file/d/1oucfp-3fVRR0Uh_OjQ1Q1wmaO89yvrP6/view)

## Example query:  
> “Having 4+ years of experience in Python”  
Returns a ranked list of matching employees.
