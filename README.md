# ⚖️ LegalEase — Indian Law RAG Assistant

> **Three-stage retrieval pipeline:** HyDE → ChromaDB semantic search → Cohere cross-encoder reranker → GPT-4o-mini cited answer

A production-grade RAG system that lets you chat with Indian legal documents (IPC, RTI, Consumer Protection Act, IT Act, etc.) and get answers with exact section citations.

---

## Architecture

```
User question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: HyDE (Hypothetical Document Embedding)        │
│  → GPT generates a hypothetical expert answer           │
│  → That answer is embedded (not the raw question)       │
│  → Dense vectors match "answer-like" content better     │
└─────────────────────┬───────────────────────────────────┘
                      │  embedding vector
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: ChromaDB Semantic Retrieval                   │
│  → HNSW cosine similarity index                         │
│  → Retrieves top_k=15 candidate chunks                  │
│  → Each chunk carries: section, doc_title, page, text   │
└─────────────────────┬───────────────────────────────────┘
                      │  15 candidate chunks
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Cohere Cross-Encoder Reranker                 │
│  → rerank-english-v3.0 scores all 15 chunks             │
│  → Keeps top 5 by relevance score                       │
│  → Re-ranks against the ORIGINAL question (not HyDE)    │
└─────────────────────┬───────────────────────────────────┘
                      │  5 reranked chunks + scores
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: GPT-4o-mini Answer Generation                 │
│  → System prompt enforces citation format [1], [2]…     │
│  → Ends with "Key Sections" bullet list                 │
│  → Never speculates beyond provided context             │
└─────────────────────────────────────────────────────────┘
```

**Why HyDE?**
Standard RAG embeds the question and finds similar chunks. But questions and legal text live in different semantic spaces. HyDE generates what an *answer* would look like, then embeds that — pulling chunks that semantically match expert answers, not student questions. This typically improves retrieval hit-rate by 10-30% on domain-specific corpora.

**Why two-stage retrieval (embed → rerank)?**
Bi-encoders (used for initial retrieval) are fast but imprecise. Cross-encoders (Cohere reranker) score query+chunk together — far more accurate but too slow to run on all chunks. Two stages = speed + precision.

---

## Project Structure

```
legalease/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI route handlers
│   │   ├── core/config.py         # Pydantic settings
│   │   ├── models/schemas.py      # Request/response models
│   │   └── services/rag_engine.py # Core RAG pipeline
│   ├── scripts/
│   │   └── seed_data.py           # Download + ingest seed PDFs
│   ├── data/                      # Place your PDF files here
│   ├── chroma_db/                 # ChromaDB persists here (auto-created)
│   ├── .env.example
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx                # Main layout + chat
    │   ├── components/
    │   │   ├── SourceCard.jsx     # Reranked chunk card with score bar
    │   │   ├── StatsBar.jsx       # Pipeline stats (latency, chunks)
    │   │   ├── HydePanel.jsx      # Collapsible HyDE doc viewer
    │   │   ├── UploadPanel.jsx    # PDF upload + ingest UI
    │   │   └── DocumentsPanel.jsx # Indexed docs list with delete
    │   └── utils/api.js           # Backend API calls
    └── index.html
```

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- OpenAI API key ([platform.openai.com](https://platform.openai.com))
- Cohere API key ([cohere.com](https://cohere.com)) — free tier works

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and COHERE_API_KEY

# Start the API server
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 2. Seed data (optional but recommended)

```bash
# Download and ingest 4 free Indian law PDFs automatically
python scripts/seed_data.py

# OR: place your own PDFs in backend/data/ and run:
python scripts/seed_data.py --local
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check + chunk count |
| `/api/v1/query` | POST | Ask a legal question |
| `/api/v1/ingest` | POST | Upload + index a PDF |
| `/api/v1/documents` | GET | List all indexed documents |
| `/api/v1/documents/{title}` | DELETE | Remove document from index |

### Query request
```json
{
  "question": "What are the penalties for cybercrime under IT Act?",
  "top_k": 15,
  "hyde_enabled": true
}
```

### Query response
```json
{
  "answer": "Under Section 66 of the IT Act 2000, [1] ...",
  "sources": [
    {
      "text": "Section 66. Computer related offences...",
      "score": 0.94,
      "section": "Section 66",
      "doc_title": "Information Technology Act 2000",
      "page": 24
    }
  ],
  "hypothetical_doc": "The IT Act 2000 under Section 66 prescribes...",
  "retrieval_stats": {
    "chunks_retrieved": 15,
    "chunks_after_rerank": 5,
    "hyde_used": true,
    "latency_ms": 1840
  }
}
```

---

## Deployment

### Backend (Render free tier)
1. Push to GitHub
2. New Web Service on render.com → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add env vars in Render dashboard

### Frontend (Vercel)
```bash
cd frontend
npm run build
# Deploy dist/ to Vercel or run: npx vercel
```
Update `VITE_API_URL` or the Vite proxy to point to your Render backend URL.

---

## Interview talking points

When asked about this project in interviews, explain:

1. **Why HyDE?** — Questions and legal documents live in different semantic spaces. HyDE bridges this by embedding what a good answer *would look like*, improving retrieval precision significantly.

2. **Why two-stage retrieval?** — Bi-encoders are O(1) per chunk at query time (pre-computed embeddings) so fast for large corpora. Cross-encoders see (query, chunk) together so more accurate but O(n) — too slow to run on all chunks. Two-stage gets the best of both.

3. **Chunk size choice (512 tokens, 64 overlap)?** — Legal text has dense information. 512 keeps each chunk semantically coherent around one section/clause. 64-token overlap prevents answers from being split across chunk boundaries.

4. **Section-aware metadata?** — Regex extracts "Section 302", "Article 21" etc. at ingest time. Stored as metadata, not in the chunk text, so retrieval isn't distorted but citations are always available.

5. **Eval metrics to add** — Hit rate (does ground truth chunk appear in top-k?), MRR (mean reciprocal rank), and RAGAS faithfulness score. Adding these to README shows production thinking.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-small (OpenAI) |
| RAG framework | LlamaIndex |
| Vector DB | ChromaDB (persistent HNSW) |
| Reranker | Cohere rerank-english-v3.0 |
| Retrieval technique | HyDE + two-stage |
| Backend | FastAPI + Python |
| Frontend | React + Vite |
| PDF parsing | pdfplumber |

---

*LegalEase answers from indexed documents only. Always consult a qualified lawyer for actual legal advice.*
