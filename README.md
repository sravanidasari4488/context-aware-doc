# Context-Aware Document Q&A Bot
A RAG-powered chat app that answers questions from your uploaded PDF and TXT documents using TF-IDF retrieval and Google Gemini.

## About the project
This project turns static documents into an interactive Q&A experience. Upload one or more files, ask questions in natural language, and get answers grounded in the document content—with source citations and confidence scores.
It is designed for demos, study workflows, and lightweight document exploration without a heavy backend or vector database. Retrieval runs entirely in the browser; only answer generation calls Google Gemini.
**Screenshots:** [View demo screenshots](https://drive.google.com/drive/folders/1SgZnn1_zS-AooTXdQqQV76ijUHGLlysg?usp=drive_link)
## How it works
The app uses a simple Retrieval-Augmented Generation (RAG) pipeline:
1. **Document upload and parsing** — PDF files are parsed with PDF.js; TXT files are read directly in the browser.
2. **Chunking** — Text is split into **600-character** chunks with **120-character** overlap, preferring sentence boundaries (`.`, `!`, `?`).
3. **Retrieval** — Chunks are indexed with **TF-IDF** vectors. User queries are matched with **cosine similarity**, including fuzzy token expansion for typos and follow-up-aware query rewriting.
4. **Answer generation** — The **top 7** relevant chunks are sent to **Google Gemini** (`gemini-2.5-flash-lite` by default), which answers using only those excerpts plus recent conversation history.
## Features
- PDF and TXT upload with drag and drop
- TF-IDF-based similarity search
- Source references with page numbers
- Confidence scoring (High / Medium / Low)
- Out-of-scope question detection
- Multi-document support
- Follow-up question handling
- Conversation history
## Limitations
Retrieval uses **TF-IDF** (keyword / term-frequency matching), not semantic embeddings. Questions that paraphrase the document without sharing vocabulary may score poorly or be marked out of scope, even when a human would see them as related. This is an intentional tradeoff: the app stays fast, dependency-light, and fully client-side for indexing—no vector database or embedding API required. For production-grade semantic recall, dense embeddings (e.g. sentence transformers) would be the natural next step.
## Tech stack
| Layer | Technology |
| --- | --- |
| UI | React 18 |
| Build | Vite 5 |
| Styling | Tailwind CSS (CDN) |
| LLM | Google Gemini API (`gemini-2.5-flash-lite`) |
| PDF parsing | PDF.js 3.11 |
## Getting started
### Prerequisites
- Node.js 18+ and npm
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
### Setup
```bash
# Clone the repository
git clone https://github.com/sravanidasari4488/context-aware-doc.git
cd context-aware-doc
# Install dependencies
npm install
```
Create a `.env` file in the project root:
```env
VITE_GEMINI_API_KEY=your_api_key_here
```
Optional: pin a preferred model (defaults to `gemini-2.5-flash-lite`, with `gemini-2.5-flash` as fallback):
```env
VITE_GEMINI_MODEL=gemini-2.5-flash-lite
```
Start the development server:
```bash
npm run dev
```
Open the local URL shown in the terminal (typically `http://localhost:5173`).
### Production build
```bash
npm run build
npm run preview
```
## Project structure
```text
context-aware-doc/
├── ContextAwareDocQABot.jsx   # Main app: upload, indexing, chat, Gemini
├── main.jsx                   # React entry point
├── index.html                 # HTML shell + Tailwind CDN
├── vite.config.js             # Vite configuration
├── package.json
├── .env                       # Local API key (not committed)
└── README.md
```
## Notes
- Keep your Gemini API key in `.env` only; never commit it.
- Free-tier Gemini quotas apply (requests per minute and per day). The app spaces requests and falls back across models when needed.
- After changing PDF parsing or indexing logic, re-upload documents so the index is rebuilt.
