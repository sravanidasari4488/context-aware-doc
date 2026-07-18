# Context-Aware Document Q&A Bot
A RAG-powered chat app that answers questions from your uploaded PDF and TXT documents using TF-IDF retrieval and Google Gemini.

## About the project
This project turns static documents into an interactive Q&A experience. Upload one or more files, ask questions in natural language, and get answers grounded in the document content—with source citations and confidence scores.
It is designed for demos, study workflows, and lightweight document exploration without a heavy backend or vector database. Retrieval runs entirely in the browser; only answer generation calls Google Gemini.
## How it works
The app uses a simple Retrieval-Augmented Generation (RAG) pipeline:
1. **Document upload and parsing** — PDF files are parsed with PDF.js; TXT files are read directly in the browser.
2. **Chunking** — Text is split into **600-character** chunks with **120-character** overlap, preferring sentence boundaries (`.`, `!`, `?`).
3. **Retrieval** — Chunks are indexed with **TF-IDF** vectors. User queries are matched with **cosine similarity**, including fuzzy token expansion for typos and follow-up-aware query rewriting.
4. **Answer generation** — The **top 7** relevant chunks are sent to **Google Gemini**, which answers using only those excerpts plus recent conversation history.
## Features
- PDF and TXT upload with drag and drop
- TF-IDF-based similarity search
- Source references with page numbers
- Confidence scoring (High / Medium / Low)
- Out-of-scope question detection
- Multi-document support
- Follow-up question handling
- Conversation history
## Tech stack
| Layer | Technology |
| --- | --- |
| UI | React 18 |
| Build | Vite 5 |
| Styling | Tailwind CSS (CDN) |
| LLM | Google Gemini API |
| PDF parsing | PDF.js 3.11 |
## Getting started
### Prerequisites
- Node.js 18+ and npm
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
### Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/ContextAwareDoc.git
cd ContextAwareDoc
# Install dependencies
npm install
```
Create a `.env` file in the project root:
```env
VITE_GEMINI_API_KEY=your_api_key_here
```
Optional: pin a preferred model (defaults fall back to `gemini-2.5-flash-lite`):
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
ContextAwareDoc/
├── ContextAwareDocQABot.jsx   # Main app: upload, indexing, chat, Gemini
├── main.jsx                   # React entry point
├── index.html                 # HTML shell + Tailwind CDN
├── vite.config.js             # Vite configuration
├── package.json
├── .env                       # Local API key (not committed)
└── README.md
```
