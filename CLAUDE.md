# DocuTrust

Voice-enabled Corrective-RAG (CRAG) Streamlit app: ask questions about an uploaded PDF by text or voice and get grounded answers with source chunks. Live demo on Streamlit Cloud (see README).

## Stack
- Python, Streamlit, LangChain (`langchain-groq`, `langchain-community`)
- ChromaDB (in-memory, one collection per upload) + FastEmbed embeddings
- Groq: `openai/gpt-oss-120b` for answers, grading and query rewriting, `whisper-large-v3-turbo` for STT
- gTTS for TTS (Hindi output for Devanagari answers)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Secrets: `GROQ_API_KEY` in `.streamlit/secrets.toml` (git-ignored, never commit it or print it).

## Architecture notes
- Everything lives in `app.py`.
- All inputs (typed, spoken, suggested questions) go through the single `answer_question()` function. Keep it that way so text and voice can't drift apart.
- CRAG flow: retrieve → LLM relevance grader → if not relevant, rewrite the query and retrieve again → answer only from context, with citations.
- Each uploaded PDF gets its own Chroma collection, so chunks don't leak across uploads. A previous bug came from a shared persistent collection.
- Keep the model name in the single `LLM_MODEL` constant, because Groq retires models.
- Chunking uses the manual `split_text()` (chunk_size=1000, overlap=200) instead of `langchain_text_splitters`.

## Conventions
- Comment style: `# ── Section ──` headers plus short "Remember:" learning comments.
- Keep `requirements.txt` deployable on Streamlit Cloud (no heavy deps such as sentence-transformers).
- Update the README changelog for user-visible changes.
- Commits: no Claude co-author or "Generated with Claude Code" lines.
