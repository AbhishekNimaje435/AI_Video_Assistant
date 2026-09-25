# AI Video Assistant

A Streamlit-based AI Video Assistant that can process a YouTube video or an uploaded audio/video file, transcribe it with Sarvam AI, generate a title and summary with Mistral, extract action items/decisions/questions, and provide RAG-based chat over the transcript.

## Features

- YouTube audio extraction with `yt-dlp`
- Uploaded audio/video support in the web UI
- FFmpeg-based audio conversion
- Sarvam AI speech-to-text
- Mistral-powered title generation and summarisation
- Action-item, decision, and question extraction
- HuggingFace sentence-transformer embeddings
- Chroma vector database
- RAG-based chat with the transcript
- Streamlit UI
- Render deployment configuration

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Install FFmpeg on your local machine if it is not already available.

Create `.env` from `.env.example` and set:

```text
MISTRAL_API_KEY=...
SARVAM_API_KEY=...
SARVAM_STT_MODEL=saaras:v2.5
```

## Render deployment

This repository includes `render.yaml` and `apt.txt`.

### Option 1 — Blueprint

Create a new Render Blueprint from this repository. Render will use `render.yaml`.

### Option 2 — Manual Web Service

Build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Add these environment variables in Render:

- `MISTRAL_API_KEY`
- `SARVAM_API_KEY`
- `SARVAM_STT_MODEL` = `saaras:v2.5`

`apt.txt` installs the FFmpeg system binary required by `pydub` and `yt-dlp` audio extraction.

## Important

The deployed app cannot access a path such as `C:\Users\...` from your personal computer. Use the **YouTube URL** field or the **file uploader** in the deployed Streamlit app.

The Render filesystem is ephemeral. The Chroma database is therefore created in a temporary directory for the current processing session rather than treated as permanent storage.

## Project structure

```text
.
├── app.py
├── main.py
├── requirements.txt
├── render.yaml
├── apt.txt
├── runtime.txt
├── .python-version
├── .env.example
├── .streamlit/
│   └── config.toml
├── core/
│   ├── __init__.py
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   └── vector_store.py
└── utils/
    ├── __init__.py
    └── audio_processor.py
```
