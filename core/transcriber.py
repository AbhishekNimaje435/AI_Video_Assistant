from dotenv import load_dotenv
load_dotenv()

import os
import requests
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

import os
import requests
from pydub import AudioSegment

# -----------------------------
# Configuration
# -----------------------------


SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

SARVAM_PIECE_SECONDS = 25


# -----------------------------
# Sarvam
# -----------------------------

def _send_to_sarvam(piece_path: str) -> str:

    api_key = os.getenv("SARVAM_API_KEY")

    print("DEBUG KEY =", api_key)

    if not api_key:
        raise ValueError("SARVAM_API_KEY not found in environment variables")

    headers = {
        "api-subscription-key": api_key
    }

    with open(piece_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav",
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    result = response.json()

    return result.get("transcript", "")


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    texts = []

    for i, start in enumerate(range(0, len(audio), piece_ms)):

        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_piece_{i}.wav"

        piece.export(piece_path, format="wav")

        try:
            print(f"Sarvam piece {i+1}")

            txt = _send_to_sarvam(piece_path)

            texts.append(txt)

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(texts)


# -----------------------------
# Full Transcript
# -----------------------------

def transcribe_all(
    chunks: list,
    language: str = "english",
) -> str:

    print("Using Sarvam")

    transcripts = []

    total = len(chunks)

    for i, chunk in enumerate(chunks):

        print(f"Chunk {i+1}/{total}")

        text = transcribe_chunk(
            chunk,
            language=language,
        )

        transcripts.append(text)

    print("Transcription complete.")

    return " ".join(transcripts).strip()