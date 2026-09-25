import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_PIECE_SECONDS = 25


def _send_to_sarvam(piece_path: str) -> str:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError(
            "SARVAM_API_KEY is not configured. Add it to Render Environment Variables."
        )

    headers = {"api-subscription-key": api_key}

    with open(piece_path, "rb") as audio_file:
        files = {
            "file": (
                Path(piece_path).name,
                audio_file,
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
            timeout=180,
        )

    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(
            f"Sarvam STT request failed ({response.status_code}): {detail}"
        )

    result = response.json()
    return result.get("transcript", "")


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    texts: list[str] = []

    for index, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]
        piece_path = f"{chunk_path}_piece_{index}.wav"
        piece.export(piece_path, format="wav")

        try:
            texts.append(_send_to_sarvam(piece_path))
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(t for t in texts if t).strip()


def transcribe_all(chunks: list[str], language: str = "english") -> str:
    transcripts = []
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        print(f"Transcribing chunk {index}/{total}")
        text = transcribe_chunk(chunk, language=language)
        if text:
            transcripts.append(text)

    transcript = " ".join(transcripts).strip()
    if not transcript:
        raise RuntimeError("Sarvam returned an empty transcript.")

    return transcript
