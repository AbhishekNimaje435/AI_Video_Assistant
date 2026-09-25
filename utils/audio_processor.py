import os
import re
import tempfile
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "ai_video_assistant" / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^\w\-\. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value).strip("._")
    return (value[:80] or "audio")


def download_youtube_audio(url: str) -> str:
    """Download a YouTube video/audio stream and convert it to WAV using FFmpeg."""
    job_dir = Path(tempfile.mkdtemp(prefix="yt_", dir=str(DOWNLOAD_DIR)))
    output_template = str(job_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "geo_bypass": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = Path(ydl.prepare_filename(info)).with_suffix(".wav")
    except Exception as exc:
        raise RuntimeError(
            "YouTube download/conversion failed. Check the URL and try again. "
            f"Details: {exc}"
        ) from exc

    if not filename.exists():
        candidates = list(job_dir.glob("*.wav"))
        if not candidates:
            raise FileNotFoundError("FFmpeg did not create the expected WAV file.")
        filename = candidates[0]

    return str(filename)


def convert_to_wav(input_path: str) -> str:
    """Convert a local audio/video file to 16 kHz mono WAV."""
    input_path = Path(input_path)
    output_path = input_path.with_name(f"{input_path.stem}_converted.wav")

    try:
        audio = AudioSegment.from_file(str(input_path))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(str(output_path), format="wav")
    except Exception as exc:
        raise RuntimeError(
            "Audio/video conversion failed. Make sure the uploaded file is supported."
        ) from exc

    return str(output_path)


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """Split a WAV file into chunks suitable for downstream transcription."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks: list[str] = []

    chunk_dir = Path(tempfile.mkdtemp(prefix="chunks_"))
    for index, start in enumerate(range(0, len(audio), chunk_ms)):
        piece = audio[start:start + chunk_ms]
        chunk_path = chunk_dir / f"chunk_{index}.wav"
        piece.export(str(chunk_path), format="wav")
        chunks.append(str(chunk_path))

    if not chunks:
        raise ValueError("The audio file is empty or could not be chunked.")

    return chunks


def process_input(source: str) -> list[str]:
    """Process a YouTube URL or a local file path."""
    source = source.strip()
    if not source:
        raise ValueError("No input source was provided.")

    if source.startswith(("http://", "https://")):
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")
        wav_path = convert_to_wav(source)

    return chunk_audio(wav_path)


def process_uploaded_file(uploaded_file) -> list[str]:
    """Save a Streamlit UploadedFile to a temporary path and process it."""
    if uploaded_file is None:
        raise ValueError("No uploaded file was provided.")

    suffix = Path(uploaded_file.name).suffix.lower() or ".bin"
    upload_dir = Path(tempfile.mkdtemp(prefix="upload_"))
    input_path = upload_dir / f"input{suffix}"
    input_path.write_bytes(uploaded_file.getvalue())

    wav_path = convert_to_wav(str(input_path))
    return chunk_audio(wav_path)
