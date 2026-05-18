import json
import os
import time
import urllib.request
from typing import Any

from openai import AzureOpenAI


SPEECH_API_VERSION = "v3.2"
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
GPT4O_DEPLOYMENT_NAME = os.environ.get("AZURE_GPT4O_DEPLOYMENT", "gpt-4o")

CONTEXTUAL_RAG_SYSTEM_PROMPT = (
    "You provide short context that helps retrieval systems understand a chunk."
)
CONTEXTUAL_RAG_USER_PROMPT = """<document>
{document}
</document>
<chunk>
{chunk}
</chunk>
Return a short context for retrieval. Return only the context."""


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    final_headers = headers or {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        final_headers = {"Content-Type": "application/json", **final_headers}
    req = urllib.request.Request(url, method=method, data=data, headers=final_headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def submit_audio_transcription(audio_url: str) -> str:
    if not AZURE_SPEECH_REGION or not AZURE_SPEECH_KEY:
        raise ValueError("Set AZURE_SPEECH_REGION and AZURE_SPEECH_KEY.")
    endpoint = (
        f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/"
        f"speechtotext/{SPEECH_API_VERSION}/transcriptions"
    )
    payload = {
        "displayName": "onyx-rag-audio-job",
        "locale": "en-US",
        "contentUrls": [audio_url],
        "properties": {
            "diarizationEnabled": True,
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "languageIdentification": {
                "candidateLocales": ["en-US", "hi-IN"],
            },
        },
    }
    response = _request_json(
        "POST",
        endpoint,
        headers={"Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY},
        payload=payload,
    )
    return str(response["self"])


def wait_for_transcription(transcription_url: str, timeout_seconds: int = 3600) -> str:
    start_time = time.time()
    headers = {"Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY}
    while True:
        status_response = _request_json("GET", transcription_url, headers=headers)
        status = str(status_response.get("status", ""))
        if status == "Succeeded":
            return transcription_url
        if status in {"Failed", "Cancelled"}:
            raise RuntimeError(f"Transcription failed: {status_response}")
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError("Timed out waiting for Azure Speech transcription.")
        time.sleep(10)


def fetch_diarized_transcript(transcription_url: str) -> str:
    headers = {"Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY}
    files_url = f"{transcription_url}/files"
    files_response = _request_json("GET", files_url, headers=headers)
    values = files_response.get("values", [])
    for item in values:
        if item.get("kind") != "Transcription":
            continue
        content_url = item.get("links", {}).get("contentUrl")
        if not content_url:
            continue
        transcript_payload = _request_json("GET", str(content_url))
        phrases = transcript_payload.get("recognizedPhrases", [])
        lines: list[str] = []
        for phrase in phrases:
            speaker = phrase.get("speaker", "UNKNOWN")
            offset = phrase.get("offsetInTicks", 0)
            start_sec = int(int(offset) / 10_000_000)
            minute = start_sec // 60
            second = start_sec % 60
            best = phrase.get("nBest", [{}])[0]
            text = str(best.get("display", "")).strip()
            if text:
                lines.append(f"[{minute:02d}:{second:02d}] SPEAKER_{speaker}: {text}")
        return "\n".join(lines)
    raise RuntimeError("No transcription file found in Azure Speech response.")


def chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_words]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
    return chunks


def contextualize_chunks(full_document: str, chunks: list[str]) -> list[dict[str, Any]]:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    output: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        prompt = CONTEXTUAL_RAG_USER_PROMPT.format(document=full_document, chunk=chunk)
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": CONTEXTUAL_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        context = response.choices[0].message.content or ""
        output.append(
            {
                "chunk_id": idx,
                "context": context.strip(),
                "original_chunk": chunk,
                "embeddable_text": f"{context.strip()}\n\n{chunk}",
            }
        )
    return output


def run_audio_rag_pipeline(audio_blob_url: str) -> list[dict[str, Any]]:
    transcription_url = submit_audio_transcription(audio_blob_url)
    completed_url = wait_for_transcription(transcription_url)
    transcript = fetch_diarized_transcript(completed_url)
    chunks = chunk_text(transcript)
    return contextualize_chunks(transcript, chunks)


if __name__ == "__main__":
    media_url = os.environ.get("AUDIO_BLOB_URL", "")
    if not media_url:
        raise ValueError("Set AUDIO_BLOB_URL to an Azure Blob SAS URL for your audio file.")
    result = run_audio_rag_pipeline(media_url)
    print(f"Generated {len(result)} contextualized chunks.")
