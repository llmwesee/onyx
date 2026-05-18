import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from openai import AzureOpenAI


AZURE_VIDEO_INDEXER_LOCATION = os.environ.get("AZURE_VIDEO_INDEXER_LOCATION", "")
AZURE_VIDEO_INDEXER_ACCOUNT_ID = os.environ.get("AZURE_VIDEO_INDEXER_ACCOUNT_ID", "")
AZURE_VIDEO_INDEXER_API_KEY = os.environ.get("AZURE_VIDEO_INDEXER_API_KEY", "")

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


def get_video_indexer_access_token() -> str:
    if (
        not AZURE_VIDEO_INDEXER_LOCATION
        or not AZURE_VIDEO_INDEXER_ACCOUNT_ID
        or not AZURE_VIDEO_INDEXER_API_KEY
    ):
        raise ValueError(
            "Set AZURE_VIDEO_INDEXER_LOCATION, AZURE_VIDEO_INDEXER_ACCOUNT_ID, and AZURE_VIDEO_INDEXER_API_KEY."
        )
    url = (
        f"https://api.videoindexer.ai/Auth/{AZURE_VIDEO_INDEXER_LOCATION}/"
        f"Accounts/{AZURE_VIDEO_INDEXER_ACCOUNT_ID}/AccessToken?allowEdit=true"
    )
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Ocp-Apim-Subscription-Key": AZURE_VIDEO_INDEXER_API_KEY},
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8").strip('"')


def submit_video_index_job(video_url: str, access_token: str) -> str:
    encoded_name = urllib.parse.quote("onyx-rag-video-job")
    encoded_video_url = urllib.parse.quote(video_url, safe="")
    url = (
        f"https://api.videoindexer.ai/{AZURE_VIDEO_INDEXER_LOCATION}/"
        f"Accounts/{AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos"
        f"?accessToken={urllib.parse.quote(access_token, safe='')}"
        f"&name={encoded_name}"
        f"&videoUrl={encoded_video_url}"
        f"&sourceLanguage=AutoDetect"
    )
    response = _request_json("POST", url)
    return str(response["id"])


def wait_for_video_index(video_id: str, access_token: str, timeout_seconds: int = 7200) -> None:
    start_time = time.time()
    while True:
        state_url = (
            f"https://api.videoindexer.ai/{AZURE_VIDEO_INDEXER_LOCATION}/"
            f"Accounts/{AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos/{video_id}/Index"
            f"?accessToken={urllib.parse.quote(access_token, safe='')}"
        )
        status_response = _request_json("GET", state_url)
        state = (
            status_response.get("state")
            or status_response.get("videos", [{}])[0].get("state")
            or ""
        )
        if state == "Processed":
            return
        if state == "Failed":
            raise RuntimeError(f"Video Indexer failed for video {video_id}: {status_response}")
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError("Timed out waiting for Azure Video Indexer processing.")
        time.sleep(15)


def fetch_video_insights(video_id: str, access_token: str) -> dict[str, Any]:
    insights_url = (
        f"https://api.videoindexer.ai/{AZURE_VIDEO_INDEXER_LOCATION}/"
        f"Accounts/{AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos/{video_id}/Index"
        f"?accessToken={urllib.parse.quote(access_token, safe='')}"
    )
    return _request_json("GET", insights_url)


def build_multimodal_document(insights: dict[str, Any]) -> str:
    video = insights.get("videos", [{}])[0]
    transcript_items = video.get("insights", {}).get("transcript", [])
    ocr_items = video.get("insights", {}).get("ocr", [])

    timeline: list[tuple[float, str]] = []
    for entry in transcript_items:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        speaker = entry.get("speakerId", "UNKNOWN")
        start = float(entry.get("instances", [{}])[0].get("startSeconds", 0.0))
        timeline.append((start, f"Audio SPEAKER_{speaker}: {text}"))

    for entry in ocr_items:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        start = float(entry.get("instances", [{}])[0].get("startSeconds", 0.0))
        timeline.append((start, f"Visual OCR: {text}"))

    timeline.sort(key=lambda item: item[0])
    lines: list[str] = []
    for start, content in timeline:
        minute = int(start) // 60
        second = int(start) % 60
        lines.append(f"[{minute:02d}:{second:02d}] {content}")
    return "\n".join(lines)


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


def run_video_rag_pipeline(video_blob_url: str) -> list[dict[str, Any]]:
    token = get_video_indexer_access_token()
    video_id = submit_video_index_job(video_blob_url, token)
    wait_for_video_index(video_id, token)
    insights = fetch_video_insights(video_id, token)
    full_document = build_multimodal_document(insights)
    chunks = chunk_text(full_document)
    return contextualize_chunks(full_document, chunks)


if __name__ == "__main__":
    media_url = os.environ.get("VIDEO_BLOB_URL", "")
    if not media_url:
        raise ValueError("Set VIDEO_BLOB_URL to an Azure Blob SAS URL for your video file.")
    result = run_video_rag_pipeline(media_url)
    print(f"Generated {len(result)} contextualized chunks.")
