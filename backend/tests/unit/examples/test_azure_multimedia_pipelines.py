from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLES_DIR = REPO_ROOT / "examples" / "assistants-api"


def _load_module(file_name: str):
    path = EXAMPLES_DIR / file_name
    spec = spec_from_file_location(file_name.replace(".py", ""), path)
    assert spec
    assert spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audio_chunk_text_overlap() -> None:
    module = _load_module("multimedia_rag_pipeline.py")
    chunks = module.chunk_text("one two three four five six seven", chunk_words=4, overlap_words=1)
    assert chunks == [
        "one two three four",
        "four five six seven",
        "seven",
    ]


def test_video_chunk_text_overlap() -> None:
    module = _load_module("video_rag_pipeline.py")
    chunks = module.chunk_text("a b c d e f", chunk_words=3, overlap_words=1)
    assert chunks == ["a b c", "c d e", "e f"]


def test_build_multimodal_document_from_video_indexer_insights() -> None:
    module = _load_module("video_rag_pipeline.py")
    insights = {
        "videos": [
            {
                "insights": {
                    "transcript": [
                        {
                            "text": "Quarterly results improved.",
                            "speakerId": 2,
                            "instances": [{"startSeconds": 11.0}],
                        }
                    ],
                    "ocr": [
                        {
                            "text": "Q3 Revenue +25%",
                            "instances": [{"startSeconds": 12.0}],
                        }
                    ],
                }
            }
        ]
    }
    document = module.build_multimodal_document(insights)
    assert "[00:11] Audio SPEAKER_2: Quarterly results improved." in document
    assert "[00:12] Visual OCR: Q3 Revenue +25%" in document
