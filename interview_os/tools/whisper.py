"""Whisper Speech Recognition tool - for real-time interview transcription."""
from __future__ import annotations

import logging
from typing import Any, ClassVar

from interview_os.core.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


class WhisperTool(Tool):
    name = "whisper_transcribe"
    description = "Transcribe audio to text using Whisper"
    parameters: ClassVar[dict[str, Any]] = {
        "audio_path": {"type": "string", "description": "Path to audio file"}
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        audio_path = kwargs.get("audio_path", "")
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return ToolResult(success=True, data={"text": result["text"], "segments": result["segments"]})
        except ImportError:
            return ToolResult(success=False, error="whisper not installed. Run: pip install openai-whisper")
        except Exception:  # noqa: BLE001 - third-party model errors vary
            return ToolResult(success=False, error="Speech transcription failed")
