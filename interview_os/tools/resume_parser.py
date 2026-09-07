"""Resume Parser tool - extracts structured data from resumes."""
from __future__ import annotations

import logging
from asyncio import to_thread
from typing import Any, ClassVar

from interview_os.core.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


class ResumeParserTool(Tool):
    name = "resume_parser"
    description = "Parse a resume file (PDF/TXT) and return raw text for LLM analysis"
    parameters: ClassVar[dict[str, Any]] = {
        "file_path": {"type": "string", "description": "Path to resume file"}
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        if file_path.endswith(".pdf"):
            from interview_os.tools.pdf_parser import PDFParserTool
            result = await PDFParserTool().execute(file_path=file_path)
            if result.success:
                return ToolResult(success=True, data=result.data)
            return result
        try:
            text = await to_thread(self._read_text, file_path)
            return ToolResult(success=True, data={"text": text, "pages": 1})
        except Exception:  # noqa: BLE001 - tool returns failures as values
            return ToolResult(success=False, error="Resume text parsing failed")

    @staticmethod
    def _read_text(file_path: str) -> str:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
