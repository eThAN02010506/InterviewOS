"""PDF Parser tool - extracts text from PDF files."""
from __future__ import annotations

import logging
from typing import Any, ClassVar

from interview_os.core.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


class PDFParserTool(Tool):
    name = "pdf_parser"
    description = "Extract text content from a PDF file"
    parameters: ClassVar[dict[str, Any]] = {
        "file_path": {"type": "string", "description": "Path to PDF file"}
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return ToolResult(success=True, data={"text": text, "pages": len(reader.pages)})
        except ImportError:
            return ToolResult(success=False, error="pypdf not installed. Run: pip install pypdf")
        except Exception as exc:  # noqa: BLE001 - third-party parser errors vary
            return ToolResult(success=False, error=str(exc))
