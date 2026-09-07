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
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                page_texts = [page.extract_text() or "" for page in pdf.pages]
                text = "\n\n".join(page_texts)
                return ToolResult(success=True, data={"text": text, "pages": len(pdf.pages)})
        except ImportError:
            return ToolResult(success=False, error="pdfplumber not installed. Run: pip install pdfplumber")
        except Exception:  # noqa: BLE001 - third-party parser errors vary
            return ToolResult(success=False, error="PDF parsing failed")
