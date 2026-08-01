"""Safe resume extraction and deterministic first-pass review."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from interview_os.core.state import (
    ResumeClaim,
    ResumeFileMetadata,
    ResumeIssueSeverity,
    ResumeReview,
    ResumeValidationIssue,
)

MAX_RESUME_BYTES = 10 * 1024 * 1024
MAX_RESUME_CHARACTERS = 100_000
SUPPORTED_RESUME_TYPES = {".pdf", ".docx"}


class ResumeProcessingError(ValueError):
    """Raised when an uploaded resume cannot be handled safely."""


class ResumeProcessor:
    """Extract text from supported files, then produce a bounded review in O(n)."""

    def process(self, filename: str, content: bytes) -> tuple[str, ResumeReview]:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_RESUME_TYPES:
            raise ResumeProcessingError("仅支持 PDF 和 Word (.docx) 简历")
        if not content:
            raise ResumeProcessingError("上传的简历为空")
        if len(content) > MAX_RESUME_BYTES:
            raise ResumeProcessingError("简历不能超过 10 MB")

        text, pages = (
            self._extract_pdf(content) if suffix == ".pdf" else self._extract_docx(content)
        )
        had_encoding_artifacts = bool(
            re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", text)
        )
        text = self._normalize(text)
        if not text:
            raise ResumeProcessingError("未能从简历中提取文字；扫描版 PDF 请先进行 OCR")
        if len(text) > MAX_RESUME_CHARACTERS:
            text = text[:MAX_RESUME_CHARACTERS]

        review = ResumeReview(
            metadata=ResumeFileMetadata(
                filename=Path(filename).name,
                file_type=suffix.removeprefix("."),
                size_bytes=len(content),
                page_count=pages,
                character_count=len(text),
            ),
            issues=self._find_issues(text, had_encoding_artifacts=had_encoding_artifacts),
            claims=self._find_claims(text),
            reviewed_at=datetime.now(timezone.utc),
        )
        return text, review

    @staticmethod
    def _extract_pdf(content: bytes) -> tuple[str, int]:
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ResumeProcessingError("PDF 已加密，请上传未加密版本")
            page_texts = []
            for page in reader.pages:
                try:
                    page_texts.append(page.extract_text(extraction_mode="layout") or "")
                except TypeError:
                    page_texts.append(page.extract_text() or "")
            return "\n\n".join(page_texts), len(reader.pages)
        except ResumeProcessingError:
            raise
        except Exception as exc:
            raise ResumeProcessingError("PDF 文件损坏或格式不受支持") from exc

    @staticmethod
    def _extract_docx(content: bytes) -> tuple[str, int]:
        try:
            from docx import Document

            document = Document(BytesIO(content))
            blocks = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        blocks.append(" | ".join(cells))
            return "\n".join(blocks), 1
        except Exception as exc:
            raise ResumeProcessingError("Word 文件损坏或不是有效的 .docx 文件") from exc

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _find_issues(
        text: str, *, had_encoding_artifacts: bool = False
    ) -> list[ResumeValidationIssue]:
        issues: list[ResumeValidationIssue] = []
        if had_encoding_artifacts:
            issues.append(
                ResumeValidationIssue(
                    code="encoding_artifacts",
                    severity=ResumeIssueSeverity.WARNING,
                    field="document",
                    message="PDF 包含异常字体编码，个别公司名或符号可能需要对照原文件确认",
                )
            )
        if len(text) < 200:
            issues.append(
                ResumeValidationIssue(
                    code="low_text",
                    severity=ResumeIssueSeverity.WARNING,
                    message="提取到的文字较少，请核对解析结果",
                )
            )
        if not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
            issues.append(
                ResumeValidationIssue(
                    code="missing_email",
                    severity=ResumeIssueSeverity.WARNING,
                    field="contact",
                    message="未识别到邮箱地址",
                )
            )
        if not re.search(r"(?:\+?86[- ]?)?1[3-9]\d{9}|\+?\d[\d ()-]{7,}\d", text):
            issues.append(
                ResumeValidationIssue(
                    code="missing_phone",
                    severity=ResumeIssueSeverity.INFO,
                    field="contact",
                    message="未识别到联系电话",
                )
            )
        section_groups = {
            "experience": ("工作经历", "工作经验", "experience", "employment"),
            "education": ("教育经历", "教育背景", "education", "university"),
            "skills": ("技能", "专业能力", "skills", "technologies"),
        }
        lower = text.lower()
        for field, labels in section_groups.items():
            if not any(label in lower for label in labels):
                issues.append(
                    ResumeValidationIssue(
                        code=f"missing_{field}",
                        severity=ResumeIssueSeverity.INFO,
                        field=field,
                        message=f"未识别到{labels[0]}版块",
                    )
                )
        current_year = datetime.now(timezone.utc).year
        future_years = sorted(
            {
                int(year)
                for year in re.findall(r"\b(?:19|20)\d{2}\b", text)
                if int(year) > current_year
            }
        )
        if future_years:
            issues.append(
                ResumeValidationIssue(
                    code="future_date",
                    severity=ResumeIssueSeverity.WARNING,
                    field="timeline",
                    message=f"发现未来年份：{', '.join(map(str, future_years))}，请确认时间线",
                )
            )
        return issues

    @staticmethod
    def _find_claims(text: str) -> list[ResumeClaim]:
        claims: list[ResumeClaim] = []
        patterns = (
            (
                "education",
                re.compile(
                    r"大学|学院|university|college|bachelor|master|博士|硕士|本科", re.IGNORECASE
                ),
                "supporting_document",
            ),
            (
                "employment",
                re.compile(r"有限公司|集团|inc\.?|corp\.?|company|任职|就职", re.IGNORECASE),
                "public_source_or_reference",
            ),
            (
                "certification",
                re.compile(r"证书|认证|certified|certification|certificate", re.IGNORECASE),
                "supporting_document",
            ),
            (
                "achievement",
                re.compile(r"\d+(?:\.\d+)?\s*(?:%|倍|x\b|万|亿|million|k\b)", re.IGNORECASE),
                "candidate_confirmation",
            ),
        )
        seen: set[tuple[str, str]] = set()
        for line in text.splitlines():
            statement = line.strip()
            if not 8 <= len(statement) <= 300:
                continue
            for category, pattern, method in patterns:
                if category == "employment" and not re.search(
                    r"\b(?:19|20)\d{2}\b|\d{4}[-/.]\d{1,2}", statement
                ):
                    continue
                key = (category, statement)
                if pattern.search(statement) and key not in seen:
                    claims.append(
                        ResumeClaim(
                            category=category, statement=statement, verification_method=method
                        )
                    )
                    seen.add(key)
                    if len(claims) == 40:
                        return claims
        return claims
