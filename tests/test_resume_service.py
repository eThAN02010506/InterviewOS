from io import BytesIO

import pytest
from docx import Document

from interview_os.services.resume_service import ResumeProcessingError, ResumeProcessor


def make_docx(*paragraphs: str) -> bytes:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "技能"
    table.cell(0, 1).text = "Python, FastAPI"
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def test_docx_resume_extracts_text_and_review_items():
    content = make_docx(
        "Ada Lovelace | ada@example.com | 13800138000",
        "教育经历 University 本科 2020",
        "工作经历 Example有限公司 2021-2025",
        "将接口延迟降低 35%，服务 10 万用户",
    )

    text, review = ResumeProcessor().process("ada.docx", content)

    assert "Python, FastAPI" in text
    assert review.metadata.file_type == "docx"
    assert review.metadata.character_count == len(text)
    categories = {claim.category for claim in review.claims}
    assert {"education", "employment", "achievement"} <= categories


def test_resume_review_flags_future_year_without_calling_it_false():
    content = make_docx(
        "ada@example.com 工作经历 2099 Example有限公司", "教育经历 大学", "技能 Python"
    )
    _, review = ResumeProcessor().process("future.docx", content)

    issue = next(item for item in review.issues if item.code == "future_date")
    assert issue.severity == "warning"
    assert "请确认" in issue.message
    assert all(claim.status == "unverified" for claim in review.claims)


@pytest.mark.parametrize("filename", ["resume.doc", "resume.txt", "resume.jpg"])
def test_resume_rejects_unsupported_formats(filename):
    with pytest.raises(ResumeProcessingError, match="PDF 和 Word"):
        ResumeProcessor().process(filename, b"content")


def test_resume_rejects_empty_document():
    with pytest.raises(ResumeProcessingError, match="为空"):
        ResumeProcessor().process("resume.pdf", b"")


def _register_chinese_font() -> str:
    """Register a system Chinese font for reportlab so CJK renders, not tofu."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for index, path in enumerate(candidates):
        import os

        if os.path.exists(path):
            name = f"CJKTest{index}"
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:  # noqa: BLE001, S112 - try the next candidate font
                continue
    raise RuntimeError("No usable system CJK font found for the reportlab test")


def make_pdf_with_table() -> bytes:
    from io import BytesIO as _BytesIO

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table

    font_name = _register_chinese_font()
    registerFontFamily(font_name, normal=font_name, bold=font_name, italic=font_name, boldItalic=font_name)
    buffer = _BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)
    body = ParagraphStyle("body", fontName=font_name, fontSize=10, leading=14)
    story = [
        Paragraph("Ada Lovelace 简历", body),
        Paragraph("工作经历：字节跳动 高级工程师 2022-至今", body),
        Table(
            [
                ["负责内容", "推荐系统架构"],
                ["关键成果", "延迟降低 40%"],
            ],
            colWidths=[40 * mm, 60 * mm],
            style=[
                ("FONTNAME", (0, 0), (-1, -1), font_name),
            ],
        ),
        Paragraph("项目经验：分布式缓存系统，QPS 提升 3 倍。", body),
    ]
    document.build(story)
    return buffer.getvalue()


def test_pdf_resume_extracts_text_and_table():
    content = make_pdf_with_table()

    text, review = ResumeProcessor().process("ada.pdf", content)

    assert review.metadata.file_type == "pdf"
    assert review.metadata.page_count >= 1
    assert "Ada Lovelace 简历" in text
    assert "延迟降低 40%" in text
    assert "推荐系统架构" in text


def test_pdf_extraction_keeps_line_breaks():
    content = make_pdf_with_table()
    text, _ = ResumeProcessor().process("lines.pdf", content)
    # Multi-line headings must not be flattened into one space-separated blob;
    # pdfplumber keeps structural line breaks.
    assert "Ada Lovelace 简历" in text


def test_resume_normalization_removes_control_characters_and_reports_artifacts():
    normalized = ResumeProcessor._normalize("New H\x00C Group\nExperience")
    issues = ResumeProcessor._find_issues(normalized, had_encoding_artifacts=True)
    assert "\x00" not in normalized
    assert any(issue.code == "encoding_artifacts" for issue in issues)


def test_employment_claim_requires_timeline_signal():
    claims = ResumeProcessor._find_claims(
        "Example Company is a global technology company.\n2020-01 Example Company Senior Manager"
    )
    employment = [claim.statement for claim in claims if claim.category == "employment"]
    assert employment == ["2020-01 Example Company Senior Manager"]


def test_vertical_date_range_merges_end_date_and_recognizes_english_company():
    # A vertical date layout: start + employer on one line, end date alone next.
    text = (
        "2018-07 to ZUORA\n"
        "2022-12\n"
        "Senior Recruiting Manager\n"
        "Led Talent Acquisition across APAC.\n"
    )
    claims = ResumeProcessor._find_claims(text)
    employment = [c.statement for c in claims if c.category == "employment"]
    # The end date is folded into the employer line, and the bare English
    # company name (no Inc/Corp/Company suffix) is still recognized.
    assert any("2018-07 to ZUORA 2022-12" in statement for statement in employment)


def test_parse_structured_response_fenced_json():
    from interview_os.services.resume_llm import _parse_structured_response

    raw = '''```json
    {"education":[{"category":"education","institution":"SMIC","title":"High School","date_range":"08/2020 - 06/2024","description":"Pursuing high school"}],"employment":[],"research":[{"category":"research","institution":"Tsinghua University","title":"Participant","date_range":"07/2023 - 09/2023","description":"Anomaly detection research"}]}
    ```'''
    sections = _parse_structured_response(raw)
    assert len(sections) == 2
    assert sections[0].category == "education"
    assert sections[0].institution == "SMIC"
    assert sections[1].category == "research"
    assert sections[1].institution == "Tsinghua University"


def test_parse_structured_response_invalid_returns_empty():
    from interview_os.services.resume_llm import _parse_structured_response

    assert _parse_structured_response("not json at all") == []
    assert _parse_structured_response("") == []


async def test_structure_resume_with_llm_success():
    from interview_os.services.resume_llm import structure_resume_with_llm

    class _StubLLM:
        async def chat(self, messages, **kwargs):
            return '{"education":[{"category":"education","institution":"SMIC","date_range":"08/2020 - 06/2024"}]}'

    sections = await structure_resume_with_llm("SMIC high school", _StubLLM())
    assert len(sections) == 1
    assert sections[0].category == "education"


async def test_structure_resume_with_llm_failure_returns_empty():
    from interview_os.services.resume_llm import structure_resume_with_llm

    class _BadLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("llm down")

    assert await structure_resume_with_llm("SMIC", _BadLLM()) == []
