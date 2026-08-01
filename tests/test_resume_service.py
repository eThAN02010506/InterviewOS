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
