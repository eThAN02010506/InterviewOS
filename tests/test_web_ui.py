"""Static contracts for the dependency-free web client."""

import re
from pathlib import Path

WEB_DIR = Path(__file__).parents[1] / "interview_os" / "web"


def test_web_document_has_unique_element_ids():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    identifiers = re.findall(r'\bid="([^"]+)"', html)

    assert len(identifiers) == len(set(identifiers))


def test_candidate_next_action_and_mock_completion_have_ui_contracts():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="candidate-next-action"' in html
    assert 'id="candidate-next-action" type="button" data-new-practice="true"' in html
    assert 'id="mock-status" aria-live="polite"' in html
    assert "function candidateHomeAction(session)" in script
    assert 'data-route="candidate-report"' in script
    assert "最后一题反馈" in script
    assert "新建练习会话" in script
    assert 'id="mock-requirements"' in html
    assert 'id="mock-example"' in html
    assert "这道题问了什么，你实际回答了什么" in script
    assert "基于你本次回答的重组示范" in script


def test_accessible_interaction_states_are_styled():
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert ":focus-visible" in styles
    assert "button:disabled" in styles
    assert ".loading::after" in styles
    assert "prefers-reduced-motion" in styles


def test_role_theme_and_mobile_layout_contracts_are_present():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert "document.documentElement.dataset.role = role" in script
    assert "function localizedNextAction(value)" in script
    assert 'html[data-role="interviewer"]' in styles
    assert ".user-chip { display: none; }" in styles
    assert ".review-summary, .review-claim" in styles
    assert "overflow-x: hidden" in styles


def test_mock_auto_speech_is_scoped_to_visible_mock_view():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "state.view==='mock'&&document.visibilityState==='visible'" in script
    assert "if (view !== 'mock') { cancelMockQuestionSpeech(); clearMockQuestionAudio(); }" in script
    assert "if(e.target.checked&&state.view==='mock')" in script


def test_lan_microphone_requires_secure_context_with_actionable_message():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="secure-context-warning"' in html
    assert "window.isSecureContext" in script
    assert "局域网语音功能需要 HTTPS" in script


def test_live_streaming_ui_serializes_preview_and_filters_short_vad_audio():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "VAD_MIN_SPEECH_MS" in script
    assert "liveVadVoicedMs>=VAD_MIN_SPEECH_MS" in script
    assert "!['in-speech','finalizing'].includes(liveVadState)" in script
    assert "liveVadHeaderChunk" in script
    assert "liveVadUtteranceChunks.push(liveVadHeaderChunk)" in script
    assert "}finally{asrPreviewBusy=false;}" in script
    assert "resetAsrPreview(){asrPreviewSequence+=1;asrPreviewLastAt=0" in script
    assert "$('live-stream-plan').disabled = status !== 'active'" in script
    assert "noiseSuppression:true" in script


def test_final_report_distinguishes_locked_scores_from_model_narrative():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert "AI 按证据框架生成 · 分数与缺口由规则锁定" in script
    assert "item.assessment" in script
    assert "item.next_probe" in script
    assert ".evaluation-narrative" in styles
    assert ".next-probe" in styles
