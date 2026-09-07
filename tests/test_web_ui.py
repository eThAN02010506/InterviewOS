"""Static contracts for the dependency-free web client."""

import re
from pathlib import Path

WEB_DIR = Path(__file__).parents[1] / "interview_os" / "web"


def web_scripts() -> str:
    """Return the application and extracted ES modules as one static contract."""
    paths = [WEB_DIR / "app.js", *(WEB_DIR / "modules").glob("*.js")]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_web_document_has_unique_element_ids():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    identifiers = re.findall(r'\bid="([^"]+)"', html)

    assert len(identifiers) == len(set(identifiers))


def test_web_entrypoint_uses_explicit_es_modules():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert '<script type="module" src="./app.js?v=STATIC_VERSION">' in html
    assert '<link rel="stylesheet" href="./styles.css?v=STATIC_VERSION">' in html
    assert '<meta name="interview-os-api-base" content="">' in html
    assert "from './modules/api.js'" in script
    assert "from './modules/runtime.js'" in script
    assert "from './modules/candidate-view.js'" in script
    assert "from './modules/live-view.js'" in script
    assert "from './modules/mock-view.js'" in script
    assert "from './modules/state.js'" in script
    assert "from './modules/ui.js'" in script
    assert (WEB_DIR / "modules" / "api.js").is_file()
    assert (WEB_DIR / "modules" / "runtime.js").is_file()
    assert (WEB_DIR / "modules" / "candidate-view.js").is_file()
    assert (WEB_DIR / "modules" / "live-view.js").is_file()
    assert (WEB_DIR / "modules" / "mock-view.js").is_file()
    assert (WEB_DIR / "modules" / "state.js").is_file()
    assert (WEB_DIR / "modules" / "ui.js").is_file()


def test_all_application_requests_cross_the_runtime_aware_api_boundary():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    api_module = (WEB_DIR / "modules" / "api.js").read_text(encoding="utf-8")
    runtime_module = (WEB_DIR / "modules" / "runtime.js").read_text(encoding="utf-8")

    assert "fetch(" not in script
    assert "fetch(resolveApiUrl(path, runtime)" in api_module
    assert "X-InterviewOS-Bootstrap" in api_module
    assert "__INTERVIEW_OS_RUNTIME__" in runtime_module
    assert "desktop_runtime_config" in runtime_module
    assert "mode === 'desktop'" in runtime_module
    assert "headers.set('X-InterviewOS-Bootstrap'" in api_module
    assert "api.raw" in api_module
    assert "api.blob" in api_module
    assert "api.text" in api_module
    assert "api.abortActive" in api_module
    assert "api.setGlobalAbortSignal" in api_module
    assert "controller.abort(error);" in api_module
    assert "const requestSessionId = state.sessionId" in api_module
    assert "state.sessionId !== requestSessionId" in api_module
    assert "requestContext.assertFresh()" in api_module
    assert "const data = await requestContext.response.json();" in api_module
    assert "response.json().catch" not in api_module.split("async function api", 1)[1]


def test_desktop_shutdown_waits_for_recording_flush_before_sidecar_exit():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert "tasks.push(stopSessionRecorder())" in script
    assert "desktop-shutdown-requested" in script
    assert "desktop_shutdown_bridge_ready" in script
    assert "desktop_renderer_ready_to_exit" in script
    assert "desktop_renderer_cancel_exit" in script
    assert "onCloseRequested" in script
    assert "flushActiveMedia()" in script
    assert "Promise.allSettled(tasks)" in script
    assert "finalizeMockVoiceRecording({quiet:true})" in script
    assert "mockVoiceStartPending" in script
    assert "mockVoiceFinalizePromise" in script
    assert "mockVoicePendingUpload" in script
    assert "restoreMockAnswerDraft(sessionDraft)" in script
    assert "recordings/${recordingId}/audio" in script
    assert "wav.type==='audio/wav'?'answer.wav':liveAudioFilename" in script
    assert "question:retryVoiceTarget?.question||mockSession.pending_follow_up" in script
    assert 'id="mock-discard-draft"' in html
    assert "discardMockDraftIfNeeded" in script
    assert "runMediaTransition" in script
    assert "requestedSessionId!==state.sessionId" in script
    assert "sessionRecordingSessionId" in script
    assert "liveRecordingSessionId" in script
    assert "liveContinuousQueue[0]" in script
    assert "form.append('file',part.blob,filename)" in script
    assert "if(!recordingId)throw new Error('语音片段未登记，已拒绝上传')" in script
    assert "form.append('recording_id',recordingId)" in script
    assert "mediaFlushInProgress" in script
    assert "mediaTransitionPromise" in script
    assert "waitForDesktopShutdownStep" in script
    assert "await promise.catch(()=>{})" in script
    assert "api.abortActive(error)" in script
    assert "prepareCurrentSessionBoundary" in script
    assert "enforceFailClosedPause" in script
    assert "api.raw(`/api/live-interviews/${sessionId}/status`" in script
    assert "keepalive:true" in script
    assert "cancelLiveSuggestionStream" in script
    assert "sessionAudioSaveError" in script
    assert "sessionAudioPending" in script
    assert "if(sessionPartQueue.length||sessionAudioUploading)await uploadSessionAudio()" in script
    assert "status-barrier" in script
    assert "expected_revision" in script
    assert "audio/captures/${recordingId}" in script
    assert "settledLiveAudioCaptureRegistration" in script
    assert "freezeLocalCapture({includeMock:true,sessionId:state.sessionId})" in script
    assert "markLivePauseRequired(shutdownSessionId)" in script
    assert "if(hadLocalCapture&&sessionId)markLivePauseRequired(sessionId)" in script
    assert "interviewos.pauseRequiredSession" in script
    assert "pauseCaptureEpoch" in script
    assert "pauseStatusRevision" in script
    assert "remoteOwnershipChanged:true" in script
    assert "recoverPageBoundary" in script
    assert "本机旧轮次收音已停止并保存" in script
    assert "sessionRecorderStartSessionId" in script
    assert "LIVE_CONTINUOUS_QUEUE_HIGH_WATER" in script
    assert "staleAudioSettingsRefresh=requireRuntimeSettings().catch" in script
    assert "stillLocallyOwned(captureEntries[index])" in script
    assert "expected_capture_epoch" in script
    assert "已恢复原会话" in script
    assert "authenticationRecoverySessionId=livePauseRequiredSessionId" in script
    assert "livePauseRequiredSessionId===state.sessionId" in script
    assert (
        "if(liveContinuousResumeSessionId===operationSessionId){\n"
        "        await waitForCaptureStartup(startLiveContinuousVad({fromTransition:true}));"
        in script
    )


def test_runtime_settings_hydration_fails_closed_before_audio_capture():
    script = web_scripts()

    assert "let runtimeSettingsReady = false" in script
    assert "runtimeSettingsReady=true" in script
    assert "await loadSettings({required:true})" in script
    assert "!runtimeSettingsReady||liveStatusUncertainSessionId" in script
    assert "!settingsReady || status === 'completed'" in script
    assert "settingsLoadGeneration" in script
    assert "!runtimeSettingsReady||authenticationSuspended||desktopShutdownPreparing" in script
    assert "请先暂停实时面试并保存所有录音" in script


def test_every_secret_bearing_audio_provider_is_configurable_in_the_ui():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="setting-live-audio-key" type="password"' in html
    assert "api_key:providerSecret('setting-live-audio-key','clear-live-audio-key')" in script
    assert "'setting-live-audio-key'" in script
    assert "await requireRuntimeSettings();clearAuthenticationEditors();" in script
    for clear_id in (
        "clear-tavily-key",
        "clear-brave-key",
        "clear-llm-key",
        "clear-asr-key",
        "clear-tts-key",
        "clear-live-audio-key",
        "clear-resume-llm-key",
    ):
        assert f'id="{clear_id}" type="checkbox"' in html
        assert f"'{clear_id}'" in script


def test_mock_and_live_microphones_are_mutually_exclusive_and_generation_safe():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "function liveMicrophoneCaptureActive()" in script
    assert "if(liveMicrophoneCaptureActive()){toast('实时面试仍在使用麦克风" in script
    assert "if(mockVoiceCaptureBusy()){toast('请先停止并保存当前模拟回答录音" in script
    assert "preflightLiveGeneration(operationSessionId,operationGeneration,'暂停')" in script
    assert "preflightLiveGeneration(operationSessionId,operationGeneration,'恢复')" in script
    assert "preflightLiveGeneration(operationSessionId,operationGeneration,'结束')" in script
    assert "liveTransitionMatches(data.state?.live_interview" in script
    assert "freezeLocalCapture({includeMock:true,sessionId:operationSessionId})" in script
    assert "flushActiveMedia({includeMock:true,includeSessionRecording:true})" in script
    assert "if(status==='active')body.operation_id=operationId" in script
    assert "const resumeOperationId=pausing?'':newRecordingPartId()" in script
    assert "liveTransitionMatches(data.state?.live_interview,operationGeneration,'active',resumeOperationId)" in script


def test_fail_closed_pause_never_targets_an_unknown_live_generation():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    calls = re.findall(r"await enforceFailClosedPause\(([^)\n]+)\)", script)

    assert calls
    assert all("," in arguments for arguments in calls)
    assert "markLiveStatusUncertain(sessionId,expectedGeneration,alternateGeneration)" in script
    assert "liveStatusUncertainExpectedGeneration" in script
    assert "liveStatusUncertainAlternateGeneration" in script
    assert "throw liveOwnershipChangedError(current,'暂停新的监听轮次')" in script


def test_mock_recording_preserves_user_edits_and_serializes_mutations():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "mockVoiceStartPending=false" in script
    assert "mockAnswerEditRevision+=1" in script
    assert "mockAnswerEditRevision!==voiceContext.answerEditRevision" in script
    assert "answerWasEdited?'最终转写已保存" in script
    assert "mockSpeechFeedbackPollGeneration" in script
    assert "mockPendingRecordingId!==recordingId" in script
    assert "mockMutationInProgress=true" in script
    assert "navigateMockInterview('next'" in script
    assert "speakButton.disabled=!current||captureBusy" in script
    assert "cancelMockQuestionSpeech();\n  clearMockQuestionAudio();" in script


def test_long_and_background_recordings_fail_safe_with_visible_feedback():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "MOCK_RECORDING_LIMIT_MS = 10 * 60 * 1000" in script
    assert "MOCK_AUDIO_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024" in script
    assert "boundedAsrPreviewChunks" in script
    assert "实时转写暂不可用" in script
    assert "pauseContinuousListeningForHiddenPage" in script
    assert "liveVadVoicedMs+=analysisElapsed" in script
    assert "asrPreviewAbort?.abort()" in script
    assert "实时转写预览超时" in script
    assert "initializeDesktopShutdownBridge" in script
    assert "desktopShutdownBridgeReady=false" in script


def test_removed_mock_audio_and_empty_session_lists_clear_stale_client_state():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "if(!response?.audio_file){if(mockVoicePendingUpload)return;cancelMockAnswerAudioLoad();clearObjectUrl('answer')" in script
    assert "localStorage.removeItem('interviewos.session');\n      select.value = '';" in script
    assert "state.sessionId=data.id;state.session=null;activeLoadedSessionId=''" in script
    assert "await loadSession().catch(()=>{});toast(`运行失败" in script
    assert "function clearSessionEditors()" in script
    assert "'mock-answer','live-text','transcript-input'" in script


def test_long_live_recording_rolls_before_pcm_upload_limit():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "const LIVE_RECORDING_ROLLOVER_MS = 10 * 60 * 1000" in script
    assert "const LIVE_RECORDING_WARNING_MS = 9 * 60 * 1000" in script
    assert "liveRecordingLimitReachedId=recordingId" in script
    assert "本段已达到 10 分钟安全上限" in script
    assert "请点击继续录音，或开启连续监听" in script


def test_whole_session_recording_rotates_and_uploads_bounded_fifo_parts():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "const SESSION_RECORDING_PART_MS = 5 * 60 * 1000" in script
    assert "const SESSION_RECORDING_PART_BYTES = 64 * 1024 * 1024" in script
    assert "const SESSION_PART_QUEUE_HIGH_WATER = 3" in script
    assert "const SESSION_PART_QUEUE_BYTES_HIGH_WATER = 192 * 1024 * 1024" in script
    assert "sessionPartQueue.sort((left,right)=>left.sequence-right.sequence)" in script
    assert "while(sessionPartQueue.length)" in script
    assert "const part=sessionPartQueue.find(item=>item.sequence===sessionNextUploadSequence)" in script
    assert "sessionFinalizedPartSequences.has(sessionNextUploadSequence)" in script
    assert "form.append('recording_id',part.recordingId)" in script
    assert "form.append('expected_audio_revision',String(part.archiveRevision))" in script
    assert "function mergeSessionAudioArchiveState" in script
    assert "for(const part of [...(currentLive.audio_parts||[]),...(incomingLive.audio_parts||[])])" in script
    assert "mergeSessionAudioArchiveState(data.state,recordingSessionId,part.archiveRevision)" in script


def test_audio_delete_conflict_refreshes_revision_without_dropping_local_recovery_parts():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    refresh = script.index("replaceSessionAudioArchiveState(current.state,sessionId)")
    conflict = script.index("录音集合已在其他窗口发生变化，请再次确认删除")
    discard = script.index(
        "sessionPartQueue=sessionPartQueue.filter(part=>part.sessionId!==sessionId)"
    )
    assert refresh < conflict < discard
    assert "audio_archive_revision:Number(incomingLive.audio_archive_revision||0)" in script
    assert "await startSessionRecordingPart(successor)" in script
    assert "requestSessionPartStop(oldPart,`rotate-${trigger}`)" in script
    assert script.index("await startSessionRecordingPart(successor)") < script.index(
        "requestSessionPartStop(oldPart,`rotate-${trigger}`)"
    )
    assert "sessionQueueAtHighWater(oldPart)" in script
    assert "requestSessionRecorderStop('upload-failed')" in script
    assert "sessionPartUploadsSuppressed=true" in script
    assert "sessionPartQueue=sessionPartQueue.filter(part=>part.sessionId!==sessionId)" in script
    assert "sessionPartQueue.length||sessionAudioUploading||pendingLiveCaptureStart" in script


def test_candidate_next_action_and_mock_completion_have_ui_contracts():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = web_scripts()

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
    assert 'id="custom-question-form"' in html
    assert 'id="custom-question-text"' in html
    assert "/questions`" in script
    assert "问题已解析并加入练习" in script
    assert 'id="mock-question-understanding"' in html
    assert 'id="mock-question-understanding-body"' in html
    assert "这道题真正想验证什么" in script
    assert "data-related-question" in script
    assert "展开深度解析" in script
    assert "回答质量分界" in script
    assert "从已确认经历中选材" in script
    assert "由浅入深的追问题树" in script
    assert "恢复引导" in script
    assert "压力迁移" in script
    assert "追问原因：" in script


def test_accessible_interaction_states_are_styled():
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert ":focus-visible" in styles
    assert "button:disabled" in styles
    assert ".loading::after" in styles
    assert "prefers-reduced-motion" in styles


def test_role_theme_and_mobile_layout_contracts_are_present():
    script = web_scripts()
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert "document.documentElement.dataset.role = role" in script
    assert "function localizedNextAction(value)" in script
    assert 'html[data-role="interviewer"]' in styles
    assert ".user-chip { display: none; }" in styles
    assert ".review-summary, .review-claim" in styles
    assert "overflow-x: hidden" in styles


def test_company_context_hydration_targets_both_real_form_fields():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="candidate-company-context"' in html
    assert 'id="enterprise-context"' in html
    assert "'candidate-company-context': s?.company?.context" in script
    assert "'enterprise-context': s?.company?.context" in script


def test_mock_auto_speech_is_scoped_to_visible_mock_view():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "state.view==='mock'&&document.visibilityState==='visible'" in script
    assert (
        "if (view !== 'mock') { cancelMockQuestionSpeech(); clearMockQuestionAudio(); pauseMockAnswerPlayback(); }"
        in script
    )
    assert "if(e.target.checked&&state.view==='mock')" in script


def test_lan_microphone_requires_secure_context_with_actionable_message():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="secure-context-warning"' in html
    assert "window.isSecureContext" in script
    assert "局域网语音功能需要 HTTPS" in script


def test_live_streaming_ui_serializes_preview_and_filters_short_vad_audio():
    script = web_scripts()

    assert "VAD_MIN_SPEECH_MS" in script
    assert "liveVadVoicedMs>=VAD_MIN_SPEECH_MS" in script
    assert "!['in-speech','finalizing'].includes(liveVadState)" in script
    assert "liveVadHeaderChunk" in script
    assert "liveVadUtteranceChunks.push(liveVadHeaderChunk)" in script
    assert "if(asrPreviewAbort===controller){asrPreviewAbort=null;asrPreviewBusy=false;}" in script
    assert "resetAsrPreview(){asrPreviewSequence+=1;asrPreviewLastAt=0" in script
    assert "$('live-stream-plan').disabled = liveStatusUncertain || status !== 'active'" in script
    assert "noiseSuppression:true" in script


def test_final_report_distinguishes_locked_scores_from_model_narrative():
    script = web_scripts()
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert "AI 按证据框架生成 · 分数与缺口由规则锁定" in script
    assert "item.assessment" in script
    assert "item.next_probe" in script
    assert ".evaluation-narrative" in styles
    assert ".next-probe" in styles
