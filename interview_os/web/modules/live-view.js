import {state} from './state.js';
import {$, esc, tags} from './ui.js';

export function renderLiveView(runtime) {
  const {
    liveContinuousMode, liveContinuousQueueLength, liveContinuousUploading,
    livePendingAudio, liveRecorderActive, liveSuggestionStreamActive, microphoneAvailabilityMessage,
    liveStatusUncertain, mediaTransitionInProgress, sessionAudioPending, sessionAudioSaveError,
    sessionAudioUploading, sessionRecordingActive, sessionRecordingError, settingsReady,
    utterancePendingAudio, utteranceCaptureBusy,
    updateLiveVadNote
  } = runtime;
  const live = state.session?.live_interview;
  if (!$('live-transcript')) return;
  const statusLabels = {idle:'尚未开始',active:'监听中',paused:'已暂停',completed:'已结束'};
  const status = live?.status || 'idle';
  $('live-status-label').textContent = liveStatusUncertain ? '状态确认中 · 本机麦克风已停止' : statusLabels[status] || status;
  $('live-status-dot').className = liveStatusUncertain ? 'paused' : status === 'active' ? 'active' : status;
  $('live-consent-panel').classList.toggle('hidden', status !== 'idle' && status !== 'completed');
  $('live-pause').textContent = status === 'paused' ? '恢复' : '暂停';
  $('live-start').disabled = mediaTransitionInProgress || liveStatusUncertain || !settingsReady || status === 'completed';
  $('live-pause').disabled = mediaTransitionInProgress || liveStatusUncertain || !['active','paused'].includes(status);
  $('live-finish').disabled = mediaTransitionInProgress || liveStatusUncertain || !['active','paused'].includes(status) || liveContinuousMode || liveContinuousQueueLength > 0 || liveContinuousUploading;
  const microphoneUnavailable = !!microphoneAvailabilityMessage();
  $('live-record').disabled = mediaTransitionInProgress || liveStatusUncertain || !settingsReady || status !== 'active' || liveRecorderActive || microphoneUnavailable;
  $('live-dialogue').disabled = mediaTransitionInProgress || liveStatusUncertain || !settingsReady || status !== 'active' || liveRecorderActive || microphoneUnavailable;
  $('live-continuous').disabled = mediaTransitionInProgress || liveStatusUncertain || !settingsReady || status !== 'active' || liveRecorderActive || microphoneUnavailable;
  $('live-stop-continuous').disabled = mediaTransitionInProgress || !liveContinuousMode;
  const retryPendingAudio=$('live-retry-pending-audio');
  if(retryPendingAudio){const retryRecorder=!!sessionRecordingError&&status==='active';retryPendingAudio.textContent=retryRecorder?'重试启动全场录音':'重试保存待处理音频';retryPendingAudio.hidden=liveContinuousMode||(!retryRecorder&&!livePendingAudio&&liveContinuousQueueLength===0)||liveContinuousUploading;retryPendingAudio.disabled=mediaTransitionInProgress||liveStatusUncertain||!['active','paused','completed'].includes(status)||liveContinuousUploading;}
  const discardPendingAudio=$('live-discard-pending-audio');
  if(discardPendingAudio){discardPendingAudio.hidden=liveContinuousMode||!utterancePendingAudio;discardPendingAudio.disabled=mediaTransitionInProgress||utteranceCaptureBusy;}
  $('live-plan').disabled = liveStatusUncertain || status !== 'active';
  $('live-stream-plan').disabled = liveStatusUncertain || status !== 'active' || liveSuggestionStreamActive;
  if (liveContinuousMode || liveContinuousQueueLength || liveContinuousUploading) {
    if (liveContinuousMode) {
      updateLiveVadNote();
    } else {
      const uploading = liveContinuousUploading ? '，正在转写 1 段' : '';
      $('live-recording-note').textContent = `连续监听已停止：队列 ${liveContinuousQueueLength} 段${uploading}。分段会先标记为“待确认”，请审阅说话人与文本后再归档证据。`;
    }
  }
  const sessionRecording = $('live-session-recording');
  if (sessionRecording) {
    const audioFile = live?.audio_file;
    const audioParts = live?.audio_parts || [];
    const recordingStatus = $('session-recording-status');
    const downloadBtn = $('session-audio-download');
    const deleteBtn = $('session-audio-delete');
    const partsNode = $('session-audio-parts');
    if (sessionRecordingActive || sessionAudioUploading) {
      sessionRecording.hidden = false;
      const saved=audioParts.length?`，已保存 ${audioParts.length} 段`:'';
      recordingStatus.textContent = sessionAudioUploading?`正在保存新的全场录音${saved}`:status === 'paused' ? `全场录音已暂停${saved}` : `全场录音中…${saved}`;
      recordingStatus.classList.remove('error');
      downloadBtn.hidden = !audioFile;
      deleteBtn.hidden = !audioFile;
      deleteBtn.disabled = true;
    } else if (sessionAudioPending || sessionAudioSaveError || sessionRecordingError) {
      sessionRecording.hidden = false;
      const saved=audioParts.length?`；此前已保存 ${audioParts.length} 段`:'；此前没有已保存分片';
      recordingStatus.textContent = `${sessionAudioSaveError||sessionRecordingError||'新的全场录音分片正在等待保存'}${saved}`;
      recordingStatus.classList.add('error');
      downloadBtn.hidden = !audioFile;
      deleteBtn.hidden = !audioFile && !sessionAudioPending;
      deleteBtn.disabled = mediaTransitionInProgress;
    } else if (audioFile && state.sessionId) {
      sessionRecording.hidden = false;
      recordingStatus.textContent = audioParts.length > 1 ? `全场录音已保存 ${audioParts.length} 段` : '全场录音已保存';
      recordingStatus.classList.remove('error');
      downloadBtn.hidden = false;
      deleteBtn.hidden = false;
      deleteBtn.disabled = mediaTransitionInProgress;
    } else {
      sessionRecording.hidden = true;
      deleteBtn.hidden = true;
    }
    if(partsNode){
      partsNode.innerHTML=audioParts.length>1?audioParts.map((part,index)=>`<button class="btn compact" type="button" data-session-audio-id="${esc(part.id)}" data-session-audio-file="${esc(part.audio_file)}">下载第 ${index+1} 段</button>`).join(''):'';
    }
  }
  const segments = live?.segments || [];
  const recordedSegmentIds = new Set((state.session?.live_interview_records || []).flatMap(record => record.transcript_segment_ids || []));
  const recordBySegmentId = new Map((state.session?.live_interview_records || []).flatMap(record => (record.transcript_segment_ids || []).map(segmentId => [segmentId, record])));
  const pendingCandidateSegments = segments.filter(segment => segment.speaker === 'candidate' && !recordedSegmentIds.has(segment.id));
  const confirmedRecords = (state.session?.live_interview_records || []).filter(record => record.source === 'live_interview');
  const boundarySuggestions = live?.answer_boundary_suggestions || [];
  const confirmedCount = confirmedRecords.length;
  const actionCard = live?.action_card;
  if ($('live-action-card')) {
    const refs = actionCard?.source_refs || [];
    $('live-action-card').className = actionCard ? `action-card ${esc(actionCard.priority || 'medium')}` : 'action-card empty-state';
    $('live-action-card').innerHTML = actionCard ? `<div class="action-top"><span>${esc(actionCard.action_type)}</span><b>${esc(actionCard.priority)}</b></div><h3>${esc(actionCard.title)}</h3><p>${esc(actionCard.detail)}</p><div class="action-evidence">${esc(actionCard.evidence_status || '尚无证据')}</div><div class="action-ctas"><button type="button" class="action-primary" data-live-action="primary">${esc(actionCard.primary_cta)}</button>${actionCard.secondary_cta?`<button type="button" class="action-secondary" data-live-action="secondary">${esc(actionCard.secondary_cta)}</button>`:''}</div>${refs.length?`<div class="action-refs">${refs.map(ref=>`<small>${esc(ref)}</small>`).join('')}</div>`:''}` : '开始实时会话后显示下一步动作。';
  }
  $('live-transcript').className = segments.length ? 'transcript-stream' : 'transcript-stream empty-state';
  $('live-transcript').innerHTML = segments.length ? segments.map(segment => {
    const isCandidate = segment.speaker === 'candidate';
    const recorded = recordedSegmentIds.has(segment.id);
    const record = recordBySegmentId.get(segment.id);
    const evidenceAction = isCandidate ? (recorded ? `<small class="evidence-confirmed">已归档为证据</small><button class="btn compact" data-live-revoke="${esc(record?.id || '')}">撤销证据</button>` : `<button class="btn compact" data-live-confirm="${esc(segment.id)}">确认为证据</button>`) : '';
    const editAction = recorded ? '' : `<button class="btn compact" data-live-edit="${esc(segment.id)}">修改</button>`;
    return `<div class="transcript-segment ${esc(segment.speaker)}" data-segment-id="${esc(segment.id)}"><span>${segment.speaker==='candidate'?'候选人':segment.speaker==='interviewer'?'面试官':'待确认'} · ${esc(segment.source)}</span><p>${esc(segment.text)}</p><div class="transcript-actions">${editAction}${evidenceAction}</div></div>`;
  }).join('') : '尚无转写片段。';
  const suggestion = [...(live?.suggestions || [])].reverse().find(item => item.status === 'pending') || [...(live?.suggestions || [])].reverse()[0];
  if (!liveSuggestionStreamActive) {
    $('live-suggestion').className = suggestion ? 'suggestion-card' : 'empty-state';
    $('live-suggestion').innerHTML = suggestion ? `<div class="suggestion-meta"><span>${esc(suggestion.question_type)}</span><b>${esc(suggestion.competency)}</b><em>${Math.round((suggestion.confidence||0)*100)}%</em></div><h4>${esc(suggestion.final_question||suggestion.suggested_question)}</h4><p>${esc(suggestion.rationale)}</p>${suggestion.evidence_gap?`<div class="evidence-gap"><small>待补证据</small>${esc(suggestion.evidence_gap)}</div>`:''}${tags(suggestion.expected_signals||[])}${suggestion.status==='pending'?`<div class="suggestion-actions"><button class="btn primary compact" data-live-decision="adopted" data-suggestion-id="${esc(suggestion.id)}">采用</button><button class="btn compact" data-live-decision="edited" data-suggestion-id="${esc(suggestion.id)}">编辑后采用</button><button class="btn compact" data-live-decision="skipped" data-suggestion-id="${esc(suggestion.id)}">跳过</button></div>`:`<small>处理结果：${esc(suggestion.status)}</small>`}` : '录入候选人回答后，AI 会准备一个有证据目标的问题。';
  }
  const competencies = state.session?.job?.competencies || [];
  if ($('live-review')) {
    $('live-confirm-all').disabled = !pendingCandidateSegments.length;
    $('live-confirm-merged').disabled = pendingCandidateSegments.length < 2;
    const evidence = state.session?.evidence || [];
    const coveredCompetencies = new Set(evidence.map(item => item.competency).filter(Boolean));
    const readiness = evidence.length >= 3 && coveredCompetencies.size >= 2
      ? '已满足招聘评价门槛'
      : `招聘评价仍需 ${Math.max(0, 3 - evidence.length)} 条证据、${Math.max(0, 2 - coveredCompetencies.size)} 个胜任力覆盖`;
    $('live-review').className = pendingCandidateSegments.length || confirmedRecords.length ? 'review-queue' : 'review-queue empty-state';
    const rollingHtml = live?.rolling_summary || live?.duplicate_segments_dropped ? `<div class="review-subtitle">长面试上下文</div><div class="review-claim summary"><div><small>摘要至 #${esc(live.summarized_until_sequence||0)} · 去重 ${esc(live.duplicate_segments_dropped||0)} 段</small><span>${esc(live.last_duplicate_reason || (live.rolling_summary ? live.rolling_summary.slice(-180) : '最近 12 段以内暂不需要滚动摘要'))}</span></div></div>` : '';
    const boundaryHtml = boundarySuggestions.length ? `<div class="review-subtitle">自动边界建议</div>${boundarySuggestions.map((item,index)=>`<div class="review-claim boundary"><div><small>${Math.round((item.confidence||0)*100)}% · ${esc(item.suggested_competency)}</small><span>${esc(item.reason)}（${(item.answer_segment_ids||[]).length} 段）</span><div class="boundary-factors">${(item.confidence_factors||[]).map(factor=>`<em>${esc(factor)}</em>`).join('')}</div></div><div class="claim-actions"><button type="button" data-live-boundary-index="${index}">按建议合并确认</button></div></div>`).join('')}` : '';
    const pendingHtml = pendingCandidateSegments.map(segment => `<div class="review-claim"><div><small>候选人回答 #${esc(segment.sequence)}</small><span>${esc(segment.text.slice(0, 180))}${segment.text.length > 180 ? '…' : ''}</span></div><div class="claim-actions"><button type="button" data-live-edit="${esc(segment.id)}">修改</button><button type="button" data-live-confirm="${esc(segment.id)}">确认证据</button></div></div>`).join('');
    const scoringLabel = record => record.scoring_status === 'scored' ? '已评分' : record.scoring_status === 'failed' ? `评分失败${record.scoring_error ? '：' + record.scoring_error : ''}` : '评分中…';
    const confirmedHtml = confirmedRecords.slice(-5).reverse().map(record => `<div class="review-claim confirmed"><div><small>已归档 · ${esc(record.competency)} · ${esc(scoringLabel(record))}</small><span>${esc(record.answer.slice(0, 160))}${record.answer.length > 160 ? '…' : ''}</span></div><div class="claim-actions"><button type="button" data-score-review="${esc(record.id)}">人工复核评分</button><button type="button" data-live-reevaluate="${esc(record.id)}">AI 重评</button><button type="button" data-live-revoke="${esc(record.id)}">撤销</button></div></div>`).join('');
    $('live-review').innerHTML = `<div class="review-summary"><strong>${pendingCandidateSegments.length} 条待确认</strong><span>${confirmedCount} 条已归档 · ${esc(readiness)}</span></div>${rollingHtml}${boundaryHtml}${pendingHtml || '<p class="review-more">暂无待确认候选人回答。</p>'}${confirmedHtml ? `<div class="review-subtitle">最近归档证据</div>${confirmedHtml}` : ''}`;
  }
  $('live-competencies').className = competencies.length ? 'progress-list' : 'progress-list empty-state';
  const guidance = live?.coverage_guidance || [];
  const priorityLabels = {high:'高优先级',medium:'继续补证',low:'基本覆盖'};
  const guidanceHtml = guidance.length ? `<div class="coverage-guidance">${guidance.map(item=>`<div class="coverage-card ${esc(item.priority)}"><small>${esc(priorityLabels[item.priority]||item.priority)} · ${esc(item.suggested_question_type)}</small><strong>${esc(item.competency)}</strong><span>${esc(item.evidence_count)} 条证据 · 最强信号 ${Math.round((item.strongest_confidence||0)*100)}%</span><p>${esc(item.reason)}</p><em>${esc(item.sample_question)}</em></div>`).join('')}</div>` : '';
  $('live-competencies').innerHTML = competencies.length ? competencies.map(name => {const evidence=(state.session?.evidence||[]).filter(item=>item.competency===name).length;return `<div class="progress-item"><span>${esc(name)}</span><div class="progress-track"><i style="width:${Math.min(100,evidence*34)}%"></i></div><b>${evidence}</b></div>`;}).join('') + guidanceHtml : '先完成岗位分析或面试设计。';
  if ($('live-question-usage')) {
    const usage = live?.question_usage || [];
    const usageLabels = {pending:'待问',suggested:'已建议',used:'已采用'};
    const counts = usage.reduce((acc,item)=>{acc[item.status]=(acc[item.status]||0)+1;return acc;},{});
    $('live-question-usage').className = usage.length ? 'usage-list' : 'usage-list empty-state';
    $('live-question-usage').innerHTML = usage.length ? `<div class="review-summary"><strong>${counts.pending||0} 道待问</strong><span>${counts.suggested||0} 道已建议 · ${counts.used||0} 道已采用</span></div>${usage.slice(0,12).map(item=>`<div class="usage-item ${esc(item.status)}"><small>${esc(usageLabels[item.status]||item.status)} · ${esc(item.round_name||'未分轮')}</small><strong>${esc(item.question)}</strong><span>${esc(item.competency)} · 建议 ${esc(item.suggested_count||0)} 次${item.last_suggestion_status?` · 最近 ${esc(item.last_suggestion_status)}`:''}</span></div>`).join('')}` : '生成面试蓝图后显示问题使用情况。';
  }
}
