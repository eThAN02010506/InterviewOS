const state = {
  sessionId: localStorage.getItem('interviewos.session') || '',
  role: localStorage.getItem('interviewos.role') || 'candidate',
  token: localStorage.getItem('interviewos.token') || '',
  session: null,
  view: '',
  settings: null
};
let mockRetry = false;
let mockRetryResponseId = '';
let mockQuestionAudioUrl = '';
let mockAnswerAudioUrl = '';
let mockAnswerAudioResponseId = '';
let mockPendingRecordingId = '';
let mockSpokenQuestionKey = '';
let mockCurrentSpeechKey = '';
let mockDisplayedResponseId = '';
let mockSpeechRequestSequence = 0;
let mockSpeechAbort = null;
let mockAnswerAudioRequestSequence = 0;
let mockAnswerAudioAbort = null;
let mockVoiceGeneration = 0;
let mockVoiceSessionId = '';
let activeLoadedSessionId = '';
let liveRecorder = null;
let liveAudioChunks = [];
let liveMediaStream = null;
let liveContinuousMode = false;
let liveContinuousQueue = [];
let liveContinuousUploading = false;
let liveContinuousChunkIndex = 0;
const ASR_PREVIEW_INTERVAL_MS = 2500;
let asrPreviewBusy = false;
let asrPreviewLastAt = 0;
let asrPreviewSequence = 0;
// VAD: silence-based utterance chunking for continuous listening.
const VAD_SPEECH_RMS = 0.02;        // RMS above this counts as speech
const VAD_SILENCE_MS = 600;         // silence of this length ends an utterance
const VAD_MIN_SPEECH_MS = 700;      // ignore brief noise/filler triggers
const VAD_GRACE_MS = 300;           // ignore speech briefly after a finalize
const VAD_TIMESLICE_MS = 250;       // MediaRecorder timeslice for continuous mode
const VAD_LEVEL_INTERVAL_MS = 100;  // indicator/level loop cadence
let liveVadAudioContext = null;
let liveVadAnalyser = null;
let liveVadData = null;
let liveVadState = 'idle';          // idle | in-speech | finalizing
let liveVadSpeechStart = 0;
let liveVadLastSpeech = 0;
let liveVadVoicedMs = 0;
let liveVadLastFinalizeAt = 0;
let liveVadUtteranceChunks = [];
let liveVadUtteranceBytes = 0;
let liveVadHeaderChunk = null;
let liveVadFinalizing = false;
let liveVadTimer = null;
// Whole-session recorder: captures the entire live interview as one audio file
// (independent of the per-utterance VAD path, which discards silence gaps).
let sessionRecorder = null;
let sessionChunks = [];
let sessionRecordingActive = false;
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const cssEscape = value => globalThis.CSS?.escape ? CSS.escape(String(value)) : String(value).replace(/["\\]/g, '\\$&');
const optional = id => $(id).value.trim() || null;
const safeUrl = value => { try { const url = new URL(value); return ['http:','https:'].includes(url.protocol) ? url.href : '#'; } catch { return '#'; } };
const formatDateTime = value => {
  if (!value) return '未知时间';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '未知时间' : date.toLocaleString();
};

function microphoneAvailabilityMessage() {
  if (!window.isSecureContext) return '当前是非安全的局域网 HTTP 页面，浏览器会阻止麦克风。请改用 HTTPS 地址后重试；文字输入仍可使用。';
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) return '当前浏览器不支持麦克风录制，请使用最新版 Chrome、Edge 或 Safari，或改用文字输入。';
  return '';
}

function microphoneConstraints() {
  return {audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1}};
}

function ensureMicrophoneAvailable() {
  const message = microphoneAvailabilityMessage();
  if (!message) return true;
  toast(message, true);
  return false;
}

function renderSecureContextWarning() {
  const node = $('secure-context-warning');
  if (!node) return;
  const message = !window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1'
    ? '局域网语音功能需要 HTTPS。当前页面仍可使用文字功能，但浏览器不会授予麦克风权限。'
    : '';
  node.textContent = message;
  node.classList.toggle('hidden', !message);
}

const microphoneActionIds = new Set(['mock-voice', 'live-record', 'live-dialogue', 'live-continuous']);
document.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button || !microphoneActionIds.has(button.id) || !microphoneAvailabilityMessage()) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  ensureMicrophoneAvailable();
}, true);

async function api(path, options = {}) {
  const authHeaders = state.token ? {Authorization: `Bearer ${state.token}`} : {};
  const headers = options.body instanceof FormData ? {...authHeaders, ...(options.headers || {})} : {'Content-Type':'application/json', ...authHeaders, ...(options.headers || {})};
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    // Login/register/logout 401s are expected (wrong password) and must not
    // clear the stored token; every other 401 means the session expired.
    const authEndpoint = /^\/api\/auth\/(login|register|logout)$/.test(path);
    if (response.status === 401 && !authEndpoint) { clearAuthenticatedState(); showLogin(); }
    throw new Error(data.detail || `请求失败 (${response.status})`);
  }
  return data;
}

function toast(message, error = false) {
  const node = $('toast'); node.textContent = message; node.className = error ? 'show error' : 'show';
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.className = '', 3200);
}

function busy(form, active) { form.classList.toggle('loading', active); form.setAttribute('aria-busy', active ? 'true' : 'false'); }
function showWorkflowStarting() {
  if (!state.session) return;
  state.session.strategy = {summary:'', key_risks:[], answer_framework:[], topics_to_emphasize:[], topics_to_avoid:[], likely_questions:[]};
  state.session.blueprint = {position:'', rounds:[]};
  state.session.mock_interview = {questions:[]};
  state.session.company.public_sources = [];
  state.session.workflow = {status:'running', current_step:'starting', completed_steps:0, total_steps:0, error:''};
  state.session.autopilot = {...(state.session.autopilot || {}), enabled:true, status:'running', phase:'intelligence', pause_reason:''};
  renderState();
}
function tags(items = []) { return `<div class="tag-list">${items.map(v => `<span class="tag">${esc(v)}</span>`).join('')}</div>`; }
function list(title, items = []) { return items.length ? `<div class="result-block"><h4>${esc(title)}</h4><ul>${items.map(v => `<li>${esc(v)}</li>`).join('')}</ul></div>` : ''; }

const navigation = {
  candidate: [
    ['candidate-home', '总览'], ['candidate', '面试准备'], ['mock', '模拟面试'], ['candidate-report', '改进报告']
  ],
  interviewer: [
    ['interviewer-home', '总览'], ['enterprise', '面试设计'], ['live', '实时辅助'], ['evaluation', '候选人评估']
  ]
};
const globalViews = new Set(['settings', 'debug']);
const viewMeta = {
  'candidate-home':['CANDIDATE WORKSPACE','候选人工作台'],
  candidate:['PERSONAL STRATEGY','面试准备'], mock:['PRACTICE & EVIDENCE','模拟面试'],
  'candidate-report':['GROWTH & REVIEW','个人改进报告'],
  'interviewer-home':['INTERVIEWER WORKSPACE','面试官工作台'],
  enterprise:['INTERVIEW ARCHITECTURE','面试设计'], live:['LIVE INTERVIEW COPILOT','实时面试辅助'], evaluation:['EVIDENCE REVIEW','候选人评估'],
  settings:['RUNTIME CONFIGURATION','模型与搜索'], debug:['LOCAL OBSERVABILITY','Debug Console']
};

function candidateHomeAction(session) {
  if (!session) return {view:'', label:'创建第一个会话'};
  if (session.mock_session?.status === 'completed') return {view:'candidate-report', label:'查看本轮改进报告'};
  if (session.mock_session?.status === 'active') return {view:'mock', label:'继续模拟面试'};
  if (session.strategy?.summary || session.mock_interview?.questions?.length) return {view:'mock', label:'开始模拟面试'};
  return {view:'candidate', label:'补充资料并生成策略'};
}

function renderNavigation() {
  $('nav').innerHTML = navigation[state.role].map(([view, label], index) =>
    `<button class="nav-item${state.view === view ? ' active' : ''}" data-view="${view}"><span>0${index + 1}</span>${label}</button>`
  ).join('');
  document.querySelectorAll('[data-role]').forEach(button => button.classList.toggle('active', button.dataset.role === state.role));
  document.querySelectorAll('[data-global-view]').forEach(button => button.classList.toggle('active', button.dataset.globalView === state.view));
}

function setRole(role) {
  if (!navigation[role]) return;
  state.role = role;
  localStorage.setItem('interviewos.role', role);
  setView(`${role}-home`);
}

function setView(view) {
  const allowed = navigation[state.role].some(([name]) => name === view) || globalViews.has(view);
  if (!allowed) view = `${state.role}-home`;
  if (view !== 'live' && liveSuggestionAbort) liveSuggestionAbort.abort();
  if (view !== 'mock') { cancelMockQuestionSpeech(); clearMockQuestionAudio(); }
  state.view = view;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${view}`));
  [$('view-eyebrow').textContent, $('view-title').textContent] = viewMeta[view];
  renderNavigation();
  location.hash = view;
  if (view === 'debug') refreshDebug();
  if (view === 'settings') loadSettings();
  if (view === 'mock') renderMock();
  if (view === 'live') renderLive();
}

async function checkHealth() {
  try { await api('/health'); $('health-dot').classList.add('ok'); $('health-label').textContent = '本地服务已连接'; }
  catch { $('health-dot').classList.remove('ok'); $('health-label').textContent = '本地服务不可用'; }
}

async function loadSessions() {
  if (!state.token) { showLogin(); return; }
  try {
    const sessions = await api('/api/interviews/sessions'); const select = $('session-select');
    select.innerHTML = '<option value="">选择会话</option>' + sessions.map(s => `<option value="${esc(s.id)}">${esc(s.candidate_name || '未命名')} · ${esc(s.job_title || '未指定岗位')}</option>`).join('');
    if (state.sessionId && sessions.some(s => s.id === state.sessionId)) select.value = state.sessionId;
    else if (sessions[0]) select.value = state.sessionId = sessions[0].id;
    if (state.sessionId) await loadSession();
  } catch (error) { toast(error.message, true); }
}

function showLogin() {
  const dialog = $('login-dialog'); if (!dialog) return;
  dialog.showModal();
}

function hideLogin() { $('login-dialog')?.close(); }

function clearAuthenticatedState() {
  resetMockAudioExperience();
  activeLoadedSessionId = '';
  state.token = ''; state.sessionId = ''; state.session = null;
  localStorage.removeItem('interviewos.token');
  localStorage.removeItem('interviewos.session');
  const select = $('session-select');
  if (select) select.innerHTML = '<option value="">选择会话</option>';
  $('user-chip').textContent = '';
  renderAll();
}

function logout() {
  if (state.token) { api('/api/auth/logout', {method: 'POST'}).catch(() => {}); }
  clearAuthenticatedState();
  showLogin();
}

$('logout-btn').onclick = logout;
$('close-login-dialog').onclick = () => hideLogin();
let loginMode = 'login';
$('login-toggle').onclick = () => {
  loginMode = loginMode === 'login' ? 'register' : 'login';
  $('login-title').textContent = loginMode === 'login' ? '登录' : '注册账号';
  $('login-submit').textContent = loginMode === 'login' ? '登录' : '注册';
  $('login-toggle').textContent = loginMode === 'login' ? '注册账号' : '返回登录';
};
$('login-form').onsubmit = async e => {
  e.preventDefault();
  const username = $('login-username').value.trim();
  const password = $('login-password').value;
  if (!username || !password) return;
  const button = $('login-submit'); busy(e.currentTarget, true);
  try {
    const data = await api(`/api/auth/${loginMode}`, {method: 'POST', body: JSON.stringify({username, password})});
    state.token = data.token; localStorage.setItem('interviewos.token', data.token);
    $('user-chip').textContent = data.username;
    $('login-username').value = ''; $('login-password').value = '';
    hideLogin(); await loadSessions();
    toast(loginMode === 'register' ? '账号已创建并登录' : '已登录');
  } catch (error) { toast(error.message, true); }
  finally { busy(e.currentTarget, false); }
};

async function loadSession() {
  if (activeLoadedSessionId !== state.sessionId) {
    resetMockAudioExperience();
    // Never render the previous candidate under a newly selected session ID
    // while the replacement request is pending or if it fails.
    state.session = null;
    activeLoadedSessionId = '';
  }
  if (!state.sessionId) { activeLoadedSessionId = ''; state.session = null; renderState(); return; }
  const requestedSessionId = state.sessionId;
  try { const data = await api(`/api/interviews/sessions/${requestedSessionId}`); if (state.sessionId !== requestedSessionId) return; state.session = data.state; activeLoadedSessionId = requestedSessionId; localStorage.setItem('interviewos.session', requestedSessionId); hydrateSessionForms(); renderState(); }
  catch (error) { if (state.sessionId === requestedSessionId) { state.session = null; renderState(); } toast(error.message, true); }
}

function hydrateSessionForms() {
  const s = state.session; if (!s) return;
  const values = {
    'candidate-resume': s.candidate?.raw_resume_text,
    'enterprise-resume': s.candidate?.raw_resume_text,
    'candidate-jd': s.job?.raw_description || s.job?.title,
    'enterprise-jd': s.job?.raw_description || s.job?.title,
    'candidate-company': s.company?.name,
    'enterprise-company': s.company?.name,
    'interviewer-name': s.interviewer?.name,
    'interviewer-position': s.interviewer?.position,
  };
  Object.entries(values).forEach(([id, value]) => { if ($(id) && value) $(id).value = value; });
}

function renderState() {
  const s = state.session;
  renderSecureContextWarning();
  $('metric-workflow').textContent = s?.workflow?.status || '未开始';
  $('metric-step').textContent = s?.autopilot?.enabled ? `AI · ${s.autopilot.phase || s.autopilot.status}` : (s?.workflow?.current_step || '等待输入资料');
  const sources = (s?.company?.public_sources?.length || 0) + (s?.interviewer?.public_expressions?.length || 0);
  $('metric-sources').textContent = sources; $('metric-questions').textContent = s?.mock_interview?.questions?.length || 0; $('metric-evidence').textContent = s?.evidence?.length || 0;
  $('next-action').textContent = s?.next_action || '创建一个会话，然后选择候选人准备或企业面试设计。';
  const homeAction=candidateHomeAction(s); const homeButton=$('candidate-next-action'); homeButton.textContent=homeAction.label; if(homeAction.view){homeButton.dataset.route=homeAction.view;delete homeButton.dataset.newPractice;}else{delete homeButton.dataset.route;homeButton.dataset.newPractice='true';}
  renderResumeReview(); renderJDReview(); renderFacts(); maybePromptEntityResolution();
  const profiles = [['候选人', !!s?.candidate?.skills?.length],['岗位',!!s?.job?.competencies?.length],['公司',!!s?.company?.dna],['面试官',!!s?.interviewer?.name]];
  $('profile-progress').innerHTML = profiles.map(([name, done]) => `<div class="progress-item"><span>${name}</span><div class="progress-track"><i style="width:${done?100:8}%"></i></div><b>${done?'完成':'待分析'}</b></div>`).join('');
  $('hiring-candidate').textContent = s?.candidate?.name || '待分析';
  $('hiring-competencies').textContent = s?.job?.competencies?.length || 0;
  $('hiring-rounds').textContent = s?.blueprint?.rounds?.length || 0;
  $('hiring-evidence').textContent = s?.evidence?.length || 0;
  $('hiring-progress').innerHTML = (s?.job?.competencies || []).map(name => {
    const score = Math.round((s?.evaluated_competencies?.[name] || 0) * 100);
    return `<div class="progress-item"><span>${esc(name)}</span><div class="progress-track"><i style="width:${Math.max(score, 5)}%"></i></div><b>${score || '—'}</b></div>`;
  }).join('') || '<div class="empty-state">完成岗位分析后显示。</div>';
  renderStrategy(); renderBlueprint(); renderSources(); renderMock(); renderReports(); renderLive();
}

function renderAll() { renderState(); }

function renderLive() {
  const live = state.session?.live_interview;
  if (!$('live-transcript')) return;
  const statusLabels = {idle:'尚未开始',active:'监听中',paused:'已暂停',completed:'已结束'};
  const status = live?.status || 'idle';
  $('live-status-label').textContent = statusLabels[status] || status;
  $('live-status-dot').className = status === 'active' ? 'active' : status;
  $('live-consent-panel').classList.toggle('hidden', status !== 'idle' && status !== 'completed');
  $('live-pause').textContent = status === 'paused' ? '恢复' : '暂停';
  $('live-pause').disabled = !['active','paused'].includes(status);
  $('live-finish').disabled = !['active','paused'].includes(status) || liveContinuousMode || liveContinuousQueue.length > 0 || liveContinuousUploading;
  const microphoneUnavailable = !!microphoneAvailabilityMessage();
  $('live-record').disabled = status !== 'active' || !!liveRecorder || microphoneUnavailable;
  $('live-dialogue').disabled = status !== 'active' || !!liveRecorder || microphoneUnavailable;
  $('live-continuous').disabled = status !== 'active' || !!liveRecorder || microphoneUnavailable;
  $('live-stop-continuous').disabled = !liveContinuousMode;
  $('live-plan').disabled = status !== 'active';
  $('live-stream-plan').disabled = status !== 'active' || !!liveSuggestionStream;
  if (liveContinuousMode || liveContinuousQueue.length || liveContinuousUploading) {
    if (liveContinuousMode) {
      updateLiveVadNote();
    } else {
      const uploading = liveContinuousUploading ? '，正在转写 1 段' : '';
      $('live-recording-note').textContent = `连续监听已停止：队列 ${liveContinuousQueue.length} 段${uploading}。分段会先标记为“待确认”，请审阅说话人与文本后再归档证据。`;
    }
  }
  const sessionRecording = $('live-session-recording');
  if (sessionRecording) {
    const audioFile = live?.audio_file;
    const recordingStatus = $('session-recording-status');
    const downloadBtn = $('session-audio-download');
    const isError = recordingStatus?.classList.contains('error');
    if (audioFile && state.sessionId) {
      sessionRecording.hidden = false;
      recordingStatus.textContent = '全场录音已保存';
      recordingStatus.classList.remove('error');
      downloadBtn.hidden = false;
    } else if (sessionRecordingActive) {
      sessionRecording.hidden = false;
      recordingStatus.textContent = '全场录音中…';
      recordingStatus.classList.remove('error');
      downloadBtn.hidden = true;
    } else if (isError) {
      sessionRecording.hidden = false;
      downloadBtn.hidden = true;
    } else {
      sessionRecording.hidden = true;
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
  if (!liveSuggestionStream) {
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

function nearestLiveQuestionSegmentId(answerSegmentId) {
  const segments = state.session?.live_interview?.segments || [];
  const index = segments.findIndex(segment => segment.id === answerSegmentId);
  if (index < 0) return null;
  for (let i = index - 1; i >= 0; i -= 1) {
    if (segments[i].speaker === 'interviewer') return segments[i].id;
  }
  return null;
}

function spotlight(node, message = '') {
  if (!node) return false;
  node.scrollIntoView({behavior:'smooth', block:'center'});
  node.classList.add('spotlight');
  setTimeout(() => node.classList.remove('spotlight'), 1800);
  if (message) toast(message);
  return true;
}

function firstPendingCandidateSegment() {
  const recordedSegmentIds = new Set((state.session?.live_interview_records || []).flatMap(record => record.transcript_segment_ids || []));
  return (state.session?.live_interview?.segments || []).find(segment => segment.speaker === 'candidate' && !recordedSegmentIds.has(segment.id));
}

function strongestBoundarySuggestion() {
  return [...(state.session?.live_interview?.answer_boundary_suggestions || [])].sort((a,b)=>(b.confidence||0)-(a.confidence||0))[0];
}

function renderResumeReview() {
  const review = state.session?.resume_review;
  const targets = [$('candidate-resume-review'), $('enterprise-resume-review')];
  targets.forEach(node => {
    if (!review?.metadata?.filename) { node.innerHTML = ''; node.classList.remove('visible'); return; }
    const claims = review.claims || [];
    const unresolved = claims.filter(claim => claim.status === 'unverified');
    const reviewed = claims.filter(claim => claim.status !== 'unverified');
    const statusLabels = {
      confirmed:'本人确认 · 未外部核验', modified:'已修订确认 · 未外部核验',
      needs_documents:'待补充材料', disputed:'存在争议', ignored:'已忽略'
    };
    const pendingClaim = claim => `<div class="review-claim"><div><small>${esc(claim.category)} · 简历自述 · 待核验</small><span>${esc(claim.statement)}</span></div><div class="claim-actions"><button type="button" data-claim-action="confirmed" data-claim-id="${esc(claim.id)}">本人确认</button><button type="button" data-claim-action="modified" data-claim-id="${esc(claim.id)}">修改并确认</button><button type="button" data-claim-action="needs_documents" data-claim-id="${esc(claim.id)}">要求材料</button><button type="button" data-claim-action="ignored" data-claim-id="${esc(claim.id)}">忽略</button></div></div>`;
    const reviewedClaim = claim => `<div class="review-claim reviewed"><div><small>${esc(claim.category)} · ${esc(statusLabels[claim.status] || claim.status)}</small><span>${esc(claim.statement)}</span>${claim.note ? `<em>${esc(claim.note)}</em>` : ''}</div><div class="claim-actions"><button type="button" data-claim-action="unverified" data-claim-id="${esc(claim.id)}">恢复待核验</button></div></div>`;
    node.classList.add('visible');
    const structuredHtml = (review.structured_by === 'llm' && review.structured?.length)
      ? `<div class="review-subtitle">AI 结构化板块（${esc(review.structured_by)}）</div>${review.structured.map(s => `<div class="review-claim structured"><div><small>${esc(s.category)}${s.date_range ? ` · ${esc(s.date_range)}` : ''}</small><strong>${esc(s.institution)}${s.title ? ` — ${esc(s.title)}` : ''}</strong>${s.description ? `<span>${esc(s.description)}</span>` : ''}</div></div>`).join('')}` : '';
    node.innerHTML = `<div class="review-summary"><strong>${esc(review.metadata.filename)}</strong><span>${review.metadata.character_count} 字 · ${review.issues.length} 项提示 · ${unresolved.length} 项待确认</span></div>
      ${structuredHtml}
      ${(review.issues || []).map(issue => `<div class="review-issue ${esc(issue.severity)}"><b>${esc(issue.severity === 'warning' ? '请检查' : '提示')}</b><span>${esc(issue.message)}</span></div>`).join('')}
      ${unresolved.slice(0, 8).map(pendingClaim).join('')}
      ${unresolved.length > 8 ? `<p class="review-more">继续处理后将自动显示剩余 ${unresolved.length - 8} 项。</p>` : ''}
      ${reviewed.length ? `<div class="review-subtitle">已处理（不等于外部真实性验证）</div>${reviewed.map(reviewedClaim).join('')}` : ''}`;
  });
}

function renderJDReview() {
  const review=state.session?.job_review;
  [$('candidate-jd-review'),$('enterprise-jd-review')].forEach(node=>{
    if(!review?.requirements?.length && !review?.warnings?.length){node.innerHTML='';return;}
    const requirements=review.requirements||[];
    const explicit=requirements.filter(x=>x.origin==='explicit');
    const inferred=requirements.map((item,index)=>({...item,index})).filter(x=>x.origin==='inferred');
    const sources=review.public_sources||[];const researchLabel=review.public_research_status==='completed'?`已检索 ${sources.length} 个公开 JD 来源`:review.public_research_status==='no_reliable_sources'?'未找到可靠公开 JD':review.public_research_status==='not_configured'?'未配置公开搜索':review.public_research_status==='consent_required'?'需允许公开检索':'已完成结构检查';
    const sourcesHtml=sources.length?`<div class="review-subtitle">公开 JD 候选来源</div>${sources.map(source=>`<div class="review-claim"><div><small>${esc(source.source_quality||'未评级')}</small><span>${esc(source.title||source.url)}</span></div><div class="claim-actions"><a href="${esc(safeUrl(source.url))}" target="_blank" rel="noreferrer">查看来源</a></div></div>`).join('')}`:'';
    node.innerHTML=`<div class="review-summary"><strong>JD 完整度 ${Math.round((review.completeness_score||0)*100)}%</strong><span>${review.is_title_only?researchLabel:'已完成结构检查'}</span></div>${(review.warnings||[]).map(item=>`<p class="jd-warning">${esc(item)}</p>`).join('')}${review.missing_sections?.length?`<p class="jd-warning">请补充：${esc(review.missing_sections.join('、'))}</p>`:''}${sourcesHtml}${list('明确要求',explicit.map(x=>x.text))}${inferred.length?`<div class="review-subtitle">公开来源/AI 推测（需确认）</div>${inferred.map(item=>`<div class="review-claim"><div><small>推测要求 #${item.index+1}</small><span>${esc(item.text)}</span></div><div class="claim-actions"><button type="button" data-jd-action="confirm" data-jd-index="${item.index}">确认</button><button type="button" data-jd-action="edit" data-jd-index="${item.index}">编辑</button><button type="button" data-jd-action="delete" data-jd-index="${item.index}">删除</button></div></div>`).join('')}`:''}`;
  });
}

function renderFacts(){
  const node=$('fact-result'); const cards=state.session?.fact_cards||[];
  if(!cards.length){node.className='fact-list empty-state';node.textContent='尚未形成事实卡。';return;}
  const labels={verified:'已验证',inferred:'推测',conflict:'冲突',accepted:'已确认',rejected:'已排除'};
  const categoryLabels={company:'公司',interviewer:'面试官',technology:'技术方向',public_opinion:'公开观点',past_employer:'过往雇主业务背景（不代表候选人经历）'};
  const qualityLabels={official:'官方来源',high:'高可信来源',secondary:'二级来源',unrated:'未评级来源'};
  const grouped=cards.reduce((acc,card)=>{const key=card.status==='conflict'||card.status==='rejected'?card.status:card.category;(acc[key] ||= []).push(card);return acc;},{});
  const order=['conflict','company','interviewer','past_employer','technology','public_opinion','rejected'];
  const groups=[...order.filter(key=>grouped[key]),...Object.keys(grouped).filter(key=>!order.includes(key))];
  node.className='fact-list';node.innerHTML=groups.map(key=>`<section class="fact-group"><div class="fact-group-head"><strong>${esc(key==='conflict'?'冲突信息':key==='rejected'?'已排除信息':categoryLabels[key]||key)}</strong><span>${grouped[key].length} 条</span></div>${grouped[key].map(card=>`<article class="fact-card ${esc(card.status)}"><div><span>${esc(card.category)}</span><b>${esc(labels[card.status]||card.status)}</b></div><strong>${esc(card.subject)}</strong><p>${esc(card.claim)}</p><small>${esc(card.note||'')} · 置信度 ${Math.round((card.confidence||0)*100)}% · ${card.source_count||((card.source_urls||[]).length)} 个来源 · ${esc(qualityLabels[card.source_quality]||card.source_quality||'未评级来源')} · ${card.cache_hit?'含缓存':'实时/新鲜来源'} · 抓取 ${esc(formatDateTime(card.source_fetched_at))} · 生成 ${esc(formatDateTime(card.generated_at))}</small>${card.source_filter_reason?`<small>筛选：${esc(card.source_filter_reason)}</small>`:''}${(card.source_urls||[]).map((url,i)=>`<a href="${esc(safeUrl(url))}" target="_blank" rel="noreferrer">来源 ${i+1}</a>`).join('')}<div class="claim-actions fact-actions"><button type="button" data-fact-action="accept" data-fact-id="${esc(card.id)}">确认使用</button><button type="button" data-fact-action="reject" data-fact-id="${esc(card.id)}">排除</button><button type="button" data-fact-action="reset" data-fact-id="${esc(card.id)}">恢复待审</button></div></article>`).join('')}</section>`).join('');
}

function maybePromptEntityResolution(){
  const dialog=$('entity-dialog'); if(dialog.open)return;
  const item=(state.session?.entity_resolutions||[]).find(x=>x.status==='pending'); if(!item)return;
  dialog.dataset.resolutionId=item.id;$('entity-copy').textContent=`搜索结果显示“${item.proposed_name}”可能是“${item.input_name}”的正确实体。是否将“${item.input_name}”更正为“${item.proposed_name}”？`;
  $('entity-name').value=item.proposed_name||item.input_name||'';
  dialog.showModal();
}

function renderReports() {
  const s = state.session;
  const evidence = s?.evidence || [];
  const evaluation = s?.evaluation;
  const feedback = s?.feedback;
  const score = evaluation?.finalized_at ? Math.round(evaluation.overall_score * 100) : (evidence.length ? Math.round(evidence.reduce((sum, item) => sum + item.confidence, 0) / evidence.length * 100) : null);
  $('report-score').textContent = score ?? '—';
  $('evaluation-score').textContent = score ?? '—';
  const recommendationLabels = {strong_hire:'强烈建议录用',hire:'建议录用',lean_hire:'倾向录用',lean_no_hire:'倾向不录用',no_hire:'不建议录用',insufficient_evidence:'证据不足'};
  $('recommendation-label').textContent = recommendationLabels[evaluation?.recommendation] || '待评估';
  const competencyItems = evaluation?.competencies?.length ? evaluation.competencies : evidence;
  const narrativeLabel=evaluation?.narrative_source==='model'?'AI 按证据框架生成 · 分数与缺口由规则锁定':'确定性证据聚合';
  const narrativeHeader=evaluation?.summary?`<div class="evaluation-narrative"><small>${esc(narrativeLabel)}</small><p>${esc(evaluation.summary)}</p></div>`:'';
  const evidenceHtml = narrativeHeader+competencyItems.map(item => `<div class="evidence-row"><div><strong>${esc(item.competency)}</strong><small>${esc((item.supporting_evidence || [item.signal]).filter(Boolean).join(' · '))}</small>${item.assessment?`<p class="competency-assessment">${esc(item.assessment)}${item.narrative_evidence_ids?.length?` <small>引用 ${item.narrative_evidence_ids.length} 条本能力证据</small>`:''}</p>`:''}${item.gaps?.length?`<em>缺口：${esc(item.gaps.join(' · '))}</em>`:''}${item.next_probe?`<p class="next-probe"><b>建议追问</b>${esc(item.next_probe)}</p>`:''}</div><b>${Math.round((item.score ?? item.confidence) * 100)}</b></div>`).join('');
  $('candidate-evidence').className = evidence.length ? '' : 'empty-state';
  $('candidate-evidence').innerHTML = evidenceHtml || '完成模拟面试后生成。';
  $('evaluation-evidence').className = evidence.length ? '' : 'empty-state';
  $('evaluation-evidence').innerHTML = evidenceHtml || '尚无面试证据。';
  const responses = s?.mock_session?.responses || [];
  const gaps = [...new Set(responses.flatMap(item => item.evaluation?.missing_signals || []))];
  $('candidate-priorities').innerHTML = feedback?.overall ? `<p>${esc(feedback.overall)}</p>${list('优先改进', feedback.improvements)}${list('行动计划', feedback.action_plan)}` : (gaps.length ? list('需要补强', gaps) : '<div class="empty-state">尚无足够回答数据。</div>');
  const missing = evaluation?.competencies?.flatMap(item => item.gaps || []) || s?.missing_signals || [];
  $('evaluation-gaps').innerHTML = feedback?.recommendation_reasoning ? `<p>${esc(feedback.recommendation_reasoning)}</p>${list('面试官备注', feedback.interviewer_notes)}${list('仍缺信号', [...new Set(missing)])}` : (missing.length ? list('尚缺证据', missing) : '<div class="empty-state">当前没有标记的信号缺口。</div>');
}

function renderStrategy() {
  const strategy = state.session?.strategy; const node = $('strategy-result');
  if (!strategy?.summary && !strategy?.key_risks?.length) { node.className='empty-state'; node.textContent='完成分析后，这里会显示风险、回答框架和重点话题。'; return; }
  node.className=''; node.innerHTML=`<p>${esc(strategy.summary)}</p>${list('关键风险',strategy.key_risks)}${list('回答框架',strategy.answer_framework)}${list('重点强调',strategy.topics_to_emphasize)}${list('可能问题',strategy.likely_questions)}`;
}

function renderSources() {
  const s=state.session; const sources=[...(s?.company?.public_sources||[]),...(s?.interviewer?.public_expressions||[]),...(s?.past_employer_sources||[])]; const node=$('source-result');
  if (!sources.length) {
    const statuses=[s?.company?.public_research_status,s?.interviewer?.public_research_status,s?.past_employer_research_status];
    node.className='source-list empty-state';
    node.textContent=statuses.includes('no_reliable_sources')?'已检索，但没有找到可可靠归属于该公司或人物的公开资料。':statuses.includes('failed')?'公开检索失败，请检查搜索设置后重试。':'尚未检索。';
    return;
  }
  const qualityLabels={official:'官方',high:'高可信',secondary:'二手来源',unrated:'未评级'};
  node.className='source-list'; node.innerHTML=sources.map(x=>{const alias=x.identity_match==='corroborated_alias'?`名称近似匹配：输入“${x.input_identity}”，来源“${x.matched_identity}” · `:'';return `<a href="${esc(safeUrl(x.url))}" target="_blank" rel="noreferrer"><strong>${esc(x.title||x.url||'公开资料')}</strong><small>${esc(alias)}${esc(qualityLabels[x.source_quality]||'未评级')} · ${x.cache_hit?'缓存命中':'新请求'} · 抓取 ${esc(formatDateTime(x.fetched_at))} · ${esc(x.filter_reason||x.source_quality_reason||'待复核来源')}</small><small>${esc((x.snippet||x.text||'').slice(0,150))}</small></a>`}).join('');
}

function renderBlueprint() {
  const blueprint=state.session?.blueprint; const node=$('blueprint-result');
  if (!blueprint?.rounds?.length) { node.className='empty-state'; node.textContent='工作流完成后显示每轮目标、问题和强信号。'; return; }
  node.className=''; node.innerHTML=blueprint.rounds.map((round,i)=>`<div class="round-card"><p class="eyebrow">ROUND ${i+1}</p><h4>${esc(round.name)}</h4><p>${esc(round.goal)}</p>${tags(round.evaluation_criteria)}${(round.questions||[]).map(q=>`<div class="question-item"><strong>${esc(q.question)}</strong><small>${esc(q.competency)}</small></div>`).join('')}</div>`).join('');
}

function renderSpokenAnswerAnalysis(evaluation) {
  const analysis=evaluation?.spoken_analysis;
  if(!analysis?.question_coverage?.length&&!analysis?.semantic_steps?.length)return '';
  const typeLabels={behavioral_example:'\u884c\u4e3a\u6848\u4f8b',situational:'\u60c5\u666f\u63a8\u6f14',motivation:'\u52a8\u673a\u5339\u914d',methodology:'\u65b9\u6cd5\u8bba',general:'\u7efc\u5408\u56de\u7b54'};
  const steps=(analysis.semantic_steps||[]).map((item,index)=>`<li><strong>${index+1}. ${esc(item.label)}</strong><span>${esc(item.evidence)}</span></li>`).join('');
  const coverage=(analysis.question_coverage||[]).map(item=>`<div class="review-claim"><div><small>${esc({covered:'\u5df2\u8986\u76d6',partial:'\u90e8\u5206\u8986\u76d6',missing:'\u672a\u8986\u76d6'}[item.status]||item.status)}</small><span>${esc(item.requirement)}</span><small>${esc(item.evidence||item.suggestion)}</small></div></div>`).join('');
  const cleaned=analysis.cleaned_transcript&&analysis.cleaned_transcript!==analysis.raw_transcript?`<div class="result-block"><h4>\u6e05\u6d17\u540e\u8bed\u4e49\u7a3f\uff08\u539f\u8f6c\u5199\u4ecd\u4fdd\u7559\uff09</h4><p>${esc(analysis.cleaned_transcript)}</p></div>`:'';
  const calibrated=(analysis.calibration_notes||[]).length?list('\u8bc1\u636e\u4e00\u81f4\u6027\u6821\u51c6',analysis.calibration_notes):'';
  return `<div class="result-block"><h4>\u56de\u7b54\u7c7b\u578b\uff1a${esc(typeLabels[analysis.answer_type]||analysis.answer_type||'\u7efc\u5408\u56de\u7b54')}</h4>${steps?`<ol class="semantic-steps">${steps}</ol>`:''}</div>${cleaned}${coverage?`<div class="result-block"><h4>\u95ee\u9898\u8981\u6c42\u8986\u76d6</h4>${coverage}</div>`:''}${calibrated}`;
}

function renderQuestionAlignment(evaluation) {
  const coverage=evaluation?.spoken_analysis?.question_coverage||[];
  if(!coverage.length)return '';
  const labels={covered:'已回答',partial:'回答了一部分',missing:'没有回答'};
  return `<section class="alignment-feedback"><div class="review-subtitle">这道题问了什么，你实际回答了什么</div>${coverage.map(item=>`<article class="alignment-row ${esc(item.status)}"><div><strong>${esc(item.requirement)}</strong><b>${esc(labels[item.status]||item.status)}</b></div><p>${item.evidence?`你的原话：“${esc(item.evidence)}”`:'你的回答中没有找到对应内容。'}</p>${item.status!=='covered'?`<small>下次这样补：${esc(item.suggestion)}</small>`:''}</article>`).join('')}</section>`;
}

function renderRetryComparison(current, previous) {
  if(!current||!previous)return '';
  const dimensions=[['证据','content'],['深度','technical_depth'],['结构','structure'],['结果','impact']];
  const deltas=dimensions.map(([label,key])=>{const before=Math.round((previous.evaluation?.[key]||0)*100),after=Math.round((current.evaluation?.[key]||0)*100),delta=after-before;return `<div class="score"><span>${label}</span><strong>${delta>0?'+':''}${delta}</strong><small>${before} → ${after}</small></div>`}).join('');
  const beforeCoverage=new Map((previous.evaluation?.spoken_analysis?.question_coverage||[]).map(item=>[item.requirement,item.status]));
  const improved=(current.evaluation?.spoken_analysis?.question_coverage||[]).filter(item=>item.status==='covered'&&beforeCoverage.get(item.requirement)!=='covered').map(item=>item.requirement);
  return `<div class="result-block retry-comparison"><h4>本次重答对比</h4><div class="score-grid">${deltas}</div>${improved.length?list('新补齐的要求',improved):'<small>本次尚未新增完整覆盖项，可继续针对首要缺口练习。</small>'}<div class="claim-actions">${previous.audio_file?`<button type="button" data-play-mock-audio="${esc(previous.id)}">播放上一版</button><button type="button" data-delete-mock-audio="${esc(previous.id)}">删除上一版录音</button>`:''}</div></div>`;
}

function renderMock() {
  const plan=state.session?.mock_interview?.questions||[]; const session=state.session?.mock_session; const index=session?.current_question_index||0;
  $('mock-progress').textContent=''; $('mock-status').textContent=session?.status==='active'?'面试进行中':session?.status==='evaluating'?'正在生成报告':session?.status==='completed'?'本轮已完成':'准备开始';
  const current=session?.status==='active'?plan[index]:null; const questionText=session?.pending_follow_up||current?.question; $('start-mock').style.display=session?.status==='idle'||!session?'inline-block':'none';
  const framework=$('mock-framework'); const frameworkText=$('mock-framework-text');
  const currentResponses = current ? [...(session?.responses||[])].reverse().filter(response => response.question_id === current.id) : [];
  const currentResponse = session?.pending_follow_up ? currentResponses.find(response => response.question === questionText) : currentResponses[0];
  const mainResponse = currentResponses.find(response => !response.is_follow_up);
  let retryResponse = mockRetry ? currentResponses.find(response => response.id === mockRetryResponseId) : null;
  if (mockRetry && !retryResponse) { mockRetry=false; mockRetryResponseId=''; retryResponse=null; }
  const displayResponse = retryResponse || currentResponse;
  const justAnswered = !!displayResponse;
  const displayedQuestionText = justAnswered ? displayResponse.question : questionText;
  const displayedAsFollowUp = justAnswered ? displayResponse.is_follow_up : !!session?.pending_follow_up;
  const currentAnswered = !!currentResponse;
  const isRetrying = !!(mockRetry && retryResponse);
  $('answer-form').style.display = current && (!justAnswered || isRetrying) ? 'block' : 'none';
  if (framework) { const hasFw = current && current.answer_framework && (!justAnswered || (isRetrying && !retryResponse.is_follow_up)); framework.classList.toggle('hidden', !hasFw); if (hasFw) frameworkText.textContent = current.answer_framework; }
  const requirements=$('mock-requirements');const showRequirements=current&&!displayedAsFollowUp&&(current.question_requirements||[]).length;if(requirements){requirements.classList.toggle('hidden',!showRequirements);requirements.innerHTML=showRequirements?`<small>这道题需要回答</small>${tags(current.question_requirements)}`:'';}
  const example=$('mock-example');const showExample=current&&!displayedAsFollowUp&&current.example_answer&&(!justAnswered||isRetrying);if(example){example.classList.toggle('hidden',!showExample);if(showExample){$('mock-example-note').textContent=current.example_answer_note||'教学示例为虚构场景，请替换为你的真实经历。';$('mock-example-text').textContent=current.example_answer;}}
  const completed=session?.status==='completed'; const responseCount=session?.responses?.length||0; const scoredResponses=(session?.responses||[]).filter(item=>item.evaluation); const averageScore=scoredResponses.length?Math.round(scoredResponses.reduce((sum,item)=>sum+((item.evaluation.content+item.evaluation.technical_depth+item.evaluation.structure+item.evaluation.impact)/4),0)/scoredResponses.length*100):0;
  $('mock-question').className=current?'question-copy':completed?'question-copy completion-state':'question-copy empty-state'; $('mock-question').innerHTML=current?`<small>${displayedAsFollowUp?'证据追问':esc(current.competency||'综合能力')}</small>${esc(displayedQuestionText)}`:completed?`<small>练习已保存</small><strong>本轮完成 ${responseCount} 次回答${averageScore?` · 平均 ${averageScore} 分`:''}</strong><p>先查看改进报告梳理共性问题；需要练习另一岗位或候选人时，新建会话可避免证据混用。</p><div class="completion-actions"><button class="btn primary" type="button" data-route="candidate-report">查看改进报告</button><button class="btn" type="button" data-new-practice>新建练习会话</button></div>`:'先完成候选人准备工作流，生成个性化问题。';
  const questionAudio=$('mock-speak-question')?.closest('.question-audio-controls'); if(questionAudio)questionAudio.classList.toggle('hidden',!current);
  const speakButton=$('mock-speak-question');if(speakButton)speakButton.disabled=!current;
  const voiceButton=$('mock-voice');if(voiceButton){voiceButton.disabled=!current||!!microphoneAvailabilityMessage();voiceButton.title=microphoneAvailabilityMessage();}
  const speechKey=current?`${current.id}:${displayedQuestionText}`:'';
  if(speechKey!==mockCurrentSpeechKey){cancelMockQuestionSpeech();clearMockQuestionAudio();clearMockAnswerExperience();mockCurrentSpeechKey=speechKey;}
  mockDisplayedResponseId=displayResponse?.id||'';
  syncMockAnswerAudio(displayResponse);
  const mockViewVisible=state.view==='mock'&&document.visibilityState==='visible'&&$('view-mock')?.classList.contains('active');
  if(current&&session?.status==='active'&&mockViewVisible&&$('mock-auto-speak')?.checked&&speechKey!==mockSpokenQuestionKey){mockSpokenQuestionKey=speechKey;setTimeout(()=>speakCurrentMockQuestion(true),0);}
  const actions=$('mock-actions');
  if (actions) {
    // 上一题/下一题/结束 are always available during an active session so the
    // user can navigate freely; 重新来 only appears once the current question
    // has been answered.
    const finishBtn=$('mock-finish'); const nextBtn=$('mock-next'); const prevBtn=$('mock-prev'); const retryBtn=$('mock-retry'); const retryMainBtn=$('mock-retry-main'); const reviewBtn=$('mock-review-score');
    const isActive = session?.status==='active';
    if (finishBtn) finishBtn.style.display = isActive ? 'inline-block' : 'none';
    if (nextBtn) nextBtn.style.display = isActive ? 'inline-block' : 'none';
    if (prevBtn) prevBtn.style.display = (isActive && index > 0) ? 'inline-block' : 'none';
    if (retryMainBtn) { retryMainBtn.style.display = (isActive && mainResponse && (session?.pending_follow_up || currentResponse?.is_follow_up)) ? 'inline-block' : 'none'; retryMainBtn.dataset.responseId = mainResponse?.id || ''; }
    if (retryBtn) { retryBtn.style.display = (isActive && currentAnswered) ? 'inline-block' : 'none'; retryBtn.textContent = currentResponse?.is_follow_up ? '重答当前追问' : '重新来'; retryBtn.dataset.responseId = currentResponse?.id || ''; }
    if (reviewBtn) { reviewBtn.style.display = (isActive && currentAnswered) ? 'inline-block' : 'none'; reviewBtn.dataset.scoreReview = currentResponse?.id || ''; }
    actions.classList.toggle('hidden', !isActive || isRetrying);
    if (isActive && retryBtn) retryBtn.disabled = false;
  }
  const last=displayResponse||(completed?(session?.responses||[]).at(-1):null); const node=$('coach-result'); $('coach-title').textContent=completed&&last?'最后一题反馈':'四维评价'; if(completed&&last){mockDisplayedResponseId=last.id;syncMockAnswerAudio(last);}
  const showEval = !!last;
  if (!showEval) { const pendingReviews=(session?.responses||[]).filter(item=>item.evaluation?.review_status==='pending'); node.className=pendingReviews.length?'review-queue':'empty-state'; node.innerHTML=pendingReviews.length?`<div class="review-subtitle">尚待人工复核的规则评分</div>${pendingReviews.map(item=>`<div class="review-claim"><div><small>${esc(item.competency)}</small><span>${esc(item.question)}</span></div><div class="claim-actions"><button type="button" data-score-review="${esc(item.id)}">人工复核评分</button></div></div>`).join('')}`:'提交回答后显示内容、深度、结构和影响力评分。'; return; }
  const e=last.evaluation; const sourceLabel=e.scoring_source==='human'?'人工已复核':e.scoring_source==='deterministic_rule'?'规则评分 · 待复核':'AI 评分'; const dimensionLabels={content:'岗位相关证据',technical_depth:'决策与专业深度',structure:'表达结构',impact:'结果与复盘'}; const dimensionHtml=(e.dimension_feedback||[]).map(item=>`<div class="dimension-card"><div><strong>${esc(dimensionLabels[item.dimension]||item.dimension)}</strong><b>${Math.round((item.score||0)*100)} · ${esc(item.level)}</b></div><p>${esc(item.evidence)}</p><small>下一步：${esc(item.suggestion)}</small></div>`).join(''); node.className=''; node.innerHTML=`<div class="score-grid">${[['证据',e.content],['深度',e.technical_depth],['结构',e.structure],['结果',e.impact]].map(([n,v])=>`<div class="score"><span>${n}</span><strong>${Math.round(v*100)}</strong></div>`).join('')}</div><div class="claim-actions"><small>${esc(sourceLabel)}</small><button type="button" data-score-review="${esc(last.id)}">人工复核评分</button></div>${dimensionHtml?`<div class="dimension-feedback">${dimensionHtml}</div>`:''}${list('优先改进',e.feedback)}<div class="result-block"><h4>基于你本次回答的重组示范</h4><p>${esc(e.improved_answer)}</p></div>`;
  if(last.speech_delivery?.strengths?.length||last.speech_delivery?.improvements?.length)renderSpeechFeedback(last.speech_delivery);
  node.insertAdjacentHTML('beforeend', renderSpokenAnswerAnalysis(last.evaluation));
  const details=node.innerHTML;
  const previous=[...(session?.attempt_history||[])].reverse().find(item=>item.question_id===last.question_id&&item.question===last.question);
  const headline=(e.feedback||[])[0]||'已完成本题证据检查，可以查看具体依据或立即重答。';
  node.innerHTML=`<div class="coach-summary"><small>本题最优先改进</small><strong>${esc(headline)}</strong><div class="claim-actions">${completed?'':`<button type="button" data-quick-retry="${esc(last.id)}">按建议重答</button>`}${last.audio_file?`<button type="button" data-delete-mock-audio="${esc(last.id)}">删除本次录音</button>`:''}</div></div>${renderQuestionAlignment(e)}${renderRetryComparison(last,previous)}<details class="coach-details"><summary>查看完整分析、评分和校准依据</summary>${details}</details>`;
}

async function ensureSession() { if (state.sessionId) return true; $('session-dialog').showModal(); toast('请先创建一个会话'); return false; }

$('nav').addEventListener('click', e => { const button=e.target.closest('[data-view]'); if(button) setView(button.dataset.view); });
document.querySelectorAll('[data-role]').forEach(button => button.onclick = () => setRole(button.dataset.role));
document.querySelectorAll('[data-global-view]').forEach(button => button.onclick = () => setView(button.dataset.globalView));
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>setView(b.dataset.go));
$('new-session').onclick=()=> $('session-dialog').showModal();
$('close-session-dialog').onclick=()=> $('session-dialog').close();
$('session-select').onchange=async e=>{state.sessionId=e.target.value;await loadSession();};
$('session-form').onsubmit=async e=>{e.preventDefault();try{const data=await api('/api/interviews/sessions',{method:'POST',body:JSON.stringify({candidate_name:$('new-candidate').value,job_title:$('new-job').value,company_name:$('new-company').value})});state.sessionId=data.id;$('session-dialog').close();await loadSessions();toast('会话已创建');}catch(error){toast(error.message,true)}};

document.querySelectorAll('.resume-file').forEach(input => input.onchange = async event => {
  const file = event.target.files?.[0];
  if (!file || !await ensureSession()) return;
  const upload = event.target.closest('.resume-upload');
  const form = new FormData(); form.append('file', file);
  const structure = upload?.querySelector('.resume-llm-structure');
  if (structure?.checked) form.append('structure', 'llm');
  busy(upload, true);
  try {
    const data = await api(`/api/resumes/${state.sessionId}/upload`, {method:'POST', body:form});
    state.session = data.state;
    $(upload.dataset.resumeTarget).value = state.session.candidate.raw_resume_text;
    renderState(); toast(state.session.resume_review?.structured_by === 'llm' ? 'AI 结构化完成，已按板块分类' : '简历已解析，请先查看校验项');
  } catch (error) { toast(error.message, true); }
  finally { busy(upload, false); event.target.value = ''; }
});

document.addEventListener('click', async event => {
  const button = event.target.closest('[data-claim-action]');
  if (!button || !state.sessionId) return;
  if(button.dataset.claimAction==='modified'){
    const claim=state.session?.resume_review?.claims?.find(x=>x.id===button.dataset.claimId);if(!claim)return;
    $('claim-dialog').dataset.claimId=claim.id;$('claim-statement').value=claim.statement;$('claim-note').value='';$('claim-dialog').showModal();return;
  }
  try {
    const data = await api(`/api/resumes/${state.sessionId}/claims/${button.dataset.claimId}`, {method:'PATCH', body:JSON.stringify({status:button.dataset.claimAction})});
    state.session = data.state; renderState(); toast({confirmed:'已记录为本人确认，仍未外部核验',needs_documents:'已标记为需要材料',ignored:'已忽略，后续 Agent 不会使用',unverified:'已恢复为简历自述·待核验'}[button.dataset.claimAction]||'已更新');
  } catch (error) { toast(error.message, true); }
});

document.addEventListener('click', async event => {
  const button = event.target.closest('[data-jd-action]');
  if (!button || !state.sessionId) return;
  const index = Number(button.dataset.jdIndex);
  const item = state.session?.job_review?.requirements?.[index];
  if (!item) return;
  let text = '';
  if (button.dataset.jdAction === 'edit') {
    text = window.prompt('编辑为明确岗位要求', item.text) || '';
    if (!text.trim()) return;
  }
  try {
    const data = await api(`/api/analysis/job/${state.sessionId}/requirements/${index}`, {
      method: 'PATCH',
      body: JSON.stringify({action: button.dataset.jdAction, text})
    });
    state.session = data.state;
    renderState();
    toast({confirm:'已确认为明确要求',edit:'已编辑并确认',delete:'已删除该推测要求'}[button.dataset.jdAction] || 'JD 要求已更新');
  } catch (error) { toast(error.message, true); }
});

document.addEventListener('click', async event => {
  const button = event.target.closest('[data-fact-action]');
  if (!button || !state.sessionId) return;
  const action = button.dataset.factAction;
  const note = action === 'reset'
    ? ''
    : (window.prompt(action === 'accept' ? '确认说明（可选）' : '排除原因（可选）', '') || '');
  try {
    const data = await api(`/api/intelligence/${state.sessionId}/facts/${button.dataset.factId}`, {
      method: 'PATCH',
      body: JSON.stringify({action, note})
    });
    state.session = data.state;
    renderState();
    toast({accept:'事实卡已确认',reject:'事实卡已排除',reset:'事实卡已恢复待审'}[action] || '事实卡已更新');
  } catch (error) { toast(error.message, true); }
});

$('cancel-claim').onclick=()=>$('claim-dialog').close();
$('save-claim').onclick=async()=>{const dialog=$('claim-dialog');try{const data=await api(`/api/resumes/${state.sessionId}/claims/${dialog.dataset.claimId}`,{method:'PATCH',body:JSON.stringify({status:'modified',statement:$('claim-statement').value,note:$('claim-note').value})});state.session=data.state;dialog.close();renderState();toast('修改后的事实已确认，后续 Agent 将使用新表述');}catch(error){toast(error.message,true)}};

function openScoreReview(recordId) {
  const records=[...(state.session?.mock_session?.responses||[]),...(state.session?.live_interview_records||[])];
  const record=records.find(item=>item.id===recordId);if(!record)return;
  const evaluation=record.evaluation||{};const evidence=(state.session?.evidence||[]).find(item=>item.source_record_id===recordId);
  const dialog=$('score-review-dialog');dialog.dataset.recordId=recordId;
  $('review-content').value=Math.round((evaluation.content||0)*100);$('review-depth').value=Math.round((evaluation.technical_depth||0)*100);$('review-structure').value=Math.round((evaluation.structure||0)*100);$('review-impact').value=Math.round((evaluation.impact||0)*100);$('review-polarity').value=evidence?.polarity||'neutral';$('review-note').value='';dialog.showModal();
}
document.addEventListener('click',event=>{const button=event.target.closest('[data-score-review]');if(button)openScoreReview(button.dataset.scoreReview);});
$('cancel-score-review').onclick=()=>$('score-review-dialog').close();
$('score-review-form').onsubmit=async event=>{event.preventDefault();const dialog=$('score-review-dialog');const score=id=>Number($(id).value)/100;try{const data=await api(`/api/evaluations/${state.sessionId}/answers/${dialog.dataset.recordId}/review`,{method:'PATCH',body:JSON.stringify({content:score('review-content'),technical_depth:score('review-depth'),structure:score('review-structure'),impact:score('review-impact'),evidence_polarity:$('review-polarity').value,note:$('review-note').value})});state.session=data.state;dialog.close();renderAll();toast('人工评分已确认，旧报告已失效，请重新生成');}catch(error){toast(error.message,true)}};

async function resolveEntity(accept){const dialog=$('entity-dialog');const proposedName=accept?$('entity-name').value.trim():'';if(accept&&!proposedName){toast('请填写确认后的实体名称',true);return;}try{const data=await api(`/api/intelligence/${state.sessionId}/entities/${dialog.dataset.resolutionId}`,{method:'PATCH',body:JSON.stringify({accept,proposed_name:proposedName})});state.session=data.state;dialog.close();hydrateSessionForms();renderState();toast(accept?'实体名称已确认':'已保留原名称');}catch(error){toast(error.message,true)}}
$('accept-entity').onclick=()=>resolveEntity(true);$('reject-entity').onclick=()=>resolveEntity(false);

$('candidate-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);showWorkflowStarting();try{const payload={role:'candidate',resume_text:$('candidate-resume').value,job_description:$('candidate-jd').value,company_name:$('candidate-company').value,company_context:$('candidate-company-context').value,interviewer_name:$('interviewer-name').value,interviewer_position:$('interviewer-position').value,authorized_public_research:$('candidate-research-consent').checked};const endpoint=$('candidate-autopilot').checked?`/api/autopilot/${state.sessionId}/run`:'/api/workflows/candidate-prep';if(!$('candidate-autopilot').checked)payload.session_id=state.sessionId;const data=await api(endpoint,{method:'POST',body:JSON.stringify(payload)});state.session=data.state;await loadSessions();toast(state.session.autopilot?.enabled?'AI 已推进到需要你回答的阶段':'候选人策略已生成');}catch(error){await loadSession();toast(`运行失败：${error.message}`,true)}finally{busy(form,false)}};
$('enterprise-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);showWorkflowStarting();try{const payload={role:'interviewer',resume_text:$('enterprise-resume').value,job_description:$('enterprise-jd').value,company_name:$('enterprise-company').value,company_context:$('enterprise-context').value,authorized_public_research:$('enterprise-research-consent').checked};const endpoint=$('enterprise-autopilot').checked?`/api/autopilot/${state.sessionId}/run`:'/api/workflows/enterprise-design';if(!$('enterprise-autopilot').checked)payload.session_id=state.sessionId;const data=await api(endpoint,{method:'POST',body:JSON.stringify(payload)});state.session=data.state;await loadSessions();toast(state.session.autopilot?.enabled?'AI 已完成设计，等待采集真实面试证据':'面试 Blueprint 已生成');}catch(error){await loadSession();toast(`运行失败：${error.message}`,true)}finally{busy(form,false)}};

$('start-mock').onclick=async()=>{if(!await ensureSession())return;try{await api(`/api/mock-interviews/${state.sessionId}/start`,{method:'POST'});await loadSession();toast('模拟面试已开始');}catch(error){toast(error.message,true)}};
function startMockScoringProgress(){mockVoiceNote('正在提交回答并提取可核验证据…');const timers=[setTimeout(()=>mockVoiceNote('模型正在评分并校准证据一致性；完成后会自动展示结果…'),1200),setTimeout(()=>mockVoiceNote('本地模型仍在推理；你的回答保留在输入框中，请勿重复提交。'),12000)];return()=>timers.forEach(clearTimeout);}
$('answer-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;const mockSession=state.session?.mock_session;const question=state.session?.mock_interview?.questions?.[mockSession?.current_question_index];if(!question)return;busy(form,true);const stopProgress=startMockScoringProgress();try{await api(`/api/mock-interviews/${state.sessionId}/answers`,{method:'POST',body:JSON.stringify({question_id:mockSession.pending_parent_question_id||question.id,answer:$('mock-answer').value,retry:mockRetry,retry_response_id:mockRetryResponseId||null,recording_id:mockPendingRecordingId||null})});$('mock-answer').value='';mockPendingRecordingId='';mockRetry=false;mockRetryResponseId='';await loadSession();toast('回答已评分');}catch(error){mockVoiceNote('评分未完成，回答仍保留，可检查后重试。');toast(error.message,true)}finally{stopProgress();busy(form,false)}};
$('mock-retry').onclick=e=>{mockRetry=true;mockRetryResponseId=e.currentTarget.dataset.responseId||'';mockPendingRecordingId='';$('mock-answer').value='';renderMock();};
$('mock-retry-main').onclick=e=>{mockRetry=true;mockRetryResponseId=e.currentTarget.dataset.responseId||'';mockPendingRecordingId='';$('mock-answer').value='';renderMock();};
document.addEventListener('click',async event=>{const route=event.target.closest('[data-route]');if(route){setView(route.dataset.route);return;}const fresh=event.target.closest('[data-new-practice]');if(fresh){$('session-dialog').showModal();return;}const retry=event.target.closest('[data-quick-retry]');if(retry){mockRetry=true;mockRetryResponseId=retry.dataset.quickRetry||'';mockPendingRecordingId='';$('mock-answer').value='';renderMock();$('mock-answer')?.focus();return;}const play=event.target.closest('[data-play-mock-audio]');if(play){await loadMockAnswerAudio(play.dataset.playMockAudio,true);mockVoiceNote('正在回放上一版回答');return;}const remove=event.target.closest('[data-delete-mock-audio]');if(remove){try{await api(`/api/mock-interviews/${state.sessionId}/answers/${remove.dataset.deleteMockAudio}/audio`,{method:'DELETE'});await loadSession();toast('录音已从本机删除，文字与评分仍保留');}catch(error){toast(error.message,true);}}});
$('mock-next').onclick=async()=>{if(!await ensureSession())return;cancelMockQuestionSpeech();mockRetry=false;mockRetryResponseId='';try{await api(`/api/mock-interviews/${state.sessionId}/next`,{method:'POST'});await loadSession();toast('下一题');}catch(error){toast(error.message,true)}};
$('mock-prev').onclick=async()=>{if(!await ensureSession())return;cancelMockQuestionSpeech();mockRetry=false;mockRetryResponseId='';try{await api(`/api/mock-interviews/${state.sessionId}/previous`,{method:'POST'});await loadSession();toast('上一题');}catch(error){toast(error.message,true)}};
$('mock-finish').onclick=async()=>{if(!await ensureSession())return;cancelMockQuestionSpeech();try{await api(`/api/mock-interviews/${state.sessionId}/finish`,{method:'POST'});await loadSession();toast('面试已结束');}catch(error){toast(error.message,true)}};
let mockVoiceRecorder=null;
let mockVoiceChunks=[];
let mockVoiceStream=null;
function mockVoiceNote(msg){const n=$('mock-voice-note');if(n)n.textContent=msg||'';}
function clearObjectUrl(kind){const value=kind==='question'?mockQuestionAudioUrl:mockAnswerAudioUrl;if(value)URL.revokeObjectURL(value);if(kind==='question')mockQuestionAudioUrl='';else mockAnswerAudioUrl='';}
function clearAudioElement(id){const audio=$(id);if(!audio)return;audio.pause();audio.removeAttribute('src');audio.load();audio.classList.add('hidden');}
function clearMockQuestionAudio(){clearObjectUrl('question');clearAudioElement('mock-question-audio');}
function cancelMockAnswerAudioLoad(){mockAnswerAudioRequestSequence+=1;if(mockAnswerAudioAbort)mockAnswerAudioAbort.abort();mockAnswerAudioAbort=null;}
function clearMockAnswerExperience(){cancelMockAnswerAudioLoad();clearObjectUrl('answer');mockAnswerAudioResponseId='';mockPendingRecordingId='';clearAudioElement('mock-answer-playback');renderSpeechFeedback(null);mockVoiceNote('');}
async function loadMockAnswerAudio(responseId,historical=false){
  cancelMockAnswerAudioLoad();
  const sequence=mockAnswerAudioRequestSequence;const sessionId=state.sessionId;const controller=new AbortController();mockAnswerAudioAbort=controller;mockAnswerAudioResponseId=responseId;
  try{const headers=state.token?{Authorization:`Bearer ${state.token}`}:{},response=await fetch(`/api/mock-interviews/${sessionId}/answers/${responseId}/audio`,{headers,signal:controller.signal});if(!response.ok)throw new Error(`录音读取失败 (${response.status})`);const blob=await response.blob();if(sequence!==mockAnswerAudioRequestSequence||sessionId!==state.sessionId||(!historical&&responseId!==mockDisplayedResponseId))return;clearObjectUrl('answer');mockAnswerAudioUrl=URL.createObjectURL(blob);const audio=$('mock-answer-playback');audio.src=mockAnswerAudioUrl;audio.classList.remove('hidden');}catch(error){if(error.name!=='AbortError'){if(mockAnswerAudioResponseId===responseId)mockAnswerAudioResponseId='';mockVoiceNote(error.message);}}finally{if(mockAnswerAudioAbort===controller)mockAnswerAudioAbort=null;}
}
function syncMockAnswerAudio(response){
  if(!response?.audio_file){if(mockAnswerAudioResponseId&&mockAnswerAudioResponseId!==response?.id){cancelMockAnswerAudioLoad();clearObjectUrl('answer');mockAnswerAudioResponseId='';clearAudioElement('mock-answer-playback');}return;}
  if(mockAnswerAudioUrl&&!mockAnswerAudioResponseId){mockAnswerAudioResponseId=response.id;return;}
  if(mockAnswerAudioResponseId!==response.id)loadMockAnswerAudio(response.id);
}
function cancelMockQuestionSpeech(){mockSpeechRequestSequence+=1;if(mockSpeechAbort)mockSpeechAbort.abort();mockSpeechAbort=null;}
function resetMockAudioExperience(){
  mockVoiceGeneration+=1;mockVoiceSessionId='';asrPreviewSequence+=1;
  cancelMockQuestionSpeech();
  if(mockVoiceRecorder){mockVoiceRecorder.ondataavailable=null;mockVoiceRecorder.onstop=null;try{if(mockVoiceRecorder.state!=='inactive')mockVoiceRecorder.stop();}catch{}mockVoiceRecorder=null;}
  if(mockVoiceStream)mockVoiceStream.getTracks().forEach(track=>track.stop());
  mockVoiceStream=null;mockVoiceChunks=[];mockCurrentSpeechKey='';mockDisplayedResponseId='';mockSpokenQuestionKey='';
  clearMockQuestionAudio();clearMockAnswerExperience();
  if($('mock-voice'))$('mock-voice').disabled=false;if($('mock-stop-voice'))$('mock-stop-voice').disabled=true;
}
async function speakCurrentMockQuestion(automatic=false){
  const session=state.session?.mock_session;const question=state.session?.mock_interview?.questions?.[session?.current_question_index];if(!state.sessionId||!question)return;
  if(automatic&&(state.view!=='mock'||document.visibilityState!=='visible'||!$('view-mock')?.classList.contains('active'))){mockSpokenQuestionKey='';return;}
  cancelMockQuestionSpeech();const requestSequence=mockSpeechRequestSequence;const requestedSessionId=state.sessionId;const requestedSpeechKey=mockCurrentSpeechKey;const responseId=mockDisplayedResponseId;const controller=new AbortController();mockSpeechAbort=controller;
  const button=$('mock-speak-question');if(button)button.disabled=true;
  try{const headers=state.token?{Authorization:`Bearer ${state.token}`}:{},query=responseId?`?response_id=${encodeURIComponent(responseId)}`:'',response=await fetch(`/api/mock-interviews/${requestedSessionId}/questions/${question.id}/speech${query}`,{method:'POST',headers,signal:controller.signal});if(!response.ok){const data=await response.json().catch(()=>({}));throw new Error(data.detail||`请求失败 (${response.status})`);}const blob=await response.blob();if(requestSequence!==mockSpeechRequestSequence||requestedSessionId!==state.sessionId||requestedSpeechKey!==mockCurrentSpeechKey)return;clearMockQuestionAudio();mockQuestionAudioUrl=URL.createObjectURL(blob);const audio=$('mock-question-audio');audio.src=mockQuestionAudioUrl;audio.classList.remove('hidden');await audio.play();}catch(error){if(error.name!=='AbortError'&&!automatic)toast(`问题朗读失败：${error.message}`,true);}finally{if(mockSpeechAbort===controller)mockSpeechAbort=null;if(requestSequence===mockSpeechRequestSequence&&button)button.disabled=false;}
}
$('mock-speak-question').onclick=()=>speakCurrentMockQuestion(false);
$('mock-auto-speak').onchange=e=>{localStorage.setItem('interviewos.autoSpeak',e.target.checked?'1':'0');if(e.target.checked&&state.view==='mock'){mockSpokenQuestionKey='';renderMock();}};
$('mock-auto-speak').checked=localStorage.getItem('interviewos.autoSpeak')!=='0';
document.addEventListener('visibilitychange',()=>{if(document.visibilityState!=='visible'){cancelMockQuestionSpeech();clearMockQuestionAudio();}});
function renderSpeechFeedback(feedback){const node=$('mock-speech-feedback');if(!node)return;if(!feedback){node.classList.add('hidden');node.innerHTML='';return;}const rows=[['语速',feedback.pace],['停顿',feedback.pauses],['填充词',feedback.fillers],['音量稳定',feedback.volume],['语调',feedback.intonation],['清晰度',feedback.clarity]].filter(([,value])=>value&&value!=='无法判断');node.classList.remove('hidden');node.innerHTML=`<div class="review-subtitle">语音表达辅导 · ${feedback.source==='audio_model'?'音频模型':'本地指标'}</div>${rows.map(([name,value])=>`<p><strong>${esc(name)}</strong><span>${esc(value)}</span></p>`).join('')}${list('可执行改进',feedback.improvements||[])}<small>${esc(feedback.disclaimer||'仅用于表达训练，不进入录用评价。')}</small>`;}
async function pollMockSpeechFeedback(sessionId,recordingId,generation){for(let attempt=0;attempt<20;attempt+=1){await new Promise(resolve=>setTimeout(resolve,1500));if(generation!==mockVoiceGeneration||sessionId!==state.sessionId)return;try{const data=await api(`/api/mock-interviews/${sessionId}/recordings/${recordingId}/speech-feedback`);renderSpeechFeedback(data.speech_feedback);if(data.status==='completed'){mockVoiceNote('语音表达分析已完成；可回放、修改文字或提交回答');return;}}catch(error){if(attempt>2){mockVoiceNote(`表达分析暂不可用：${error.message}`);return;}}}mockVoiceNote('转写已完成；深度语音分析仍在后台处理');}
$('mock-voice').onclick=async()=>{if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){toast('当前浏览器不支持麦克风',true);return;}const recordingSessionId=state.sessionId;const generation=++mockVoiceGeneration;mockVoiceSessionId=recordingSessionId;clearMockAnswerExperience();try{const stream=await navigator.mediaDevices.getUserMedia(microphoneConstraints());if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId){stream.getTracks().forEach(track=>track.stop());return;}mockVoiceStream=stream;const preferred=liveAudioType();mockVoiceRecorder=new MediaRecorder(mockVoiceStream,preferred?{mimeType:preferred}:undefined);mockVoiceChunks=[];mockVoiceRecorder.ondataavailable=e=>{if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId||!e.data.size)return;mockVoiceChunks.push(e.data);previewRecordedChunks(mockVoiceChunks,mockVoiceRecorder?.mimeType||preferred,text=>{if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId)return;$('mock-answer').value=text;mockVoiceNote('实时转写草稿 · 停止后确认最终文本')})};mockVoiceRecorder.onstop=()=>{if(generation===mockVoiceGeneration)mockVoiceRecorder=null;};mockVoiceRecorder.start(1000);$('mock-voice').disabled=true;$('mock-stop-voice').disabled=false;mockVoiceNote('正在录音并显示实时转写草稿…');}catch(error){if(generation!==mockVoiceGeneration)return;if(mockVoiceStream){mockVoiceStream.getTracks().forEach(t=>t.stop());mockVoiceStream=null;}mockVoiceSessionId='';toast(`无法使用麦克风：${error.message}`,true);}};
$('mock-stop-voice').onclick=async()=>{const recorder=mockVoiceRecorder;const stream=mockVoiceStream;const recordingSessionId=mockVoiceSessionId;const generation=mockVoiceGeneration;mockVoiceRecorder=null;mockVoiceStream=null;mockVoiceSessionId='';if(!recorder){mockVoiceNote('');return;}$('mock-stop-voice').disabled=true;mockVoiceNote('正在确认最终转写…');asrPreviewSequence+=1;recorder.onstop=async()=>{const chunks=mockVoiceChunks;mockVoiceChunks=[];stream?.getTracks().forEach(t=>t.stop());if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId)return;const type=recorder.mimeType||'audio/webm';const blob=new Blob(chunks,{type});clearObjectUrl('answer');clearAudioElement('mock-answer-playback');renderSpeechFeedback(null);mockAnswerAudioUrl=URL.createObjectURL(blob);mockAnswerAudioResponseId='';const playback=$('mock-answer-playback');playback.src=mockAnswerAudioUrl;playback.classList.remove('hidden');try{const wav=await encodeBlobAsWav(blob);if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId)return;const form=new FormData();form.append('file',wav,'answer.wav');const data=await api(`/api/mock-interviews/${recordingSessionId}/transcribe`,{method:'POST',body:form});if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId)return;mockPendingRecordingId=data.recording_id||'';$('mock-answer').value=(data.text||'').trim();renderSpeechFeedback(data.speech_feedback);$('mock-voice').disabled=false;mockVoiceNote('最终转写已确认，可立即修改或提交；深度语音分析在后台继续');pollMockSpeechFeedback(recordingSessionId,mockPendingRecordingId,generation);}catch(error){if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId)return;mockPendingRecordingId='';$('mock-voice').disabled=false;mockVoiceNote('录音可回放；保留实时草稿，请手动检查');toast(`语音转写失败：${error.message}`,true);}};recorder.stop();};
async function generateEvaluation(button = null) {
  if (!await ensureSession()) return;
  if (button) busy(button, true);
  try {
    const data = await api(`/api/evaluations/${state.sessionId}`, {method:'POST'});
    state.session = data.state;
    renderState();
    toast('最终评价已生成');
  } catch (error) { toast(error.message, true); }
  finally { if (button) busy(button, false); }
}
document.querySelectorAll('.evaluation-trigger').forEach(button => button.onclick = () => generateEvaluation(button));

async function submitTranscript(){if(!await ensureSession())return;const form=$('transcript-form');const lines=$('transcript-input').value.split('\n').map(x=>x.trim()).filter(Boolean);const entries=lines.map(line=>{const parts=line.split('|').map(x=>x.trim());return {competency:parts[0]||'综合能力',question:parts[1]||'',answer:parts.slice(2).join(' | ')};});if(!entries.length||entries.some(x=>!x.question||!x.answer)){toast('每行必须包含“能力 | 问题 | 回答”三部分',true);return;}busy(form,true);try{const data=await api(`/api/evaluations/${state.sessionId}/transcript`,{method:'POST',body:JSON.stringify({entries,auto_evaluate:true})});state.session=data.state;renderState();toast(`已导入 ${entries.length} 条真实回答并生成评价`);}catch(error){toast(error.message,true)}finally{busy(form,false)}}
$('transcript-form').onsubmit=e=>{e.preventDefault();submitTranscript();};
$('import-transcript').onclick=submitTranscript;

function startSessionRecorder() {
  const unavailable = microphoneAvailabilityMessage();
  if (sessionRecorder || unavailable) {
    showSessionRecordingError(unavailable || '全场录音已在运行');
    return;
  }
  navigator.mediaDevices.getUserMedia(microphoneConstraints()).then(stream => {
    const preferred = liveAudioType();
    sessionRecorder = new MediaRecorder(stream, preferred ? {mimeType: preferred} : undefined);
    sessionChunks = [];
    sessionRecorder.ondataavailable = e => { if (e.data.size) sessionChunks.push(e.data); };
    sessionRecorder.onstop = () => { sessionRecorder = null; };
    sessionRecorder.start(1000);
    sessionRecordingActive = true;
    renderLive();
  }).catch(() => { showSessionRecordingError('无法访问麦克风，未在录制全场录音（转写不受影响）'); renderLive(); });
}

function showSessionRecordingError(message) {
  sessionRecordingActive = false;
  const status = $('session-recording-status');
  if (status) { status.textContent = message; status.classList.add('error'); }
}

function stopSessionRecorder() {
  const recorder = sessionRecorder;
  if (!recorder) return;
  sessionRecordingActive = false;
  recorder.stop();
}

async function uploadSessionAudio() {
  if (!sessionRecordingActive && !sessionChunks.length) return;
  stopSessionRecorder();
  if (!sessionChunks.length) return;
  const type = (sessionRecorder?.mimeType || 'audio/webm').replace('audio/webm;codecs=opus', 'audio/webm');
  const blob = new Blob(sessionChunks, {type: sessionRecorder?.mimeType || 'audio/webm'});
  sessionChunks = [];
  try {
    const wav = await encodeBlobAsWav(blob);
    const isWav = wav.type === 'audio/wav';
    const form = new FormData();
    form.append('file', wav, `session-${state.sessionId.slice(0, 8)}.${isWav ? 'wav' : 'webm'}`);
    await api(`/api/live-interviews/${state.sessionId}/audio/final`, {method: 'POST', body: form});
    toast('全场录音已保存');
  } catch (error) { toast(`全场录音保存失败：${error.message}`, true); }
}

async function startLiveInterview(){if(!await ensureSession())return;try{const data=await api(`/api/live-interviews/${state.sessionId}/start`,{method:'POST',body:JSON.stringify({consent_confirmed:$('live-consent').checked})});state.session=data.state;renderLive();toast('实时面试已开始');if($('live-consent').checked)startSessionRecorder();}catch(error){toast(error.message,true)}}
$('live-start').onclick=startLiveInterview;
$('live-pause').onclick=async()=>{const next=state.session?.live_interview?.status==='paused'?'active':'paused';try{const data=await api(`/api/live-interviews/${state.sessionId}/status`,{method:'POST',body:JSON.stringify({status:next})});state.session=data.state;renderLive();toast(next==='paused'?'实时会话已暂停':'实时会话已恢复');}catch(error){toast(error.message,true)}};
$('live-finish').onclick=async()=>{try{if(liveContinuousMode||liveContinuousQueue.length||liveContinuousUploading){toast('请先停止连续监听并等待转写队列处理完成，再结束会话',true);renderLive();return;}if(liveRecorder&&liveRecorder.state==='recording')liveRecorder.stop();await uploadSessionAudio();const data=await api(`/api/live-interviews/${state.sessionId}/status`,{method:'POST',body:JSON.stringify({status:'completed'})});state.session=data.state;renderLive();toast('实时面试已结束，请审阅转写');}catch(error){toast(error.message,true)}};
$('live-text-form').onsubmit=async e=>{e.preventDefault();const text=$('live-text').value.trim();if(!text)return;const form=e.currentTarget;busy(form,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/segments`,{method:'POST',body:JSON.stringify({text,speaker:$('live-speaker').value})});state.session=data.state;$('live-text').value='';renderLive();toast('发言已加入实时对话');}catch(error){toast(error.message,true)}finally{busy(form,false)}};

function liveAudioType(){return ['audio/webm;codecs=opus','audio/webm','audio/mp4'].find(type=>MediaRecorder.isTypeSupported(type))||'';}
function liveAudioFilename(type,prefix='speech'){return type.includes('mp4')?`${prefix}.m4a`:`${prefix}.webm`;}
async function encodeBlobAsWav(blob){
  // Convert MediaRecorder webm/opus to a 16 kHz mono 16-bit PCM WAV so the
  // LAN ASR (Qwen3-ASR on 8007) can decode it — it rejects webm. Returns a
  // Blob; falls back to the original blob if decoding is unavailable.
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx || !blob.size) return blob;
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await new AudioCtx().decodeAudioData(arrayBuffer);
    const sampleRate = 16000;
    const offline = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * sampleRate), sampleRate);
    const source = offline.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    const pcm = rendered.getChannelData(0);
    const dataSize = pcm.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);
    let offset = 44;
    for (let i = 0; i < pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
    return new Blob([view], {type: 'audio/wav'});
  } catch (error) {
    return blob;
  }
}
function showLiveAsrPreview(text=''){
  const box=$('live-asr-preview');const copy=$('live-asr-preview-text');if(!box||!copy)return;
  copy.textContent=text;box.classList.toggle('hidden',!text);
}
function resetAsrPreview(){asrPreviewSequence+=1;asrPreviewLastAt=0;showLiveAsrPreview('');}
async function previewRecordedChunks(chunks,type,onText){
  const now=performance.now();const previewSessionId=state.sessionId;
  if(asrPreviewBusy||chunks.length<2||now-asrPreviewLastAt<ASR_PREVIEW_INTERVAL_MS||!previewSessionId)return;
  asrPreviewBusy=true;asrPreviewLastAt=now;const requestSequence=++asrPreviewSequence;
  const snapshot=new Blob([...chunks],{type:type||'audio/webm'});
  try{
    const wav=await encodeBlobAsWav(snapshot);const form=new FormData();
    form.append('file',wav,wav.type==='audio/wav'?'preview.wav':'preview.webm');form.append('language','zh');
    const data=await api(`/api/live-interviews/${previewSessionId}/audio/preview`,{method:'POST',body:form});
    if(requestSequence===asrPreviewSequence&&previewSessionId===state.sessionId&&data.text?.trim())onText(data.text.trim());
  }catch(error){
    // Preview is best-effort; the final transcription path reports failures.
  }finally{asrPreviewBusy=false;}
}
async function uploadLiveAudioBlob(blob,type,speaker,prefix='speech',mode='single'){
  const wavBlob = await encodeBlobAsWav(blob);
  const isWav = wavBlob.type === 'audio/wav';
  const form=new FormData();
  form.append('file', wavBlob, isWav ? `${prefix}.wav` : liveAudioFilename(type,prefix));
  form.append('speaker',speaker);
  form.append('language','zh');
  if(mode==='dialogue')form.append('mode','dialogue');
  const data=await api(`/api/live-interviews/${state.sessionId}/audio`,{method:'POST',body:form});state.session=data.state;return data;
}

$('live-record').onclick=async()=>{if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){toast('当前浏览器不支持麦克风录制',true);return;}try{liveMediaStream=await navigator.mediaDevices.getUserMedia(microphoneConstraints());const preferred=liveAudioType();liveRecorder=new MediaRecorder(liveMediaStream,preferred?{mimeType:preferred}:undefined);liveAudioChunks=[];resetAsrPreview();liveRecorder.ondataavailable=event=>{if(!event.data.size)return;liveAudioChunks.push(event.data);previewRecordedChunks(liveAudioChunks,liveRecorder?.mimeType||preferred,showLiveAsrPreview)};liveRecorder.onstop=uploadLiveRecording;liveRecorder.start(1000);liveDialogueMode=false;$('live-record').disabled=true;$('live-stop-record').disabled=false;$('live-recording-note').textContent=`正在录制${$('live-speaker').value==='candidate'?'候选人':$('live-speaker').value==='interviewer'?'面试官':'待确认'}发言并实时转写…`;renderLive();}catch(error){toast(`无法使用麦克风：${error.message}`,true)}};
$('live-dialogue').onclick=async()=>{if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){toast('当前浏览器不支持麦克风录制',true);return;}try{liveMediaStream=await navigator.mediaDevices.getUserMedia(microphoneConstraints());const preferred=liveAudioType();liveRecorder=new MediaRecorder(liveMediaStream,preferred?{mimeType:preferred}:undefined);liveAudioChunks=[];resetAsrPreview();liveRecorder.ondataavailable=event=>{if(!event.data.size)return;liveAudioChunks.push(event.data);previewRecordedChunks(liveAudioChunks,liveRecorder?.mimeType||preferred,showLiveAsrPreview)};liveRecorder.onstop=uploadLiveRecording;liveRecorder.start(1000);liveDialogueMode=true;$('live-record').disabled=true;$('live-stop-record').disabled=false;$('live-recording-note').textContent='对话模式：实时草稿仅展示文字，最终结果再自动区分说话人。';renderLive();toast('对话模式开始，请录完整段对话');}catch(error){toast(`无法使用麦克风：${error.message}`,true)}};
$('live-stop-record').onclick=()=>{if(liveRecorder?.state==='recording'){asrPreviewSequence+=1;liveRecorder.stop();$('live-stop-record').disabled=true;$('live-recording-note').textContent='录制结束，正在确认最终转写…'}};

let liveDialogueMode = false;
async function uploadLiveRecording(){const recorder=liveRecorder;const stream=liveMediaStream;liveRecorder=null;liveMediaStream=null;stream?.getTracks().forEach(track=>track.stop());const type=recorder?.mimeType||'audio/webm';const blob=new Blob(liveAudioChunks,{type});liveAudioChunks=[];const wasDialogue=liveDialogueMode;liveDialogueMode=false;if(!blob.size){toast('没有录到音频',true);renderLive();return;}try{await uploadLiveAudioBlob(blob,type,'unknown','speech',wasDialogue?'dialogue':'single');resetAsrPreview();$('live-recording-note').textContent=wasDialogue?'最终对话已转写并自动区分说话人；候选人回答可确认。':'最终转写已确认；候选人回答结束后可生成下一问题。';renderLive();toast(wasDialogue?'对话已自动分说话人':'音频已转为文字');}catch(error){$('live-recording-note').textContent='最终转写失败；实时草稿仍保留，可手动复制补录。';toast(error.message,true);renderLive();}}

$('live-continuous').onclick=startLiveContinuousVad;
$('live-stop-continuous').onclick=()=>{if(liveRecorder?.state==='recording'){liveContinuousMode=false;liveRecorder.stop();$('live-recording-note').textContent='连续监听停止中；正在处理已录到的音频。'}renderLive();};
function enqueueLiveContinuousChunk(blob){if(!blob.size)return;liveContinuousQueue.push({blob,type:blob.type||'audio/webm',index:++liveContinuousChunkIndex});processLiveContinuousQueue();renderLive();}
async function processLiveContinuousQueue(){if(liveContinuousUploading)return;liveContinuousUploading=true;try{while(liveContinuousQueue.length){const chunk=liveContinuousQueue.shift();if(!liveContinuousMode){$('live-recording-note').textContent=`正在转写连续监听第 ${chunk.index} 段；剩余 ${liveContinuousQueue.length} 段。`;}else{updateLiveVadNote();}try{await uploadLiveAudioBlob(chunk.blob,chunk.type,'unknown',`continuous-${chunk.index}`,liveContinuousUploadMode());resetAsrPreview();renderLive();}catch(error){$('live-recording-note').textContent=`连续监听第 ${chunk.index} 段最终转写失败；实时草稿仍保留。`;toast(`连续监听第 ${chunk.index} 段转写失败：${error.message}`,true);}}}finally{liveContinuousUploading=false;if(!liveContinuousMode){$('live-recording-note').textContent='连续监听队列已处理完成；请审阅待确认片段。';}else{updateLiveVadNote();}renderLive();}}
async function startLiveContinuousVad(){if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){toast('当前浏览器不支持麦克风连续录制',true);return;}try{liveMediaStream=await navigator.mediaDevices.getUserMedia(microphoneConstraints());liveContinuousMode=true;liveContinuousQueue=[];liveContinuousChunkIndex=0;liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;liveVadHeaderChunk=null;liveVadState='idle';liveVadSpeechStart=0;liveVadLastSpeech=0;liveVadVoicedMs=0;liveVadLastFinalizeAt=0;liveVadFinalizing=false;resetAsrPreview();liveVadAudioContext=new(window.AudioContext||window.webkitAudioContext)();liveVadAnalyser=liveVadAudioContext.createAnalyser();liveVadAnalyser.fftSize=1024;liveVadAnalyser.smoothingTimeConstant=0.3;const source=liveVadAudioContext.createMediaStreamSource(liveMediaStream);source.connect(liveVadAnalyser);liveVadData=new Float32Array(liveVadAnalyser.fftSize);liveVadTimer=setInterval(analyzeLiveVadLevel,VAD_LEVEL_INTERVAL_MS);startContinuousRecorder(liveAudioType());updateLiveVadNote();renderLive();toast('语音分段监听已开始：静音约 0.6 秒自动结束一段');}catch(error){teardownLiveContinuous();toast(`无法启动连续监听：${error.message}`,true);renderLive();}}
function startContinuousRecorder(preferred){if(!liveMediaStream)return;liveVadHeaderChunk=null;liveRecorder=new MediaRecorder(liveMediaStream,preferred?{mimeType:preferred}:undefined);liveRecorder.ondataavailable=event=>{if(!event.data.size)return;if(!liveVadHeaderChunk){liveVadHeaderChunk=event.data;if(!['in-speech','finalizing'].includes(liveVadState))return;}if(!['in-speech','finalizing'].includes(liveVadState))return;if(!liveVadUtteranceChunks.length&&liveVadHeaderChunk!==event.data){liveVadUtteranceChunks.push(liveVadHeaderChunk);liveVadUtteranceBytes+=liveVadHeaderChunk.size;}liveVadUtteranceChunks.push(event.data);liveVadUtteranceBytes+=event.data.size;previewRecordedChunks(liveVadUtteranceChunks,liveRecorder?.mimeType||preferred,showLiveAsrPreview);if(liveVadUtteranceBytes>20*1024*1024&&liveRecorder?.state==='recording'){liveVadState='finalizing';finalizeLiveUtterance();}};liveRecorder.onstop=stopLiveContinuousVad;liveRecorder.start(VAD_TIMESLICE_MS);}
function analyzeLiveVadLevel(){if(!liveVadAnalyser||!liveVadData||!liveContinuousMode)return;if(liveVadAudioContext?.state==='suspended'){liveVadAudioContext.resume();return;}liveVadAnalyser.getFloatTimeDomainData(liveVadData);let sum=0;for(let i=0;i<liveVadData.length;i++)sum+=liveVadData[i]*liveVadData[i];const rms=Math.sqrt(sum/liveVadData.length);const now=performance.now();const speaking=rms>=VAD_SPEECH_RMS;if(liveVadState==='idle'){if(now-liveVadLastFinalizeAt<VAD_GRACE_MS)return;if(speaking){resetAsrPreview();liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;liveVadState='in-speech';liveVadSpeechStart=now;liveVadLastSpeech=now;liveVadVoicedMs=VAD_LEVEL_INTERVAL_MS;}}else if(liveVadState==='in-speech'){if(speaking){liveVadLastSpeech=now;liveVadVoicedMs+=VAD_LEVEL_INTERVAL_MS;}else if(now-liveVadLastSpeech>=VAD_SILENCE_MS){if(liveVadVoicedMs>=VAD_MIN_SPEECH_MS&&liveVadUtteranceChunks.length){liveVadState='finalizing';finalizeLiveUtterance();}else{liveVadState='idle';liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;liveVadSpeechStart=0;liveVadLastSpeech=0;liveVadVoicedMs=0;resetAsrPreview();}}}updateLiveVadNote();}
function finalizeLiveUtterance(){if(liveVadFinalizing)return;liveVadFinalizing=true;liveVadLastFinalizeAt=performance.now();if(liveVadState!=='finalizing')liveVadState='finalizing';if(liveRecorder?.state==='recording')liveRecorder.stop();}
function stopLiveContinuousVad(){const hadSpeech=liveVadVoicedMs>=VAD_MIN_SPEECH_MS&&liveVadUtteranceChunks.length>0;const type=liveRecorder?.mimeType||'audio/webm';if(hadSpeech){const blob=new Blob(liveVadUtteranceChunks,{type});liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;enqueueLiveContinuousChunk(blob);}liveVadFinalizing=false;liveVadState='idle';liveVadSpeechStart=0;liveVadLastSpeech=0;liveVadVoicedMs=0;if(liveContinuousMode){if(liveMediaStream)startContinuousRecorder(liveAudioType());updateLiveVadNote();renderLive();}else{teardownLiveContinuous();$('live-recording-note').textContent=liveContinuousQueue.length||liveContinuousUploading?'连续监听已停止；剩余音频正在转写。':'连续监听已停止。';renderLive();}}
function teardownLiveContinuous(){liveContinuousMode=false;if(liveVadTimer){clearInterval(liveVadTimer);liveVadTimer=null;}if(liveVadAudioContext){liveVadAudioContext.close().catch(()=>{});liveVadAudioContext=null;}liveVadAnalyser=null;liveVadData=null;liveVadState='idle';liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;liveVadHeaderChunk=null;liveVadSpeechStart=0;liveVadLastSpeech=0;liveVadVoicedMs=0;liveVadLastFinalizeAt=0;liveVadFinalizing=false;if(liveMediaStream){liveMediaStream.getTracks().forEach(track=>track.stop());}liveMediaStream=null;liveRecorder=null;}
function liveContinuousUploadMode(){const mode=state.settings?.live_audio?.mode||document.querySelector('#setting-live-audio-mode')?.value||'asr_text';return mode==='audio_direct'?'dialogue':'single';}
function updateLiveVadNote(){const note=$('live-recording-note');if(!note)return;if(liveVadState==='in-speech'){note.textContent=`正在听取 (${Math.max(0,Math.floor((performance.now()-liveVadSpeechStart)/1000))}s)`;note.classList.add('listening');}else{note.textContent='正在聆听…';note.classList.remove('listening');}if(liveContinuousUploading||liveContinuousQueue.length){note.textContent+=` ｜ 队列 ${liveContinuousQueue.length} 段${liveContinuousUploading?'，正在转写 1 段':''}`;}}
window.addEventListener('pagehide',()=>{if(liveRecorder?.state==='recording')liveRecorder.stop();if(sessionRecordingActive||sessionChunks.length){stopSessionRecorder();fetch(`/api/live-interviews/${state.sessionId}/audio/final`,{method:'POST',headers:{Authorization:`Bearer ${state.token}`},body:(()=>{const fd=new FormData();const b=new Blob(sessionChunks,{type:'audio/webm'});sessionChunks=[];fd.append('file',b,`session-${state.sessionId.slice(0,8)}.webm`);return fd;})()}).catch(()=>{});}});

$('live-plan').onclick=async()=>{const button=$('live-plan');busy(button,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/suggestions`,{method:'POST'});state.session=data.state;renderLive();toast('下一问题已准备');}catch(error){toast(error.message,true)}finally{busy(button,false)}};
$('live-export').onclick=async()=>{if(!state.sessionId){toast('请先创建会话',true);return;}try{const resp=await fetch(`/api/interviews/sessions/${state.sessionId}/transcript`,{headers:{Authorization:`Bearer ${state.token}`}});if(!resp.ok)throw new Error('导出失败');const text=await resp.text();const blob=new Blob([text],{type:'text/plain;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`interview-${state.sessionId.slice(0,8)}.txt`;document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);toast('转写文本已导出');}catch(error){toast(error.message,true)}};
$('session-audio-download')?.addEventListener('click',async()=>{if(!state.sessionId)return;try{const resp=await fetch(`/api/live-interviews/${state.sessionId}/audio`,{headers:{Authorization:`Bearer ${state.token}`}});if(!resp.ok)throw new Error('下载失败');const blob=await resp.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`interview-${state.sessionId.slice(0,8)}.wav`;document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);toast('全场录音已下载');}catch(error){toast(error.message,true)}});

let liveSuggestionStream = null;
let liveSuggestionAbort = null;
async function streamNextSuggestion() {
  if (!state.sessionId || liveSuggestionStream) return;
  liveSuggestionStream = true;
  liveSuggestionAbort = new AbortController();
  const node = $('live-suggestion');
  if (node) {
    node.className = 'suggestion-card streaming';
    node.innerHTML = '<p class="streaming-hint">AI 正在生成下一问…</p><div id="streaming-text" class="streaming-text"></div>';
  }
  try {
    const resp = await fetch(`/api/live-interviews/${state.sessionId}/suggestions/stream`, {method:'POST', headers:state.token?{Authorization:`Bearer ${state.token}`}:{}, signal: liveSuggestionAbort.signal});
    if (!resp.ok) throw new Error((await resp.json().catch(()=>({}))).detail || '流式建议失败');
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    const box = $('streaming-text');
    let buffer = '';
    let full = '';
    let finished = false;
    while (!finished) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      finished = done;
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const event = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
        for (const line of event.split('\n')) {
          if (line.startsWith('data: ')) {
            const encoded = line.slice(6);
            let event; try { event = JSON.parse(encoded); } catch { event = {type:'append',text:encoded}; }
            if (typeof event === 'string') full += event;
            else if (event.type === 'replace') full = event.text || '';
            else full += event.text || '';
            if (box) box.textContent = full;
          }
        }
      }
      if (done) break;
    }
    // Flush any trailing buffer (partial final event).
    if (buffer.trim()) {
      for (const line of buffer.trim().split('\n')) {
        if (line.startsWith('data: ')) { const encoded=line.slice(6);try{const event=JSON.parse(encoded);if(typeof event==='string')full+=event;else if(event.type==='replace')full=event.text||'';else full+=event.text||''}catch{full+=encoded} }
      }
      if (box) box.textContent = full;
    }
    // Directly refresh state, bypassing the poll's busy guard.
    try {
      const data = await api(`/api/live-interviews/${state.sessionId}`);
      state.session = data.state;
      renderLive();
    } catch { /* ignore refresh errors */ }
    if (node) node.classList.remove('streaming');
    toast('下一问已流式生成');
  } catch (error) {
    if (error.name !== 'AbortError') {
      toast(error.message, true);
      try {
        const data = await api(`/api/live-interviews/${state.sessionId}`);
        state.session = data.state; renderLive();
      } catch { /* ignore */ }
    }
  } finally {
    liveSuggestionStream = null;
    liveSuggestionAbort = null;
  }
}
$('live-stream-plan')?.addEventListener('click', streamNextSuggestion);
$('live-confirm-all').onclick=async()=>{const competency=window.prompt('批量确认使用的能力维度；留空则由系统按问题推断','')||'';const button=$('live-confirm-all');busy(button,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/evidence/batch`,{method:'POST',body:JSON.stringify({competency})});state.session=data.state;renderAll();toast('待确认候选人回答已批量归档');}catch(error){toast(error.message,true)}finally{busy(button,false)}};
$('live-confirm-merged').onclick=async()=>{const recordedSegmentIds=new Set((state.session?.live_interview_records||[]).flatMap(record=>record.transcript_segment_ids||[]));const pending=(state.session?.live_interview?.segments||[]).filter(segment=>segment.speaker==='candidate'&&!recordedSegmentIds.has(segment.id));if(pending.length<2){toast('至少需要两段候选人发言才能合并确认',true);return;}const defaultSequences=pending.map(segment=>segment.sequence).join(',');const selected=window.prompt('输入要合并的候选人片段序号，用逗号分隔',defaultSequences)||'';const wanted=new Set(selected.split(',').map(value=>Number(value.trim())).filter(Number.isFinite));const segmentIds=pending.filter(segment=>wanted.has(segment.sequence)).map(segment=>segment.id);if(segmentIds.length<2){toast('请选择至少两段候选人发言',true);return;}const defaultCompetency=state.session?.job?.competencies?.[0]||'综合能力';const competency=window.prompt('合并后的回答对应哪个能力维度？',defaultCompetency)||'';if(!competency.trim())return;const questionSegmentId=nearestLiveQuestionSegmentId(segmentIds[0]);const button=$('live-confirm-merged');busy(button,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/evidence/merge`,{method:'POST',body:JSON.stringify({segment_ids:segmentIds,question_segment_id:questionSegmentId,competency})});state.session=data.state;renderAll();toast(`已合并 ${segmentIds.length} 段候选人回答为一条证据`);}catch(error){toast(error.message,true)}finally{busy(button,false)}};
async function confirmBoundarySuggestion(suggestion, button = null) {if(!suggestion)return;const competency=window.prompt('合并后的回答对应哪个能力维度？',suggestion.suggested_competency||'综合能力')||'';if(!competency.trim())return;if(button)busy(button,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/evidence/merge`,{method:'POST',body:JSON.stringify({segment_ids:suggestion.answer_segment_ids||[],question_segment_id:suggestion.question_segment_id||null,competency})});state.session=data.state;renderAll();toast(`已按边界建议合并 ${(suggestion.answer_segment_ids||[]).length} 段回答`);}catch(error){toast(error.message,true)}finally{if(button)busy(button,false)}}
document.addEventListener('click',async event=>{const button=event.target.closest('[data-live-boundary-index]');if(!button)return;const suggestion=state.session?.live_interview?.answer_boundary_suggestions?.[Number(button.dataset.liveBoundaryIndex)];await confirmBoundarySuggestion(suggestion,button);});
document.addEventListener('click',async event=>{const button=event.target.closest('[data-live-decision]');if(!button)return;let finalQuestion='';if(button.dataset.liveDecision==='edited'){const suggestion=state.session?.live_interview?.suggestions?.find(item=>item.id===button.dataset.suggestionId);finalQuestion=window.prompt('编辑面试问题',suggestion?.suggested_question||'')||'';if(!finalQuestion)return;}try{const data=await api(`/api/live-interviews/${state.sessionId}/suggestions/${button.dataset.suggestionId}`,{method:'PATCH',body:JSON.stringify({status:button.dataset.liveDecision,final_question:finalQuestion})});state.session=data.state;renderLive();toast(button.dataset.liveDecision==='skipped'?'已跳过建议':'已加入面试官问题');}catch(error){toast(error.message,true)}});
document.addEventListener('click',async event=>{const button=event.target.closest('[data-live-edit]');if(!button)return;const segment=state.session?.live_interview?.segments?.find(item=>item.id===button.dataset.liveEdit);if(!segment)return;const text=window.prompt('修改转写文本',segment.text)||'';if(!text.trim())return;const speaker=window.prompt('说话人：candidate / interviewer / unknown',segment.speaker)||segment.speaker;if(!['candidate','interviewer','unknown'].includes(speaker)){toast('说话人只能是 candidate、interviewer 或 unknown',true);return;}try{const data=await api(`/api/live-interviews/${state.sessionId}/segments/${button.dataset.liveEdit}`,{method:'PATCH',body:JSON.stringify({text,speaker})});state.session=data.state;renderLive();toast('转写片段已更新');}catch(error){toast(error.message,true)}});
async function confirmLiveSegment(segmentId, button = null) {const defaultCompetency=state.session?.job?.competencies?.[0]||state.session?.mock_interview?.questions?.[0]?.competency||'综合能力';const competency=window.prompt('这条回答对应哪个能力维度？',defaultCompetency)||'';if(!competency.trim())return;const questionSegmentId=nearestLiveQuestionSegmentId(segmentId);if(button)busy(button,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/segments/${segmentId}/evidence`,{method:'POST',body:JSON.stringify({question_segment_id:questionSegmentId,competency})});state.session=data.state;renderAll();toast('候选人回答已确认为证据');}catch(error){toast(error.message,true)}finally{if(button)busy(button,false)}}
document.addEventListener('click',async event=>{const button=event.target.closest('[data-live-confirm]');if(!button)return;await confirmLiveSegment(button.dataset.liveConfirm,button);});
async function runLiveAction(kind, button) {
  const action = state.session?.live_interview?.action_card?.action_type || 'start';
  const primary = kind === 'primary';
  if (action === 'start') { $('live-start')?.click(); return; }
  if (action === 'resume') { primary ? $('live-pause')?.click() : spotlight($('live-review'), '先审阅已产生的转写和证据'); return; }
  if (action === 'review_speaker') {
    const unknown = (state.session?.live_interview?.segments || []).find(segment => segment.speaker === 'unknown');
    spotlight(unknown ? document.querySelector(`[data-segment-id="${cssEscape(unknown.id)}"]`) : $('live-transcript'), '请先修改待确认说话人与文本');
    return;
  }
  if (action === 'merge_boundary') {
    if (primary) { await confirmBoundarySuggestion(strongestBoundarySuggestion(), button); return; }
    spotlight($('live-review'), '也可以逐条确认候选人回答');
    return;
  }
  if (action === 'confirm_evidence') {
    if (primary) { const segment = firstPendingCandidateSegment(); segment ? await confirmLiveSegment(segment.id, button) : toast('暂无待确认候选人回答', true); return; }
    $('live-plan')?.click();
    return;
  }
  if (action === 'decide_question') {
    spotlight($('live-suggestion'), '请采用、编辑或跳过这条问题建议');
    return;
  }
  if (action === 'plan_gap_question') {
    if (primary) { $('live-plan')?.click(); return; }
    if ((state.session?.live_interview?.action_card?.secondary_cta || '').includes('JD')) { setView('enterprise'); return; }
    spotlight($('live-competencies'), '根据能力缺口继续追问');
    return;
  }
  if (action === 'evaluate' || action === 'ready_to_evaluate') {
    if (primary) { setView('evaluation'); await generateEvaluation(button); return; }
    $('live-plan')?.click();
    return;
  }
  if (action === 'review_evidence') { spotlight($('live-review'), '请先补齐可用于评价的证据'); return; }
  spotlight($('live-action-card'), '当前动作需要人工确认');
}
document.addEventListener('click',async event=>{const button=event.target.closest('[data-live-action]');if(!button)return;await runLiveAction(button.dataset.liveAction,button);});
document.addEventListener('click',async event=>{const button=event.target.closest('[data-live-reevaluate]');if(!button)return;const record=state.session?.live_interview_records?.find(item=>item.id===button.dataset.liveReevaluate);if(!record)return;const question=window.prompt('重评使用的问题文本',record.question)||'';if(!question.trim())return;const competency=window.prompt('重评使用的能力维度',record.competency)||'';if(!competency.trim())return;try{const data=await api(`/api/live-interviews/${state.sessionId}/evidence/${record.id}/reevaluate`,{method:'PATCH',body:JSON.stringify({question,competency})});state.session=data.state;renderAll();toast('live 证据已重新评分');}catch(error){toast(error.message,true)}});
document.addEventListener('click',async event=>{const button=event.target.closest('[data-live-revoke]');if(!button)return;const recordId=button.dataset.liveRevoke;if(!recordId)return;if(!window.confirm('撤销这条已归档证据？转写片段会保留，可修改后重新确认。'))return;try{const data=await api(`/api/live-interviews/${state.sessionId}/evidence/${recordId}`,{method:'DELETE'});state.session=data.state;renderAll();toast('已撤销 live 证据，转写片段仍保留');}catch(error){toast(error.message,true)}});

async function loadSettings(){try{const s=await api('/api/settings');state.settings=s;$('setting-provider').value=s.search.selected;$('setting-base-url').value=s.llm.base_url||'';$('setting-model').value=s.llm.model||'';$('setting-input-cost').value=s.llm.input_cost_per_million||0;$('setting-output-cost').value=s.llm.output_cost_per_million||0;$('setting-search-cost').value=s.search.search_request_cost_usd||0;$('setting-asr-url').value=s.asr?.base_url||'';$('setting-asr-path').value=s.asr?.transcription_path||'/v1/audio/transcriptions';$('setting-asr-model').value=s.asr?.model||'whisper-1';$('setting-asr-timeout').value=s.asr?.timeout_seconds||90;const la=s.live_audio||{};const liveAudioMode=document.querySelector('#setting-live-audio-mode');if(liveAudioMode&&s.live_audio!==undefined){liveAudioMode.value=s.live_audio.mode||'asr_text';}
if($('setting-live-audio-name'))$('setting-live-audio-name').value=la.name||'音频直连';
if($('setting-live-audio-url'))$('setting-live-audio-url').value=la.base_url||'http://192.168.1.97:8004/v1';
if($('setting-live-audio-model'))$('setting-live-audio-model').value=la.model||'';
const tts=s.tts||{};if($('setting-tts-url'))$('setting-tts-url').value=tts.base_url||'http://192.168.1.97:8002/v1';if($('setting-tts-path'))$('setting-tts-path').value=tts.speech_path||'/audio/speech';if($('setting-tts-model'))$('setting-tts-model').value=tts.model||'qwen3-tts';if($('setting-tts-voice'))$('setting-tts-voice').value=tts.voice||'温和、专业、清晰的中文声音';
const rl=s.resume_llm||{};if($('setting-resume-llm-url'))$('setting-resume-llm-url').value=rl.base_url||'http://192.168.1.97:8004/v1';if($('setting-resume-llm-model'))$('setting-resume-llm-model').value=rl.model||'';
renderLiveAudioCapability(la.capability);}catch(error){toast(error.message,true)}}
function renderLiveAudioCapability(cap){
  const node=$('live-audio-capability');if(!node)return;
  if(!cap){node.textContent='尚未探测能力。';return;}
  node.textContent=cap.ok?`能力可用：${cap.latency_ms}ms${cap.sample?` · 样例「${cap.sample}」`:''}`:`能力不可用：${cap.error||'端点未返回可用建议'}（${cap.latency_ms}ms）`;
  node.className=cap.ok?'security-note ok':'security-note error';
}
$('probe-live-audio').onclick=async()=>{const btn=$('probe-live-audio');btn.disabled=true;try{const r=await api(`/api/settings/probe-live-audio`,{method:'POST'});renderLiveAudioCapability(r.capability);toast(r.capability.ok?'音频直连能力探测成功':'音频直连不可用',!r.capability.ok);}catch(error){toast(error.message,true)}finally{btn.disabled=false}};
$('settings-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;busy(form,true);try{await api('/api/settings',{method:'PUT',body:JSON.stringify({search:{provider:$('setting-provider').value,tavily_api_key:optional('setting-tavily'),searxng_base_url:optional('setting-searxng'),brave_api_key:optional('setting-brave'),search_request_cost_usd:Number($('setting-search-cost').value)||0},llm:{base_url:optional('setting-base-url'),model:optional('setting-model'),api_key:optional('setting-llm-key'),input_cost_per_million:Number($('setting-input-cost').value)||0,output_cost_per_million:Number($('setting-output-cost').value)||0},asr:{base_url:optional('setting-asr-url'),transcription_path:optional('setting-asr-path'),model:optional('setting-asr-model'),timeout_seconds:Number($('setting-asr-timeout').value)||90,api_key:optional('setting-asr-key')},tts:{base_url:optional('setting-tts-url'),speech_path:optional('setting-tts-path'),model:optional('setting-tts-model'),voice:optional('setting-tts-voice'),api_key:optional('setting-tts-key')},live_audio:{mode:$('setting-live-audio-mode').value,name:optional('setting-live-audio-name'),base_url:optional('setting-live-audio-url'),model:optional('setting-live-audio-model')},resume_llm:{base_url:optional('setting-resume-llm-url'),model:optional('setting-resume-llm-model'),api_key:optional('setting-resume-llm-key')}})});toast('设置已持久化并立即应用');await loadSettings();}catch(error){toast(error.message,true)}finally{busy(form,false)}};

async function refreshDebug(){try{const [status,events,sessions]=await Promise.all([api('/api/debug/status'),api('/api/debug/events?limit=100'),api('/api/debug/sessions')]);const metrics=status.llm.metrics||{};const cache=status.search.cache||{};$('debug-status').innerHTML=[['应用',status.application.status],['模型',status.llm.model||'managed'],['模型地址',status.llm.base_url||'—'],['ASR',status.asr?.model||'—'],['ASR 地址',status.asr?.base_url||'—'],['TTS',status.tts?.model||'—'],['TTS 地址',status.tts?.base_url||'—'],['模型请求 / 失败',`${metrics.requests||0} / ${metrics.failures||0}`],['Token 输入 / 输出',`${metrics.prompt_tokens||0} / ${metrics.completion_tokens||0}`],['模型累计成本',`$${metrics.estimated_cost_usd||0}`],['平均耗时',`${metrics.average_latency_ms||0} ms`],['搜索',status.search.selected],['搜索请求 / 成本',`${status.search.provider_requests||0} / $${status.search.estimated_cost_usd||0}`],['缓存命中 / 未命中',`${cache.hits||0} / ${cache.misses||0}`],['缓存容量',`${cache.size||0} / ${cache.capacity||0}`],['来源筛选',status.search.source_filter_policy||'实体精确匹配 / 可信别名'],['持久缓存',cache.persistent?'已启用':'仅运行时'],['事件持久化',status.events_persistent?'已启用':'仅运行时'],['事件容量',status.event_capacity]].map(([k,v])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');$('debug-sessions').innerHTML=sessions.sessions.map(s=>`<div class="session-row"><span>${esc(s.candidate_name||'未命名')}<small>${esc(s.job_title||'未指定岗位')}</small></span><code>${esc(s.id.slice(0,8))}</code></div>`).join('')||'<div class="empty-state">暂无会话</div>';renderEvents(events.events);}catch(error){toast(error.message,true)}}
function renderEvents(events){$('event-list').innerHTML=events.map(e=>`<div class="event-row"><span>${new Date(e.timestamp).toLocaleTimeString()}</span><span class="${esc(e.level)}">${esc(e.level)}</span><span>${esc(e.agent||e.category)} · ${esc(e.action)}${e.detail?` · ${esc(e.detail)}`:''}</span><span>${e.duration_ms?`${e.duration_ms} ms`:'—'}</span></div>`).join('')||'<div class="empty-state">暂无运行事件</div>';}
$('refresh-events').onclick=refreshDebug;$('probe-llm').onclick=async()=>{try{const d=await api('/api/debug/probes/llm',{method:'POST'});toast(`模型连接正常 · ${d.latency_ms||0} ms`);await refreshDebug();}catch(error){toast(error.message,true)}};

async function refreshLive() {
  if (!state.sessionId || !state.session?.live_interview) return;
  if (refreshLive.busy) return;
  refreshLive.busy = true;
  try {
    const data = await api(`/api/live-interviews/${state.sessionId}`);
    state.session = data.state;
    renderLive();
  } catch { /* transient poll failure is fine; next tick retries */ }
  finally { refreshLive.busy = false; }
}

setInterval(()=>{if(state.view==='debug')refreshDebug()},4000);
setInterval(()=>{if(state.view==='live')refreshLive()},3000);
const requestedView = (location.hash || '').slice(1);
if (navigation.interviewer.some(([view]) => view === requestedView)) state.role = 'interviewer';
if (navigation.candidate.some(([view]) => view === requestedView)) state.role = 'candidate';
setView(viewMeta[requestedView] ? requestedView : `${state.role}-home`);
checkHealth();
async function restoreAuthenticatedSession() {
  try {
    const user = await api('/api/auth/me');
    $('user-chip').textContent = user.username;
    await loadSessions();
  } catch (error) { toast(error.message, true); }
}
if (!state.token) { showLogin(); } else { restoreAuthenticatedSession(); }
