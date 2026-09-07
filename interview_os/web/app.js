import {createApiClient} from './modules/api.js';
import {loadRuntimeConfig} from './modules/runtime.js';
import {
  renderBlueprint,
  renderFacts,
  renderJDReview,
  renderReports,
  renderResumeReview,
  renderSources,
  renderStrategy
} from './modules/candidate-view.js';
import {renderLiveView} from './modules/live-view.js';
import {
  renderQuestionAlignment,
  renderQuestionDeepAnalysis,
  renderRetryComparison,
  renderSpokenAnswerAnalysis
} from './modules/mock-view.js';
import {
  candidateHomeAction,
  globalViews,
  localizedNextAction,
  navigation,
  state,
  viewMeta
} from './modules/state.js';
import {
  $,
  busy,
  cssEscape,
  esc,
  formatDateTime,
  list,
  optional,
  safeUrl,
  tags,
  toast
} from './modules/ui.js';

const runtime = await loadRuntimeConfig().catch(error => {
  const panel = document.createElement('main');
  const title = document.createElement('h1');
  const detail = document.createElement('p');
  title.textContent = 'InterviewOS 本地服务启动失败';
  detail.textContent = '请退出后重新打开应用；如果问题持续存在，请重新安装或联系维护人员。';
  panel.className = 'startup-error';
  panel.append(title, detail);
  document.body.replaceChildren(panel);
  throw error;
});

const api = createApiClient({
  state,
  runtime,
  onUnauthorized: handleUnauthorized
});
let desktopShutdownAbort = null;
let desktopShutdownBridgeReady = runtime.mode !== 'desktop';
let desktopShutdownBridgeInstalling = null;
let desktopShutdownBridgeError = '';

async function boundedApi(path,options,timeoutMs,timeoutMessage){
  const controller=new AbortController();
  let timedOut=false;
  const externalSignal=options?.signal;
  const abortFromExternal=()=>controller.abort(externalSignal?.reason);
  if(externalSignal?.aborted)abortFromExternal();else externalSignal?.addEventListener('abort',abortFromExternal,{once:true});
  const timer=setTimeout(()=>{timedOut=true;controller.abort();},timeoutMs);
  try{return await api(path,{...options,signal:controller.signal});}
  catch(error){
    if(error.name==='AbortError'&&timedOut){
      const timeoutError=new Error(timeoutMessage);
      timeoutError.requestTimedOut=true;
      throw timeoutError;
    }
    throw error;
  }finally{clearTimeout(timer);externalSignal?.removeEventListener('abort',abortFromExternal);}
}
function mediaRequestTimeoutMs(){const configured=Number(state.settings?.asr?.timeout_seconds)||90;return Math.min(615000,Math.max(45000,configured*1000+15000));}
function mediaApi(path,options={}){return boundedApi(path,options,mediaRequestTimeoutMs(),'音频处理超时，原始录音已保留，可稍后重试');}
async function waitForCaptureStartup(promise,timeoutMs=3000){let timer;try{return await Promise.race([Promise.resolve(promise),new Promise(resolve=>{timer=setTimeout(()=>resolve(false),timeoutMs);})]);}finally{clearTimeout(timer);}}

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
let mockVoiceStartPending = false;
let mockVoiceStartPromise = Promise.resolve();
let mockVoiceRecorderDone = Promise.resolve();
let mockVoiceFinalizePromise = null;
let mockVoiceContext = null;
let mockVoicePendingUpload = null;
let mockRestoredDraftId = '';
let mockAnswerEditRevision = 0;
let mockSpeechFeedbackPollGeneration = 0;
let mockVoiceWarningTimer = null;
let mockVoiceLimitTimer = null;
let mockMutationInProgress = false;
let activeLoadedSessionId = '';
let liveRecorder = null;
let liveAudioChunks = [];
let liveMediaStream = null;
let liveRecordingSessionId = '';
let liveRecordingId = '';
let liveRecordingMimeType = 'audio/webm';
let liveRecordingSpeaker = 'unknown';
let liveRecordingMode = 'single';
let liveRecordingSettingsSnapshot = null;
let liveRecordingRolloverTimer = null;
let liveRecordingWarningTimer = null;
let liveRecordingLimitReachedId = '';
let liveCaptureStartPending = false;
let liveCaptureGeneration = 0;
let pendingLiveCaptureStart = null;
let liveRecorderDone = Promise.resolve();
let liveContinuousMode = false;
let liveContinuousResumeSessionId = '';
let liveContinuousQueue = [];
let liveContinuousUploading = false;
let liveContinuousChunkIndex = 0;
let liveContinuousSessionId = '';
let liveContinuousGuardId = '';
let liveContinuousSettingsSnapshot = null;
const ASR_PREVIEW_INTERVAL_MS = 2500;
// 16 kHz mono PCM is roughly 1.92 MB/minute. Rolling at ten minutes keeps the
// converted upload comfortably below the backend's 25 MB per-part limit.
const LIVE_RECORDING_ROLLOVER_MS = 10 * 60 * 1000;
const LIVE_RECORDING_WARNING_MS = 9 * 60 * 1000;
const MOCK_RECORDING_LIMIT_MS = 10 * 60 * 1000;
const MOCK_RECORDING_WARNING_MS = 9 * 60 * 1000;
const MOCK_AUDIO_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024;
const ASR_PREVIEW_WINDOW_MS = 90 * 1000;
let asrPreviewBusy = false;
let asrPreviewLastAt = 0;
let asrPreviewSequence = 0;
let asrPreviewFailureNotified = false;
let asrPreviewAbort = null;
// VAD: silence-based utterance chunking for continuous listening.
const VAD_SPEECH_RMS = 0.02;        // RMS above this counts as speech
const VAD_SILENCE_MS = 600;         // silence of this length ends an utterance
const VAD_MIN_SPEECH_MS = 700;      // ignore brief noise/filler triggers
const VAD_GRACE_MS = 300;           // ignore speech briefly after a finalize
const VAD_TIMESLICE_MS = 250;       // MediaRecorder timeslice for continuous mode
const VAD_LEVEL_INTERVAL_MS = 100;  // indicator/level loop cadence
const LIVE_CONTINUOUS_QUEUE_HIGH_WATER = 6;
let liveVadAudioContext = null;
let liveVadAnalyser = null;
let liveVadData = null;
let liveVadState = 'idle';          // idle | in-speech | finalizing
let liveVadSpeechStart = 0;
let liveVadLastSpeech = 0;
let liveVadVoicedMs = 0;
let liveVadLastAnalysisAt = 0;
let liveVadLastFinalizeAt = 0;
let liveVadUtteranceChunks = [];
let liveVadUtteranceBytes = 0;
let liveVadHeaderChunk = null;
let liveVadRecordingId = '';
let liveVadCaptureRegistration = null;
let liveVadFinalizing = false;
let liveVadTimer = null;
// Whole-session recorder: captures the entire live interview as one audio file
// (independent of the per-utterance VAD path, which discards silence gaps).
// Parts rotate while reusing one MediaStream so long interviews stay bounded.
const SESSION_RECORDING_PART_MS = 5 * 60 * 1000;
const SESSION_RECORDING_PART_BYTES = 64 * 1024 * 1024;
const SESSION_PART_QUEUE_HIGH_WATER = 3;
const SESSION_PART_QUEUE_BYTES_HIGH_WATER = 192 * 1024 * 1024;
let sessionRecorder = null;
let sessionChunks = [];
let sessionRecordingActive = false;
let sessionRecorderDone = Promise.resolve();
let sessionRecordingMimeType = 'audio/webm';
let sessionRecordingSessionId = '';
let sessionRecordingPartId = '';
let sessionRecordingArchiveRevision = 0;
let sessionRecordingCaptureEpoch = 0;
let sessionRecorderStartGeneration = 0;
let sessionRecorderStartPending = false;
let sessionRecorderStartSessionId = '';
let sessionRecorderStartCaptureEpoch = 0;
let sessionRecorderStartPromise = Promise.resolve();
let sessionPendingMediaStream = null;
let sessionMediaStream = null;
let sessionCurrentPart = null;
let sessionPartQueue = [];
let sessionPartSequence = 0;
let sessionNextUploadSequence = 1;
let sessionFinalizedPartSequences = new Set();
let sessionRotationTimer = null;
let sessionRotationEnabled = false;
let sessionRotationInProgress = false;
let sessionRotationPromise = Promise.resolve();
let sessionStopInProgress = false;
let sessionStopPromise = Promise.resolve();
let sessionPartUploadsSuppressed = false;
let sessionAudioUploadPromise = Promise.resolve();
let sessionAudioUploading = false;
let sessionAudioSaveError = '';
let sessionAudioErrorSessionId = '';
let sessionRecordingError = '';
let sessionRecordingErrorSessionId = '';
let liveRecordingUploadPromise = Promise.resolve();
let liveRecordingUploading = false;
let desktopShutdownPreparing = false;
let mediaTransitionInProgress = false;
let mediaTransitionPromise = Promise.resolve();
let mediaFlushInProgress = false;
let mediaFlushTail = Promise.resolve();
let liveRefreshAbort = null;
let liveRefreshGeneration = 0;
let liveStatusUncertainSessionId = '';
let liveStatusUncertainExpectedGeneration = null;
let liveStatusUncertainAlternateGeneration = null;
const persistedPauseRequired = readPersistedPauseRequiredSession();
let livePauseRequiredSessionId = persistedPauseRequired.sessionId;
let livePauseRequiredCaptureEpoch = persistedPauseRequired.captureEpoch;
let livePauseRequiredStatusRevision = persistedPauseRequired.statusRevision;
let pageBoundaryRecoveryInProgress = false;
let visibilityPauseRequested = false;
let visibilityPausePromise = Promise.resolve();
let authenticationSuspended = false;
let authenticationRecoverySessionId = '';
let authenticationRecoveryEditors = null;
let authenticationRecoveryUsername = '';
let runtimeSettingsReady = false;
let settingsLoadGeneration = 0;
let staleAudioSettingsRefresh = null;
function readPersistedPauseRequiredSession(){try{const value=JSON.parse(sessionStorage.getItem('interviewos.pauseRequiredSession')||'{}');return{sessionId:typeof value.sessionId==='string'?value.sessionId:'',captureEpoch:Number.isInteger(value.captureEpoch)?value.captureEpoch:null,statusRevision:Number.isInteger(value.statusRevision)?value.statusRevision:null};}catch{return{sessionId:'',captureEpoch:null,statusRevision:null};}}
function markLivePauseRequired(sessionId,captureEpoch=activeLocalCaptureEpoch(sessionId),statusRevision=Number(state.session?.live_interview?.status_revision)){if(!sessionId)return;livePauseRequiredSessionId=sessionId;livePauseRequiredCaptureEpoch=Number.isInteger(captureEpoch)?captureEpoch:Number(state.session?.live_interview?.capture_epoch);livePauseRequiredStatusRevision=Number.isInteger(statusRevision)?statusRevision:null;try{sessionStorage.setItem('interviewos.pauseRequiredSession',JSON.stringify({sessionId,captureEpoch:livePauseRequiredCaptureEpoch,statusRevision:livePauseRequiredStatusRevision}));}catch{}}
function clearLivePauseRequired(sessionId){if(livePauseRequiredSessionId!==sessionId)return;livePauseRequiredSessionId='';livePauseRequiredCaptureEpoch=null;livePauseRequiredStatusRevision=null;try{sessionStorage.removeItem('interviewos.pauseRequiredSession');}catch{}}
function markLiveStatusUncertain(sessionId,expectedGeneration,alternateGeneration=null){
  if(!sessionId||!expectedGeneration)throw new Error('缺少实时会话代际，拒绝暂停未知的监听轮次');
  liveStatusUncertainSessionId=sessionId;
  liveStatusUncertainExpectedGeneration={...expectedGeneration};
  liveStatusUncertainAlternateGeneration=alternateGeneration?{...alternateGeneration}:null;
}
function clearLiveStatusUncertain(sessionId){
  if(liveStatusUncertainSessionId!==sessionId)return;
  liveStatusUncertainSessionId='';
  liveStatusUncertainExpectedGeneration=null;
  liveStatusUncertainAlternateGeneration=null;
}
function mockVoiceCaptureActive(){return Boolean(mockVoiceStartPending||mockVoiceRecorder||mockVoiceFinalizePromise);}
function mockVoiceCaptureBusy(){return Boolean(mockVoiceCaptureActive()||mockVoicePendingUpload);}
function liveMicrophoneCaptureActive(){return Boolean(sessionRecorderStartPending||sessionPendingMediaStream||sessionMediaStream||sessionRecorder||sessionRotationInProgress||pendingLiveCaptureStart||liveCaptureStartPending||liveRecorder||liveMediaStream||liveContinuousMode||liveVadFinalizing);}
function microphoneAvailabilityMessage() {
  if(runtime.mode==='desktop'&&!desktopShutdownBridgeReady)return desktopShutdownBridgeError||'桌面应用的安全关闭保护尚未就绪；请重试或重启应用。';
  if (!window.isSecureContext) return '当前是非安全的局域网 HTTP 页面，浏览器会阻止麦克风。请改用 HTTPS 地址后重试；文字输入仍可使用。';
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) return '当前浏览器不支持麦克风录制，请使用最新版 Chrome、Edge 或 Safari，或改用文字输入。';
  return '';
}

function microphoneConstraints() {
  return {audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1}};
}

function capturePageIsVisible(){return document.visibilityState==='visible';}

function ensureMicrophoneAvailable() {
  const message = microphoneAvailabilityMessage();
  if (!message) return true;
  if(runtime.mode==='desktop'&&!desktopShutdownBridgeReady)void initializeDesktopShutdownBridge();
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
function renderNavigation() {
  $('nav').innerHTML = navigation[state.role].map(([view, label], index) =>
    `<button class="nav-item${state.view === view ? ' active' : ''}" data-view="${view}"><span>0${index + 1}</span>${label}</button>`
  ).join('');
  document.querySelectorAll('[data-role]').forEach(button => button.classList.toggle('active', button.dataset.role === state.role));
  document.querySelectorAll('[data-global-view]').forEach(button => button.classList.toggle('active', button.dataset.globalView === state.view));
}

function setRole(role) {
  if (!navigation[role]) return;
  if(mockVoiceCaptureBusy()||liveMicrophoneCaptureActive()){
    toast('请先停止并保存当前录音，再切换角色',true);
    return;
  }
  state.role = role;
  document.documentElement.dataset.role = role;
  localStorage.setItem('interviewos.role', role);
  setView(`${role}-home`);
}

function setView(view) {
  const allowed = navigation[state.role].some(([name]) => name === view) || globalViews.has(view);
  if (!allowed) view = `${state.role}-home`;
  if(view!=='mock'&&state.view==='mock'&&mockVoiceCaptureBusy()){
    toast('请先停止并保存当前模拟回答录音，再离开模拟面试',true);
    return;
  }
  if(view==='mock'&&liveMicrophoneCaptureActive()){
    toast('实时面试仍在收音，请先暂停并保存录音，再进入模拟面试',true);
    return;
  }
  if(view!=='live'&&state.view==='live'&&liveMicrophoneCaptureActive()){
    toast('实时面试仍在收音，请先暂停并保存录音，再离开当前页面',true);
    return;
  }
  if(view==='live'&&mockVoiceCaptureBusy()){
    toast('模拟回答仍在录制或保存，请先完成后再进入实时面试',true);
    return;
  }
  if (view !== 'live' && liveSuggestionAbort) liveSuggestionAbort.abort();
  if (view !== 'mock') { cancelMockQuestionSpeech(); clearMockQuestionAudio(); pauseMockAnswerPlayback(); }
  state.view = view;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${view}`));
  [$('view-eyebrow').textContent, $('view-title').textContent] = viewMeta[view];
  renderNavigation();
  location.hash = view;
  if (view === 'debug') refreshDebug();
  if (view === 'settings') void loadSettings();
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
    else {
      state.sessionId = '';
      state.session = null;
      activeLoadedSessionId = '';
      localStorage.removeItem('interviewos.session');
      select.value = '';
      hydrateSessionForms();
      clearSessionEditors();
      renderState();
    }
    if (state.sessionId) await loadSession();
  } catch (error) { toast(error.message, true); throw error; }
}

function showLogin() {
  const dialog = $('login-dialog'); if (!dialog) return;
  dialog.showModal();
}

function hideLogin() { $('login-dialog')?.close(); }

function stopCaptureAfterAuthenticationLoss(){
  const interruptedSessionId=state.sessionId;
  const uncertainGeneration=liveStatusUncertainSessionId===interruptedSessionId?liveStatusUncertainExpectedGeneration:null;
  if(interruptedSessionId&&hasLocalLiveCapture(interruptedSessionId))markLivePauseRequired(interruptedSessionId);
  else if(interruptedSessionId&&uncertainGeneration)markLivePauseRequired(interruptedSessionId,uncertainGeneration.captureEpoch,uncertainGeneration.statusRevision);
  authenticationSuspended=true;
  liveCaptureGeneration+=1;
  sessionRecorderStartGeneration+=1;
  liveCaptureStartPending=false;
  sessionRecorderStartPending=false;
  sessionRecorderStartSessionId='';
  sessionRecorderStartCaptureEpoch=0;
  if(pendingLiveCaptureStart){
    const pending=pendingLiveCaptureStart;
    pendingLiveCaptureStart=null;
    void cancelLiveAudioCapture(pending.sessionId,pending.recordingId).catch(()=>{});
  }
  if(mockVoiceStartPending&&!mockVoiceRecorder)mockVoiceGeneration+=1;
  mockVoiceStartPending=false;
  clearMockVoiceRecordingTimers();
  if(mockVoiceRecorder?.state==='recording'){
    try{mockVoiceRecorder.requestData();}catch{}
    try{mockVoiceRecorder.stop();}catch{}
  }
  mockVoiceStream?.getTracks().forEach(track=>track.stop());
  if(liveContinuousMode)liveContinuousMode=false;
  if(liveRecorder){
    if(liveRecorder.state==='recording'){try{liveRecorder.stop();}catch{}}
  }else if(liveMediaStream){
    liveMediaStream.getTracks().forEach(track=>track.stop());
    liveMediaStream=null;
    liveRecordingSessionId='';
    liveRecordingId='';
  }
  requestSessionRecorderStop('authentication');
  sessionPendingMediaStream?.getTracks().forEach(track=>track.stop());
  sessionPendingMediaStream=null;
  renderAll();
}

function handleUnauthorized(){
  if(authenticationSuspended)return;
  const recoveryEditors=snapshotSessionEditors();
  const hasDraft=hasRecoverableSessionDraft(recoveryEditors);
  const hasMediaRecovery=hasUnsavedAudio()||Boolean(livePauseRequiredSessionId);
  authenticationRecoveryUsername=$('user-chip')?.textContent?.trim()||'';
  stopCaptureAfterAuthenticationLoss();
  authenticationRecoverySessionId=livePauseRequiredSessionId||((hasMediaRecovery||hasDraft)?state.sessionId:'');
  const preserveMedia=Boolean(authenticationRecoverySessionId&&hasMediaRecovery);
  const preserveDrafts=Boolean(authenticationRecoverySessionId&&hasDraft);
  clearAuthenticatedState({preserveMedia,preserveDrafts,recoveryEditors});
  showLogin();
  toast(preserveMedia||preserveDrafts?'登录已失效；麦克风已停止，未保存的录音和文字草稿已隔离。请使用原账号重新登录。':'登录已失效，请重新登录。',true);
}

function clearAuthenticatedState({preserveMedia=false,preserveDrafts=false,recoveryEditors=null}={}) {
  const preserveRecovery=preserveMedia||preserveDrafts;
  authenticationRecoveryEditors=preserveRecovery?(recoveryEditors||snapshotSessionEditors()):null;
  if(!preserveMedia)resetMockAudioExperience();
  if(!preserveMedia){clearLiveStatusUncertain(state.sessionId);clearLivePauseRequired(state.sessionId);}
  liveRefreshAbort?.abort();
  liveRefreshAbort=null;
  liveRefreshGeneration+=1;
  cancelLiveSuggestionStream();
  activeLoadedSessionId = '';
  state.token = '';
  state.settings = null;
  runtimeSettingsReady = false;
  settingsLoadGeneration += 1;
  if(!preserveRecovery)state.sessionId='';
  state.session = null;
  hydrateSessionForms();
  clearSessionEditors();
  clearSessionDialogs();
  clearAuthenticationEditors();
  localStorage.removeItem('interviewos.token');
  if(!preserveRecovery)localStorage.removeItem('interviewos.session');
  if(!preserveRecovery){authenticationRecoverySessionId='';authenticationRecoveryUsername='';}
  const select = $('session-select');
  if (select) select.innerHTML = '<option value="">选择会话</option>';
  $('user-chip').textContent = '';
  renderAll();
}

async function logout() {
  try{await runMediaTransition(()=>prepareCurrentSessionBoundary());}
  catch(error){toast(`退出已取消：${error.message}。请先保存录音。`,true);return;}
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
    authenticationSuspended=false;
    $('user-chip').textContent = data.username;
    const recoveryRequired=Boolean(authenticationRecoverySessionId);
    if(recoveryRequired){
      if(authenticationRecoveryUsername&&data.username!==authenticationRecoveryUsername){
        api('/api/auth/logout',{method:'POST'}).catch(()=>{});
        state.token='';localStorage.removeItem('interviewos.token');authenticationSuspended=true;$('user-chip').textContent='';
        showLogin();toast('这些未保存内容属于另一个账号；请使用原账号登录后恢复。',true);return;
      }
      const sessions=await api('/api/interviews/sessions');
      if(!sessions.some(session=>session.id===authenticationRecoverySessionId)){
        api('/api/auth/logout',{method:'POST'}).catch(()=>{});
        state.token='';localStorage.removeItem('interviewos.token');authenticationSuspended=true;$('user-chip').textContent='';
        showLogin();toast('这些未保存录音属于另一个账号；请使用原账号登录后恢复。',true);return;
      }
      state.sessionId=authenticationRecoverySessionId;
    }
    await requireRuntimeSettings();
    await loadSessions();
    if(recoveryRequired&&authenticationRecoveryEditors)restoreSessionEditors(authenticationRecoveryEditors);
    hideLogin();
    $('login-username').value = ''; $('login-password').value = '';
    await mediaTransitionPromise.catch(()=>{});
    await mediaFlushTail.catch(()=>{});
    if(hasUnsavedAudio()||(livePauseRequiredSessionId&&livePauseRequiredSessionId===state.sessionId)){
      try{await runMediaTransition(()=>prepareCurrentSessionBoundary());authenticationRecoverySessionId='';authenticationRecoveryEditors=null;authenticationRecoveryUsername='';toast('未保存录音已恢复保存；实时会话已安全暂停');}
      catch(error){toast(`录音仍未保存：${error.message}。请确认使用了原账号。`,true);}
    }else if(recoveryRequired){
      authenticationRecoverySessionId='';authenticationRecoveryEditors=null;authenticationRecoveryUsername='';
    }
    toast(loginMode === 'register' ? '账号已创建并登录' : '已登录');
  } catch (error) { toast(error.message, true); }
  finally { busy(e.currentTarget, false); }
};

async function loadSession() {
  liveRefreshAbort?.abort();
  liveRefreshAbort=null;
  liveRefreshGeneration+=1;
  cancelLiveSuggestionStream();
  if (activeLoadedSessionId !== state.sessionId) {
    if(!hasUnsavedAudio())resetMockAudioExperience();
    // Never render the previous candidate under a newly selected session ID
    // while the replacement request is pending or if it fails.
    state.session = null;
    activeLoadedSessionId = '';
    hydrateSessionForms();
    clearSessionEditors();
    clearSessionDialogs();
  }
  if (!state.sessionId) { activeLoadedSessionId = ''; state.session = null; hydrateSessionForms(); clearSessionEditors(); clearSessionDialogs(); renderState(); return; }
  const requestedSessionId = state.sessionId;
  try {
    let data=await api(`/api/interviews/sessions/${requestedSessionId}`);
    if(state.sessionId!==requestedSessionId)return;
    // ACTIVE describes the shared interview phase, not proof that this tab owns
    // a microphone. Loading from a second tab must not pause another tab.
    if(state.sessionId!==requestedSessionId)return;
    state.session=data.state;
    activeLoadedSessionId=requestedSessionId;
    localStorage.setItem('interviewos.session',requestedSessionId);
    hydrateSessionForms();
    renderState();
  }catch(error){
    if(state.sessionId===requestedSessionId){state.session=null;hydrateSessionForms();renderState();}
    throw error;
  }
}

function hydrateSessionForms() {
  const s = state.session;
  const values = {
    'candidate-resume': s?.candidate?.raw_resume_text,
    'enterprise-resume': s?.candidate?.raw_resume_text,
    'candidate-jd': s?.job?.raw_description || s?.job?.title,
    'enterprise-jd': s?.job?.raw_description || s?.job?.title,
    'candidate-company': s?.company?.name,
    'enterprise-company': s?.company?.name,
    'candidate-company-context': s?.company?.context,
    'enterprise-context': s?.company?.context,
    'interviewer-name': s?.interviewer?.name,
    'interviewer-position': s?.interviewer?.position,
  };
  // Session switches are a hard candidate boundary. Empty fields must clear
  // the previous candidate's form values instead of silently retaining them.
  Object.entries(values).forEach(([id, value]) => { if ($(id)) $(id).value = value ?? ''; dirtySessionEditorIds?.delete(id); });
}

const SESSION_EDITOR_IDS=[
  'candidate-resume','candidate-jd','candidate-company','candidate-company-context','interviewer-name','interviewer-position',
  'enterprise-resume','enterprise-jd','enterprise-company','enterprise-context',
  'mock-answer','live-text','transcript-input','custom-question-text','custom-question-competency',
  'new-candidate','new-job','new-company','claim-statement','claim-note','entity-name',
  'review-content','review-depth','review-structure','review-impact','review-polarity','review-note',
];
const SESSION_CHECKBOX_IDS=['candidate-research-consent','candidate-autopilot','enterprise-research-consent','enterprise-autopilot'];
const dirtySessionEditorIds=new Set();
SESSION_EDITOR_IDS.forEach(id=>$(id)?.addEventListener('input',()=>dirtySessionEditorIds.add(id)));
SESSION_CHECKBOX_IDS.forEach(id=>$(id)?.addEventListener('change',()=>dirtySessionEditorIds.add(id)));

function clearSessionEditors(){
  SESSION_EDITOR_IDS.forEach(id=>{if($(id))$(id).value='';});
  SESSION_CHECKBOX_IDS.forEach(id=>{if($(id))$(id).checked=false;});
  dirtySessionEditorIds.clear();
}

function snapshotSessionEditors(){
  const dirty=[...dirtySessionEditorIds];
  return{
    values:Object.fromEntries(dirty.filter(id=>SESSION_EDITOR_IDS.includes(id)).map(id=>[id,$(id)?.value||''])),
    checked:Object.fromEntries(dirty.filter(id=>SESSION_CHECKBOX_IDS.includes(id)).map(id=>[id,Boolean($(id)?.checked)])),
    dirty,
  };
}

function hasRecoverableSessionDraft(snapshot=snapshotSessionEditors()){
  return Boolean(snapshot?.dirty?.length);
}

function restoreSessionEditors(snapshot){
  Object.entries(snapshot?.values||{}).forEach(([id,value])=>{if($(id))$(id).value=String(value||'');});
  Object.entries(snapshot?.checked||{}).forEach(([id,checked])=>{if($(id))$(id).checked=Boolean(checked);});
  (snapshot?.dirty||[]).forEach(id=>dirtySessionEditorIds.add(id));
}

function clearSessionDialogs(){
  ['session-dialog','entity-dialog','claim-dialog','score-review-dialog'].forEach(id=>{
    const dialog=$(id);
    if(!dialog)return;
    if(dialog.open)dialog.close();
    delete dialog.dataset.claimId;
    delete dialog.dataset.resolutionId;
    delete dialog.dataset.recordId;
  });
  ['entity-copy'].forEach(id=>{if($(id))$(id).textContent='';});
}

function clearAuthenticationEditors(){
  ['login-password','setting-tavily','setting-brave','setting-llm-key','setting-asr-key','setting-tts-key','setting-live-audio-key','setting-resume-llm-key'].forEach(id=>{if($(id))$(id).value='';});
  ['clear-tavily-key','clear-brave-key','clear-llm-key','clear-asr-key','clear-tts-key','clear-live-audio-key','clear-resume-llm-key'].forEach(id=>{if($(id))$(id).checked=false;});
}

function providerSecret(inputId,clearId){
  return $(clearId)?.checked?'':optional(inputId);
}

function renderState() {
  const s = state.session;
  renderSecureContextWarning();
  const workflowLabels = {idle:'待开始', running:'分析中', completed:'已完成', failed:'需要处理'};
  $('metric-workflow').textContent = workflowLabels[s?.workflow?.status] || s?.workflow?.status || '未开始';
  $('metric-step').textContent = s?.autopilot?.enabled ? `AI · ${s.autopilot.phase || s.autopilot.status}` : (s?.workflow?.current_step || '等待输入资料');
  const sources = (s?.company?.public_sources?.length || 0) + (s?.interviewer?.public_expressions?.length || 0);
  $('metric-sources').textContent = sources; $('metric-questions').textContent = s?.mock_interview?.questions?.length || 0; $('metric-evidence').textContent = s?.evidence?.length || 0;
  $('next-action').textContent = localizedNextAction(s?.next_action);
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

function localPendingLiveCaptureIds(sessionId) {
  const ids = new Set();
  const add = value => { if (value) ids.add(value); };
  if (pendingLiveCaptureStart?.sessionId === sessionId) add(pendingLiveCaptureStart.recordingId);
  if (liveRecordingSessionId === sessionId) add(liveRecordingId);
  if (liveContinuousSessionId === sessionId) {
    add(liveContinuousGuardId);
    add(liveVadRecordingId);
  }
  for (const chunk of liveContinuousQueue) {
    if (chunk.sessionId !== sessionId) continue;
    add(chunk.recordingId);
    add(chunk.parentRecordingId);
  }
  return ids;
}

function renderLive() {
  const sessionAudioPending=sessionPartsPending(state.sessionId)||(sessionRecordingSessionId===state.sessionId&&sessionChunks.length>0);
  const localPendingCaptureCount=localPendingLiveCaptureIds(state.sessionId).size;
  const utterancePendingAudio=liveAudioChunks.length>0||liveContinuousQueue.some(chunk=>chunk.sessionId===state.sessionId)||localPendingCaptureCount>0;
  const currentSessionAudioError=sessionAudioErrorSessionId===state.sessionId?sessionAudioSaveError:'';
  const currentSessionRecordingError=sessionRecordingErrorSessionId===state.sessionId?sessionRecordingError:'';
  renderLiveView({
    liveContinuousMode,
    liveContinuousQueueLength: liveContinuousQueue.length,
    liveContinuousUploading,
    livePendingAudio:!liveRecorder&&(liveAudioChunks.length>0||liveRecordingUploading||sessionAudioPending||!!currentSessionAudioError),
    utterancePendingAudio,
    utteranceCaptureBusy:!!liveRecorder||liveCaptureStartPending||liveRecordingUploading||liveContinuousUploading,
    liveRecorderActive: !!liveRecorder || liveCaptureStartPending || liveRecordingUploading || liveAudioChunks.length > 0 || liveContinuousQueue.length > 0 || liveContinuousUploading || sessionAudioPending || sessionAudioUploading,
    liveSuggestionStreamActive: !!liveSuggestionStream,
    liveStatusUncertain:liveStatusUncertainSessionId===state.sessionId,
    mediaTransitionInProgress:mediaTransitionInProgress||mediaFlushInProgress,
    microphoneAvailabilityMessage,
    sessionAudioUploading,
    sessionAudioPending,
    sessionAudioSaveError:currentSessionAudioError,
    sessionRecordingActive,
    sessionRecordingError:currentSessionRecordingError,
    settingsReady:runtimeSettingsReady,
    updateLiveVadNote
  });
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

function maybePromptEntityResolution(){
  const dialog=$('entity-dialog'); if(dialog.open)return;
  const item=(state.session?.entity_resolutions||[]).find(x=>x.status==='pending'); if(!item)return;
  dialog.dataset.resolutionId=item.id;$('entity-copy').textContent=`搜索结果显示“${item.proposed_name}”可能是“${item.input_name}”的正确实体。是否将“${item.input_name}”更正为“${item.proposed_name}”？`;
  $('entity-name').value=item.proposed_name||item.input_name||'';
  dialog.showModal();
}

function renderMock() {
  const plan=state.session?.mock_interview?.questions||[]; const session=state.session?.mock_session; const index=session?.current_question_index||0;
  $('mock-progress').textContent=''; $('mock-status').textContent=session?.status==='active'?'面试进行中':session?.status==='evaluating'?'正在生成报告':session?.status==='completed'?'本轮已完成':'准备开始';
  const current=session?.status==='active'?plan[index]:null; const questionText=session?.pending_follow_up||current?.question; $('start-mock').style.display=session?.status==='idle'||!session?'inline-block':'none';
  const sessionDraft=session?.answer_draft;
  if(sessionDraft?.retry&&current&&sessionDraft.question_id===current.id&&!mockRetry){mockRetry=true;mockRetryResponseId=sessionDraft.retry_response_id||'';}
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
  const matchingDraft=!!(sessionDraft&&current&&sessionDraft.question_id===current.id&&sessionDraft.question?.trim()===displayedQuestionText?.trim()&&!!sessionDraft.retry===isRetrying&&(!isRetrying||sessionDraft.retry_response_id===mockRetryResponseId));
  $('answer-form').style.display = current && (!justAnswered || isRetrying) ? 'block' : 'none';
  const understandingNode=$('mock-question-understanding');const understandingBody=$('mock-question-understanding-body');const understanding=current?.understanding;const showUnderstanding=!!(current&&understanding&&!displayedAsFollowUp);
  if(understandingNode){understandingNode.classList.toggle('hidden',!showUnderstanding);if(showUnderstanding){if(understandingNode.dataset.questionId!==current.id){understandingNode.dataset.questionId=current.id;understandingNode.open=true;}$('mock-question-understanding-summary').textContent=`问题解析与举一反三 · ${understanding.answer_type_label||'综合问题'} · ${understanding.analysis_source==='model'?'语义增强':'规则解析'}`;understandingBody.innerHTML=`<h4>这道题真正想验证什么</h4><p>${esc(understanding.assessment_goal)}</p><div class="question-understanding-grid"><div><small>回答边界</small><ul>${(understanding.answer_boundary||[]).map(item=>`<li>${esc(item)}</li>`).join('')}</ul></div><div><small>常见误区</small><ul>${(understanding.common_mistakes||[]).map(item=>`<li>${esc(item)}</li>`).join('')}</ul></div><div><small>面试官可能怎样验证</small><ul>${(understanding.likely_follow_ups||[]).map(item=>`<li>${esc(item)}</li>`).join('')}</ul></div></div><p class="transfer-principle"><strong>举一反三：</strong>${esc(understanding.transfer_principle)}</p>${(understanding.related_questions||[]).length?`<div class="related-question-list"><small>同一考察目标的不同问法 · 点击可切换练习</small><div class="related-question-actions">${understanding.related_questions.map(item=>`<button type="button" data-related-question="${esc(item)}" data-related-competency="${esc(understanding.competency||current.competency||'')}">${esc(item)}</button>`).join('')}</div></div>`:''}${renderQuestionDeepAnalysis(current,understanding)}`;}else{understandingNode.removeAttribute('open');understandingNode.dataset.questionId='';understandingBody.innerHTML='';}}
  if (framework) { const hasFw = current && current.answer_framework && (!justAnswered || (isRetrying && !retryResponse.is_follow_up)); framework.classList.toggle('hidden', !hasFw); if (hasFw) frameworkText.textContent = current.answer_framework; }
  const requirements=$('mock-requirements');const showRequirements=current&&!displayedAsFollowUp&&(current.question_requirements||[]).length;if(requirements){requirements.classList.toggle('hidden',!showRequirements);requirements.innerHTML=showRequirements?`<small>这道题需要回答</small>${tags(current.question_requirements)}`:'';}
  const example=$('mock-example');const showExample=current&&!displayedAsFollowUp&&current.example_answer&&(!justAnswered||isRetrying);if(example){example.classList.toggle('hidden',!showExample);if(showExample){$('mock-example-note').textContent=current.example_answer_note||'教学示例为虚构场景，请替换为你的真实经历。';$('mock-example-text').textContent=current.example_answer;}}
  const completed=session?.status==='completed'; const responseCount=session?.responses?.length||0; const scoredResponses=(session?.responses||[]).filter(item=>item.evaluation); const averageScore=scoredResponses.length?Math.round(scoredResponses.reduce((sum,item)=>sum+((item.evaluation.content+item.evaluation.technical_depth+item.evaluation.structure+item.evaluation.impact)/4),0)/scoredResponses.length*100):0;
  const followUpStage=session?.pending_follow_up_stage||displayResponse?.follow_up_stage||'evidence';
  const followUpLabels={recovery:'恢复引导',foundation:'基础澄清',evidence:'证据验证',tradeoff:'取舍深挖',pressure:'压力迁移'};
  const questionLabel=displayedAsFollowUp?`${followUpLabels[followUpStage]||'动态'}追问`:current?.source==='custom'?`自定义问题${current.competency&&current.competency!=='自定义问题'?` · ${current.competency}`:''}`:(current?.competency||'综合能力');
  const followUpReason=displayedAsFollowUp&&session?.pending_follow_up_rationale?`<p class="follow-up-reason">追问原因：${esc(session.pending_follow_up_rationale)}</p>`:'';
  $('mock-question').className=current?'question-copy':completed?'question-copy completion-state':'question-copy empty-state'; $('mock-question').innerHTML=current?`<small>${esc(questionLabel)}</small>${esc(displayedQuestionText)}${followUpReason}`:completed?`<small>练习已保存</small><strong>本轮完成 ${responseCount} 次回答${averageScore?` · 平均 ${averageScore} 分`:''}</strong><p>先查看改进报告梳理共性问题；需要练习另一岗位或候选人时，新建会话可避免证据混用。</p><div class="completion-actions"><button class="btn primary" type="button" data-route="candidate-report">查看改进报告</button><button class="btn" type="button" data-new-practice>新建练习会话</button></div>`:'可以先添加一个你认为会被问的问题，或完成候选人准备工作流生成个性化题目。';
  const captureBusy=mockVoiceCaptureBusy()||mockMutationInProgress||mediaTransitionInProgress||mediaFlushInProgress;
  $('start-mock').disabled=mockMutationInProgress||mediaTransitionInProgress||mediaFlushInProgress;
  const customQuestionButton=$('add-custom-question'); const customQuestionNote=$('custom-question-note');
  if(customQuestionButton){const locked=['evaluating','completed'].includes(session?.status);customQuestionButton.disabled=locked||captureBusy;customQuestionNote.textContent=locked?'本轮已经结束；请新建练习会话后继续添加问题。':captureBusy?'请先停止并保存当前录音。':'加入后立即练习，并使用相同的语音、评分、追问和报告流程。';}
  const questionAudio=$('mock-speak-question')?.closest('.question-audio-controls'); if(questionAudio)questionAudio.classList.toggle('hidden',!current);
  const speakButton=$('mock-speak-question');if(speakButton)speakButton.disabled=!current||captureBusy;
  const voiceButton=$('mock-voice');if(voiceButton){const blockedMessage=!runtimeSettingsReady?'运行配置尚未成功加载':microphoneAvailabilityMessage();voiceButton.disabled=!current||captureBusy||!!blockedMessage;voiceButton.title=blockedMessage;}
  const speechKey=current?`${current.id}:${displayedQuestionText}`:'';
  if(speechKey!==mockCurrentSpeechKey){cancelMockQuestionSpeech();clearMockQuestionAudio();clearMockAnswerExperience();mockCurrentSpeechKey=speechKey;}
  mockDisplayedResponseId=displayResponse?.id||'';
  if(matchingDraft)restoreMockAnswerDraft(sessionDraft);else syncMockAnswerAudio(displayResponse);
  restorePendingMockPlayback();
  const draftActions=$('mock-draft-actions');if(draftActions)draftActions.classList.toggle('hidden',!sessionDraft&&!mockVoicePendingUpload);
  const mockViewVisible=state.view==='mock'&&document.visibilityState==='visible'&&$('view-mock')?.classList.contains('active');
  if(current&&session?.status==='active'&&mockViewVisible&&!captureBusy&&$('mock-auto-speak')?.checked&&speechKey!==mockSpokenQuestionKey){mockSpokenQuestionKey=speechKey;setTimeout(()=>speakCurrentMockQuestion(true),0);}
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
    [finishBtn,nextBtn,prevBtn,retryBtn,retryMainBtn,reviewBtn].forEach(button=>{if(button)button.disabled=captureBusy;});
  }
  const last=displayResponse||(completed?(session?.responses||[]).at(-1):null); const node=$('coach-result'); $('coach-title').textContent=completed&&last?'最后一题反馈':'四维评价'; if(completed&&last){mockDisplayedResponseId=last.id;syncMockAnswerAudio(last);}
  const showEval = !!last;
  if (!showEval) { const pendingReviews=(session?.responses||[]).filter(item=>item.evaluation?.review_status==='pending'); node.className=pendingReviews.length?'review-queue':'empty-state'; node.innerHTML=pendingReviews.length?`<div class="review-subtitle">尚待人工复核的规则评分</div>${pendingReviews.map(item=>`<div class="review-claim"><div><small>${esc(item.competency)}</small><span>${esc(item.question)}</span></div><div class="claim-actions"><button type="button" data-score-review="${esc(item.id)}">人工复核评分</button></div></div>`).join('')}`:'提交回答后显示内容、深度、结构和影响力评分。'; return; }
  const e=last.evaluation; const sourceLabel=e.scoring_source==='human'?'人工已复核':e.scoring_source==='deterministic_rule'?'规则评分 · 待复核':'AI 评分'; const dimensionLabels={content:'岗位相关证据',technical_depth:'决策与专业深度',structure:'表达结构',impact:'结果与复盘'}; const dimensionHtml=(e.dimension_feedback||[]).map(item=>`<div class="dimension-card"><div><strong>${esc(dimensionLabels[item.dimension]||item.dimension)}</strong><b>${Math.round((item.score||0)*100)} · ${esc(item.level)}</b></div><p>${esc(item.evidence)}</p><small>下一步：${esc(item.suggestion)}</small></div>`).join(''); node.className=''; node.innerHTML=`<div class="score-grid">${[['证据',e.content],['深度',e.technical_depth],['结构',e.structure],['结果',e.impact]].map(([n,v])=>`<div class="score"><span>${n}</span><strong>${Math.round(v*100)}</strong></div>`).join('')}</div><div class="claim-actions"><small>${esc(sourceLabel)}</small><button type="button" data-score-review="${esc(last.id)}">人工复核评分</button></div>${dimensionHtml?`<div class="dimension-feedback">${dimensionHtml}</div>`:''}${list('优先改进',e.feedback)}<div class="result-block"><h4>基于你本次回答的重组示范</h4><p>${esc(e.improved_answer)}</p></div>`;
  if(matchingDraft)renderSpeechFeedback(sessionDraft.speech_delivery);else if(last.speech_delivery?.strengths?.length||last.speech_delivery?.improvements?.length)renderSpeechFeedback(last.speech_delivery);
  node.insertAdjacentHTML('beforeend', renderSpokenAnswerAnalysis(last.evaluation));
  const details=node.innerHTML;
  const previous=[...(session?.attempt_history||[])].reverse().find(item=>item.question_id===last.question_id&&item.question===last.question);
  const headline=(e.feedback||[])[0]||'已完成本题证据检查，可以查看具体依据或立即重答。';
  const hasCoverageGap=(e.spoken_analysis?.question_coverage||[]).some(item=>item.status==='missing'||item.status==='partial');
  node.innerHTML=`<div class="coach-summary"><small>${hasCoverageGap?'本题最优先改进':'本题最值得继续打磨'}</small><strong>${esc(headline)}</strong><div class="claim-actions">${completed?'':`<button type="button" data-quick-retry="${esc(last.id)}" ${captureBusy?'disabled':''}>按建议重答</button>`}${last.audio_file?`<button type="button" data-delete-mock-audio="${esc(last.id)}" ${captureBusy?'disabled':''}>删除本次录音</button>`:''}</div></div>${renderQuestionAlignment(e)}${renderRetryComparison(last,previous)}<details class="coach-details"><summary>查看完整分析、评分和校准依据</summary>${details}</details>`;
}

async function ensureSession() { if (state.sessionId) return true; $('session-dialog').showModal(); toast('请先创建一个会话'); return false; }
function runMediaTransition(operation){
  if(mediaTransitionInProgress||mediaFlushInProgress)throw new Error('另一个录音或会话操作仍在完成');
  mediaTransitionInProgress=true;
  const sessionSelect=$('session-select');
  if(sessionSelect)sessionSelect.disabled=true;
  renderLive();
  const execution=(async()=>{
    try{return await operation();}
    finally{mediaTransitionInProgress=false;if(sessionSelect)sessionSelect.disabled=false;renderLive();}
  })();
  mediaTransitionPromise=execution.catch(()=>{});
  return execution;
}

async function requestLiveStatus(sessionId,status,revision=state.session?.live_interview?.status_revision,operationId='',generation=null){
  const expectedRevision=Number(revision||0);
  const body={status,expected_revision:expectedRevision};
  if(status==='active')body.operation_id=operationId;
  try{
    return await boundedApi(`/api/live-interviews/${sessionId}/status`,{method:'POST',body:JSON.stringify(body)},15000,'实时会话状态更新超时');
  }catch(mutationError){
    try{
      const current=await boundedApi(`/api/live-interviews/${sessionId}`,{},15000,'实时会话状态确认超时');
      const live=current.state?.live_interview;
      const confirmed=status==='active'
        ?Boolean(operationId&&generation&&liveTransitionMatches(live,generation,status,operationId))
        :live?.status===status;
      if(confirmed)return current;
      mutationError.liveState=current.state;
      if(mutationError.requestTimedOut||!mutationError.status||mutationError.status>=500)mutationError.liveStatusUncertain=true;
      else mutationError.liveStatusKnown=true;
    }catch(confirmError){
      mutationError.liveStatusUncertain=true;
    }
    throw mutationError;
  }
}

async function requestLiveStart(sessionId,consentConfirmed,generation,operationId){
  const expectedRevision=generation.statusRevision;
  try{
    return await boundedApi(`/api/live-interviews/${sessionId}/start`,{method:'POST',body:JSON.stringify({consent_confirmed:consentConfirmed,expected_revision:expectedRevision,operation_id:operationId})},15000,'实时会话启动超时');
  }catch(mutationError){
    try{
      const current=await boundedApi(`/api/live-interviews/${sessionId}`,{},15000,'实时会话状态确认超时');
      if(liveTransitionMatches(current.state?.live_interview,generation,'active',operationId))return current;
      mutationError.liveState=current.state;
      if(mutationError.requestTimedOut||!mutationError.status||mutationError.status>=500)mutationError.liveStatusUncertain=true;
      else mutationError.liveStatusKnown=true;
    }catch(confirmError){
      mutationError.liveStatusUncertain=true;
    }
    throw mutationError;
  }
}

function snapshotLiveGeneration(){
  const live=state.session?.live_interview||{};
  return{
    captureEpoch:Number(live.capture_epoch||0),
    statusRevision:Number(live.status_revision||0),
    status:live.status||'idle',
    operationId:live.active_start_operation_id?String(live.active_start_operation_id):'',
  };
}

function liveGenerationMatches(live,generation){
  const generationMatches=Number(live?.capture_epoch||0)===generation.captureEpoch&&Number(live?.status_revision||0)===generation.statusRevision;
  return generationMatches&&(!generation.operationId||String(live?.active_start_operation_id||'')===generation.operationId);
}

function liveTransitionMatches(live,generation,status,operationId=''){
  const expectedCaptureEpoch=status==='active'&&generation.status!=='active'?generation.captureEpoch+1:generation.captureEpoch;
  const operationMatches=!operationId||String(live?.active_start_operation_id||'')===operationId;
  return operationMatches&&live?.status===status&&Number(live.capture_epoch||0)===expectedCaptureEpoch&&Number(live.status_revision||0)===generation.statusRevision+1;
}

function transitionedLiveGeneration(generation,status,operationId=''){
  return{
    captureEpoch:status==='active'&&generation.status!=='active'?generation.captureEpoch+1:generation.captureEpoch,
    statusRevision:generation.statusRevision+1,
    status,
    operationId,
  };
}

function liveOwnershipChangedError(current,action){
  const error=new Error(`远端已进入新的监听状态，本窗口未${action}`);
  error.remoteOwnershipChanged=true;
  error.liveState=current?.state||null;
  return error;
}

async function preflightLiveGeneration(sessionId,generation,action){
  const current=await boundedApi(`/api/live-interviews/${sessionId}`,{},5000,`${action}所有权预检超时`);
  if(sessionId!==state.sessionId)throw new Error('会话已变更');
  state.session=current.state;
  if(!liveGenerationMatches(current.state?.live_interview,generation))throw liveOwnershipChangedError(current,action);
  return current;
}

async function enforceFailClosedPause(sessionId,expectedGeneration,alternateGeneration=null){
  markLiveStatusUncertain(sessionId,expectedGeneration,alternateGeneration);
  let lastError=new Error('无法确认实时会话已安全暂停');
  for(let attempt=0;attempt<4;attempt+=1){
    try{
      const current=await boundedApi(`/api/live-interviews/${sessionId}`,{},5000,'安全状态预检超时');
      const live=current.state?.live_interview||{};
      const acceptedGenerations=[expectedGeneration,alternateGeneration].filter(Boolean);
      if(!acceptedGenerations.some(generation=>liveGenerationMatches(live,generation))){
        const mutationAlreadyClosed=acceptedGenerations.some(generation=>Number(live.capture_epoch||0)===generation.captureEpoch&&Number(live.status_revision||0)===generation.statusRevision+1&&live.status!=='active');
        if(mutationAlreadyClosed){
          clearLiveStatusUncertain(sessionId);
          return current;
        }
        throw liveOwnershipChangedError(current,'暂停新的监听轮次');
      }
      const expectedRevision=Number(live.status_revision||0);
      const path=live.status==='active'?`/api/live-interviews/${sessionId}/status`:`/api/live-interviews/${sessionId}/status-barrier`;
      const body=live.status==='active'?{status:'paused',expected_revision:expectedRevision}:{expected_revision:expectedRevision};
      const data=await boundedApi(path,{method:'POST',body:JSON.stringify(body)},5000,'安全状态屏障请求超时');
      if(data.state?.live_interview?.status!=='active'){
        clearLiveStatusUncertain(sessionId);
        return data;
      }
    }catch(error){
      if(error.remoteOwnershipChanged){clearLiveStatusUncertain(sessionId);throw error;}
      lastError=error;
    }
  }
  throw lastError;
}

async function confirmRemoteLiveCaptureAuthority(sessionId){
  const data=await boundedApi(`/api/live-interviews/${sessionId}`,{},5000,'实时会话权限确认超时');
  const remoteRevision=Number(data.state?.live_interview?.status_revision||0);
  const currentRevision=sessionId===state.sessionId?Number(state.session?.live_interview?.status_revision||0):0;
  if(sessionId!==state.sessionId||remoteRevision<currentRevision)return false;
  state.session=data.state;
  return data.state?.live_interview?.status==='active';
}

function freezeLocalCapture({includeMock=false,sessionId=state.sessionId}={}){
  liveCaptureGeneration+=1;
  sessionRecorderStartGeneration+=1;
  cancelLiveRecordingRollover();
  liveCaptureStartPending=false;
  sessionRecorderStartPending=false;
  if(sessionRecorderStartSessionId===sessionId){sessionRecorderStartSessionId='';sessionRecorderStartCaptureEpoch=0;}
  if(pendingLiveCaptureStart?.sessionId===sessionId){
    const pending=pendingLiveCaptureStart;
    pendingLiveCaptureStart=null;
    void cancelLiveAudioCapture(pending.sessionId,pending.recordingId).catch(()=>{});
  }
  if(includeMock){
    if(mockVoiceStartPending&&!mockVoiceRecorder)mockVoiceGeneration+=1;
    mockVoiceStartPending=false;
    if(mockVoiceRecorder?.state==='recording'){
      try{mockVoiceRecorder.requestData();}catch{}
      try{mockVoiceRecorder.stop();}catch{}
    }
    mockVoiceStream?.getTracks().forEach(track=>track.stop());
  }
  if(liveContinuousMode&&liveContinuousSessionId===sessionId){
    liveContinuousResumeSessionId=sessionId;
    liveContinuousMode=false;
  }
  if(liveRecorder){
    if(liveRecorder.state==='recording'){try{liveRecorder.stop();}catch{}}
  }else if(liveMediaStream){
    // getUserMedia can resolve before capture registration/MediaRecorder
    // construction. Stop those tracks immediately when a pause or navigation
    // invalidates the pending start.
    liveMediaStream.getTracks().forEach(track=>track.stop());
    if(!liveAudioChunks.length&&!liveRecordingUploading){
      liveMediaStream=null;
      if(liveRecordingSessionId===sessionId){liveRecordingSessionId='';liveRecordingId='';}
    }
  }
  requestSessionRecorderStop('boundary');
  sessionPendingMediaStream?.getTracks().forEach(track=>track.stop());
  sessionPendingMediaStream=null;
}

async function prepareCurrentSessionBoundary(){
  const sessionId=state.sessionId;
  if(!sessionId)return;
  const boundaryGeneration=state.session?.live_interview?snapshotLiveGeneration():null;
  const uncertainGeneration=liveStatusUncertainSessionId===sessionId?liveStatusUncertainExpectedGeneration:null;
  cancelLiveSuggestionStream();
  const hadLocalCapture=hasLocalLiveCapture(sessionId);
  const localCaptureEpoch=hadLocalCapture?activeLocalCaptureEpoch(sessionId):null;
  const pauseCaptureEpoch=Number.isInteger(localCaptureEpoch)?localCaptureEpoch:livePauseRequiredSessionId===sessionId?livePauseRequiredCaptureEpoch:uncertainGeneration?.captureEpoch;
  const pauseStatusRevision=hadLocalCapture?Number(state.session?.live_interview?.status_revision):livePauseRequiredSessionId===sessionId?livePauseRequiredStatusRevision:uncertainGeneration?.statusRevision;
  const pauseGeneration=Number.isInteger(pauseCaptureEpoch)&&Number.isInteger(pauseStatusRevision)?{captureEpoch:pauseCaptureEpoch,statusRevision:pauseStatusRevision,status:boundaryGeneration?.status||'active'}:boundaryGeneration;
  // ACTIVE is a shared interview phase, not proof that this browser tab owns
  // the microphone. An observer tab must not pause another window merely by
  // switching sessions or signing out.
  const mustConfirmPause=hadLocalCapture||livePauseRequiredSessionId===sessionId||liveStatusUncertainSessionId===sessionId||(!state.session&&hasUnsavedAudio());
  freezeLocalCapture({includeMock:true,sessionId});
  const localDrain=flushActiveMedia();
  let drainError=null;
  try{await localDrain;}catch(error){drainError=error;}
  if(mustConfirmPause){
    if(!pauseGeneration)throw drainError||new Error('无法识别本窗口的实时监听轮次，已拒绝暂停远端会话');
    if(Number.isInteger(pauseCaptureEpoch)||Number.isInteger(pauseStatusRevision)){
      let remote;
      try{remote=await boundedApi(`/api/live-interviews/${sessionId}`,{},5000,'暂停所有权预检超时');}
      catch(error){throw drainError||error;}
      const remoteLive=remote.state?.live_interview||{};
      const captureChanged=Number.isInteger(pauseCaptureEpoch)&&Number(remoteLive.capture_epoch)!==pauseCaptureEpoch;
      const statusChanged=Number.isInteger(pauseStatusRevision)&&Number(remoteLive.status_revision)!==pauseStatusRevision;
      if(captureChanged||statusChanged){
        if(sessionId===state.sessionId)state.session=remote.state;
        clearLivePauseRequired(sessionId);
        clearLiveStatusUncertain(sessionId);
        if(drainError)throw drainError;
        return {remoteOwnershipChanged:true};
      }
      if(sessionId===state.sessionId)state.session=remote.state;
    }
    try{
      const data=await requestLiveStatus(sessionId,'paused',pauseGeneration.statusRevision);
      if(sessionId===state.sessionId)state.session=data.state;
      clearLivePauseRequired(sessionId);
      clearLiveStatusUncertain(sessionId);
    }catch(error){
      const confirmedLive=error.liveState?.live_interview;
      const captureChanged=Number.isInteger(pauseCaptureEpoch)&&Number(confirmedLive?.capture_epoch)!==pauseCaptureEpoch;
      const statusChanged=Number.isInteger(pauseStatusRevision)&&Number(confirmedLive?.status_revision)!==pauseStatusRevision;
      if(confirmedLive&&(captureChanged||statusChanged)){
        if(sessionId===state.sessionId)state.session=error.liveState;
        clearLivePauseRequired(sessionId);
        clearLiveStatusUncertain(sessionId);
        if(drainError)throw drainError;
        return {remoteOwnershipChanged:true};
      }
      try{
        const paused=await enforceFailClosedPause(sessionId,pauseGeneration,liveStatusUncertainAlternateGeneration);
        if(sessionId===state.sessionId)state.session=paused.state;
        clearLivePauseRequired(sessionId);
      }catch{
        throw drainError||error;
      }
    }
  }
  if(drainError)throw drainError;
}

$('nav').addEventListener('click', e => { const button=e.target.closest('[data-view]'); if(button) setView(button.dataset.view); });
document.querySelectorAll('[data-role]').forEach(button => button.onclick = () => setRole(button.dataset.role));
document.querySelectorAll('[data-global-view]').forEach(button => button.onclick = () => setView(button.dataset.globalView));
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>setView(b.dataset.go));
$('new-session').onclick=()=> $('session-dialog').showModal();
$('close-session-dialog').onclick=()=> $('session-dialog').close();
$('session-select').onchange=async e=>{const select=e.currentTarget;const previousSessionId=state.sessionId;const nextSessionId=select.value;if(nextSessionId===previousSessionId)return;try{await runMediaTransition(async()=>{await prepareCurrentSessionBoundary();state.sessionId=nextSessionId;await loadSession();});}catch(error){state.sessionId=previousSessionId;select.value=previousSessionId;await loadSession().catch(()=>{});toast(`无法切换会话：${error.message}。已恢复原会话，原始录音仍保留。`,true);}};
$('session-form').onsubmit=async e=>{e.preventDefault();try{await runMediaTransition(async()=>{await prepareCurrentSessionBoundary();const data=await api('/api/interviews/sessions',{method:'POST',body:JSON.stringify({candidate_name:$('new-candidate').value,job_title:$('new-job').value,company_name:$('new-company').value})});state.sessionId=data.id;state.session=null;activeLoadedSessionId='';localStorage.setItem('interviewos.session',data.id);hydrateSessionForms();clearSessionEditors();renderState();$('session-dialog').close();await loadSessions();});toast('会话已创建');}catch(error){toast(`无法创建并切换会话：${error.message}`,true)}};

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

$('candidate-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);showWorkflowStarting();try{const payload={role:'candidate',resume_text:$('candidate-resume').value,job_description:$('candidate-jd').value,company_name:$('candidate-company').value,company_context:$('candidate-company-context').value,interviewer_name:$('interviewer-name').value,interviewer_position:$('interviewer-position').value,authorized_public_research:$('candidate-research-consent').checked};const endpoint=$('candidate-autopilot').checked?`/api/autopilot/${state.sessionId}/run`:'/api/workflows/candidate-prep';if(!$('candidate-autopilot').checked)payload.session_id=state.sessionId;const data=await api(endpoint,{method:'POST',body:JSON.stringify(payload)});state.session=data.state;await loadSessions();toast(state.session.autopilot?.enabled?'AI 已推进到需要你回答的阶段':'候选人策略已生成');}catch(error){await loadSession().catch(()=>{});toast(`运行失败：${error.message}`,true)}finally{busy(form,false)}};
$('enterprise-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);showWorkflowStarting();try{const payload={role:'interviewer',resume_text:$('enterprise-resume').value,job_description:$('enterprise-jd').value,company_name:$('enterprise-company').value,company_context:$('enterprise-context').value,authorized_public_research:$('enterprise-research-consent').checked};const endpoint=$('enterprise-autopilot').checked?`/api/autopilot/${state.sessionId}/run`:'/api/workflows/enterprise-design';if(!$('enterprise-autopilot').checked)payload.session_id=state.sessionId;const data=await api(endpoint,{method:'POST',body:JSON.stringify(payload)});state.session=data.state;await loadSessions();toast(state.session.autopilot?.enabled?'AI 已完成设计，等待采集真实面试证据':'面试 Blueprint 已生成');}catch(error){await loadSession().catch(()=>{});toast(`运行失败：${error.message}`,true)}finally{busy(form,false)}};

async function runMockMutation(operation){
  if(mockMutationInProgress){toast('上一个模拟面试操作仍在处理，请稍候',true);return null;}
  mockMutationInProgress=true;
  renderMock();
  try{return await operation();}
  finally{mockMutationInProgress=false;renderMock();}
}
$('start-mock').onclick=()=>runMockMutation(async()=>{if(!await ensureSession())return;const sessionId=state.sessionId;try{await api(`/api/mock-interviews/${sessionId}/start`,{method:'POST'});if(sessionId!==state.sessionId)return;await loadSession();toast('模拟面试已开始');}catch(error){toast(error.message,true)}});
async function discardMockDraftIfNeeded({confirmUser=true}={}){
  if(mockVoiceCaptureActive()){toast('请先停止并保存当前录音',true);return false;}
  const draft=state.session?.mock_session?.answer_draft;
  const pendingUpload=mockVoicePendingUpload;
  if(!draft&&!pendingUpload)return true;
  if(confirmUser&&!window.confirm('当前有一段尚未提交的录音。删除录音和转写草稿后继续吗？'))return false;
  const sessionId=state.sessionId;
  const recordingId=draft?.recording_id||'';
  try{
    if(draft)await api(`/api/mock-interviews/${sessionId}/recordings/${recordingId}`,{method:'DELETE'});
    if(sessionId!==state.sessionId)return false;
    mockVoicePendingUpload=null;
    mockVoiceChunks=[];
    mockVoiceContext=null;
    mockVoiceSessionId='';
    if($('mock-stop-voice'))$('mock-stop-voice').disabled=true;
    if($('mock-answer'))$('mock-answer').value='';
    clearMockAnswerExperience();
    if(draft)await loadSession();else renderMock();
    toast('未提交的录音与转写草稿已删除');
    return true;
  }catch(error){toast(`无法删除录音草稿：${error.message}`,true);return false;}
}
async function addCustomQuestion(question,competency='',busyNode=null){const result=await runMockMutation(async()=>{if(!await ensureSession()||!await discardMockDraftIfNeeded())return false;if(!question.trim())return false;if(busyNode)busy(busyNode,true);const sessionId=state.sessionId;try{await api(`/api/mock-interviews/${sessionId}/questions`,{method:'POST',body:JSON.stringify({question:question.trim(),competency:competency.trim(),practice_now:true})});if(sessionId!==state.sessionId)return false;mockRetry=false;mockRetryResponseId='';await loadSession();toast('问题已解析并加入练习');return true;}catch(error){toast(error.message,true);return false;}finally{if(busyNode)busy(busyNode,false)}});return result===true;}
$('custom-question-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;const added=await addCustomQuestion($('custom-question-text').value,$('custom-question-competency').value,form);if(added){$('custom-question-text').value='';$('custom-question-competency').value='';$('custom-question-panel').open=false;}};
function startMockScoringProgress(){mockVoiceNote('正在提交回答并提取可核验证据…');const timers=[setTimeout(()=>mockVoiceNote('模型正在评分并校准证据一致性；完成后会自动展示结果…'),1200),setTimeout(()=>mockVoiceNote('本地模型仍在推理；你的回答保留在输入框中，请勿重复提交。'),12000)];return()=>timers.forEach(clearTimeout);}
$('answer-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;await runMockMutation(async()=>{busy(form,true);const stopProgress=startMockScoringProgress();try{if(mockVoiceCaptureBusy())await finalizeMockVoiceRecording({quiet:true});const sessionId=state.sessionId;const mockSession=state.session?.mock_session;const question=state.session?.mock_interview?.questions?.[mockSession?.current_question_index];if(!question)return;await api(`/api/mock-interviews/${sessionId}/answers`,{method:'POST',body:JSON.stringify({question_id:mockSession.pending_parent_question_id||question.id,answer:$('mock-answer').value,retry:mockRetry,retry_response_id:mockRetryResponseId||null,recording_id:mockPendingRecordingId||null})});if(sessionId!==state.sessionId)return;$('mock-answer').value='';mockPendingRecordingId='';mockRetry=false;mockRetryResponseId='';mockSpeechFeedbackPollGeneration+=1;await loadSession();toast('回答已评分');}catch(error){mockVoiceNote('评分未完成，回答仍保留，可检查后重试。');toast(error.message,true)}finally{stopProgress();busy(form,false);}});};
async function beginMockRetry(responseId){return runMockMutation(async()=>{if(!await discardMockDraftIfNeeded())return;mockRetry=true;mockRetryResponseId=responseId||'';mockPendingRecordingId='';$('mock-answer').value='';renderMock();});}
$('mock-retry').onclick=e=>beginMockRetry(e.currentTarget.dataset.responseId||'');
$('mock-retry-main').onclick=e=>beginMockRetry(e.currentTarget.dataset.responseId||'');
document.addEventListener('click',async event=>{const related=event.target.closest('[data-related-question]');if(related){await addCustomQuestion(related.dataset.relatedQuestion||'',related.dataset.relatedCompetency||'',related);return;}const route=event.target.closest('[data-route]');if(route){setView(route.dataset.route);return;}const fresh=event.target.closest('[data-new-practice]');if(fresh){$('session-dialog').showModal();return;}const retry=event.target.closest('[data-quick-retry]');if(retry){await beginMockRetry(retry.dataset.quickRetry||'');$('mock-answer')?.focus();return;}const play=event.target.closest('[data-play-mock-audio]');if(play){await loadMockAnswerAudio(play.dataset.playMockAudio,true);mockVoiceNote('正在回放上一版回答');return;}const remove=event.target.closest('[data-delete-mock-audio]');if(remove){await runMockMutation(async()=>{const sessionId=state.sessionId;const responseId=remove.dataset.deleteMockAudio||'';try{await api(`/api/mock-interviews/${sessionId}/answers/${responseId}/audio`,{method:'DELETE'});if(sessionId!==state.sessionId)return;await loadSession();toast('录音已从本机删除，文字与评分仍保留');}catch(error){toast(error.message,true);}});}});
$('mock-discard-draft').onclick=()=>runMockMutation(()=>discardMockDraftIfNeeded());
async function navigateMockInterview(path,message,{clearAnswer=true}={}){return runMockMutation(async()=>{if(!await ensureSession()||!await discardMockDraftIfNeeded())return;const sessionId=state.sessionId;cancelMockQuestionSpeech();try{await api(`/api/mock-interviews/${sessionId}/${path}`,{method:'POST'});if(sessionId!==state.sessionId)return;if(clearAnswer){mockRetry=false;mockRetryResponseId='';$('mock-answer').value='';}mockSpeechFeedbackPollGeneration+=1;await loadSession();toast(message);}catch(error){toast(error.message,true)}});}
$('mock-next').onclick=()=>navigateMockInterview('next','下一题');
$('mock-prev').onclick=()=>navigateMockInterview('previous','上一题');
$('mock-finish').onclick=()=>navigateMockInterview('finish','面试已结束',{clearAnswer:false});
let mockVoiceRecorder=null;
let mockVoiceChunks=[];
let mockVoiceStream=null;
function mockVoiceNote(msg){const n=$('mock-voice-note');if(n)n.textContent=msg||'';}
function clearObjectUrl(kind){const value=kind==='question'?mockQuestionAudioUrl:mockAnswerAudioUrl;if(value)URL.revokeObjectURL(value);if(kind==='question')mockQuestionAudioUrl='';else mockAnswerAudioUrl='';}
function clearAudioElement(id){const audio=$(id);if(!audio)return;audio.pause();audio.removeAttribute('src');audio.load();audio.classList.add('hidden');}
function pauseMockAnswerPlayback(){const audio=$('mock-answer-playback');if(audio&&!audio.paused)audio.pause();}
function clearMockQuestionAudio(){clearObjectUrl('question');clearAudioElement('mock-question-audio');}
function cancelMockAnswerAudioLoad(){mockAnswerAudioRequestSequence+=1;if(mockAnswerAudioAbort)mockAnswerAudioAbort.abort();mockAnswerAudioAbort=null;}
function clearMockVoiceRecordingTimers(){if(mockVoiceWarningTimer)clearTimeout(mockVoiceWarningTimer);if(mockVoiceLimitTimer)clearTimeout(mockVoiceLimitTimer);mockVoiceWarningTimer=null;mockVoiceLimitTimer=null;}
function clearMockAnswerExperience(){mockSpeechFeedbackPollGeneration+=1;cancelMockAnswerAudioLoad();clearObjectUrl('answer');mockAnswerAudioResponseId='';mockPendingRecordingId='';mockRestoredDraftId='';clearAudioElement('mock-answer-playback');renderSpeechFeedback(null);mockVoiceNote('');}
function restorePendingMockPlayback(){
  const pending=mockVoicePendingUpload;
  if(authenticationSuspended||!pending?.blob||pending.recordingSessionId!==state.sessionId||mockAnswerAudioUrl)return;
  mockAnswerAudioUrl=URL.createObjectURL(pending.blob);
  mockAnswerAudioResponseId='';
  const audio=$('mock-answer-playback');
  if(audio){audio.src=mockAnswerAudioUrl;audio.classList.remove('hidden');}
}
async function loadMockAnswerAudio(responseId,historical=false){
  if(historical){cancelMockQuestionSpeech();clearMockQuestionAudio();}
  cancelMockAnswerAudioLoad();
  const sequence=mockAnswerAudioRequestSequence;const sessionId=state.sessionId;const controller=new AbortController();mockAnswerAudioAbort=controller;mockAnswerAudioResponseId=responseId;
  try{const blob=await api.blob(`/api/mock-interviews/${sessionId}/answers/${responseId}/audio`,{signal:controller.signal});if(sequence!==mockAnswerAudioRequestSequence||sessionId!==state.sessionId||(!historical&&responseId!==mockDisplayedResponseId))return;clearObjectUrl('answer');mockAnswerAudioUrl=URL.createObjectURL(blob);const audio=$('mock-answer-playback');audio.src=mockAnswerAudioUrl;audio.classList.remove('hidden');}catch(error){if(error.name!=='AbortError'){if(mockAnswerAudioResponseId===responseId)mockAnswerAudioResponseId='';mockVoiceNote(error.message);}}finally{if(mockAnswerAudioAbort===controller)mockAnswerAudioAbort=null;}
}
function syncMockAnswerAudio(response){
  if(!response?.audio_file){if(mockVoicePendingUpload)return;cancelMockAnswerAudioLoad();clearObjectUrl('answer');mockAnswerAudioResponseId='';clearAudioElement('mock-answer-playback');return;}
  if(mockAnswerAudioUrl&&!mockAnswerAudioResponseId){mockAnswerAudioResponseId=response.id;return;}
  if(mockAnswerAudioResponseId!==response.id)loadMockAnswerAudio(response.id);
}
async function loadMockDraftAudio(recordingId){
  cancelMockAnswerAudioLoad();
  const sequence=mockAnswerAudioRequestSequence;const sessionId=state.sessionId;const controller=new AbortController();mockAnswerAudioAbort=controller;mockAnswerAudioResponseId=`draft:${recordingId}`;
  try{const blob=await api.blob(`/api/mock-interviews/${sessionId}/recordings/${recordingId}/audio`,{signal:controller.signal});if(sequence!==mockAnswerAudioRequestSequence||sessionId!==state.sessionId||mockPendingRecordingId!==recordingId)return;clearObjectUrl('answer');mockAnswerAudioUrl=URL.createObjectURL(blob);const audio=$('mock-answer-playback');audio.src=mockAnswerAudioUrl;audio.classList.remove('hidden');}catch(error){if(error.name!=='AbortError')mockVoiceNote(error.message);}finally{if(mockAnswerAudioAbort===controller)mockAnswerAudioAbort=null;}
}
function restoreMockAnswerDraft(draft){
  if(!draft?.recording_id||mockRestoredDraftId===draft.recording_id)return;
  mockRestoredDraftId=draft.recording_id;
  mockPendingRecordingId=draft.recording_id;
  if(draft.transcript?.trim())$('mock-answer').value=draft.transcript.trim();
  renderSpeechFeedback(draft.speech_delivery);
  void loadMockDraftAudio(draft.recording_id);
  mockVoiceNote(draft.transcription_status==='failed'?(draft.transcription_error||'录音已恢复，请回放后手动补充文字。'):'已恢复上次未提交的语音回答，可继续修改或提交。');
}
function cancelMockQuestionSpeech(){mockSpeechRequestSequence+=1;if(mockSpeechAbort)mockSpeechAbort.abort();mockSpeechAbort=null;}
function resetMockAudioExperience(){
  mockVoiceGeneration+=1;mockVoiceSessionId='';resetAsrPreview();
  mockVoiceStartPending=false;
  clearMockVoiceRecordingTimers();
  mockVoiceContext=null;
  cancelMockQuestionSpeech();
  if(mockVoiceRecorder){mockVoiceRecorder.ondataavailable=null;mockVoiceRecorder.onstop=null;try{if(mockVoiceRecorder.state!=='inactive')mockVoiceRecorder.stop();}catch{}mockVoiceRecorder=null;}
  if(mockVoiceStream)mockVoiceStream.getTracks().forEach(track=>track.stop());
  mockVoiceStream=null;mockVoiceChunks=[];mockCurrentSpeechKey='';mockDisplayedResponseId='';mockSpokenQuestionKey='';
  clearMockQuestionAudio();clearMockAnswerExperience();
  if($('mock-voice'))$('mock-voice').disabled=false;if($('mock-stop-voice'))$('mock-stop-voice').disabled=true;
}
async function speakCurrentMockQuestion(automatic=false){
  const session=state.session?.mock_session;const question=state.session?.mock_interview?.questions?.[session?.current_question_index];if(!state.sessionId||!question)return;
  if(mockVoiceCaptureBusy()){if(!automatic)toast('请先停止并保存当前回答录音',true);return;}
  if(automatic&&(state.view!=='mock'||document.visibilityState!=='visible'||!$('view-mock')?.classList.contains('active'))){mockSpokenQuestionKey='';return;}
  pauseMockAnswerPlayback();cancelMockQuestionSpeech();const requestSequence=mockSpeechRequestSequence;const requestedSessionId=state.sessionId;const requestedSpeechKey=mockCurrentSpeechKey;const responseId=mockDisplayedResponseId;const controller=new AbortController();mockSpeechAbort=controller;
  const button=$('mock-speak-question');if(button)button.disabled=true;
  try{const query=responseId?`?response_id=${encodeURIComponent(responseId)}`:'',blob=await api.blob(`/api/mock-interviews/${requestedSessionId}/questions/${question.id}/speech${query}`,{method:'POST',signal:controller.signal});if(requestSequence!==mockSpeechRequestSequence||requestedSessionId!==state.sessionId||requestedSpeechKey!==mockCurrentSpeechKey)return;clearMockQuestionAudio();mockQuestionAudioUrl=URL.createObjectURL(blob);const audio=$('mock-question-audio');audio.src=mockQuestionAudioUrl;audio.classList.remove('hidden');await audio.play();}catch(error){if(error.name!=='AbortError'&&!automatic)toast(`问题朗读失败：${error.message}`,true);}finally{if(mockSpeechAbort===controller)mockSpeechAbort=null;if(requestSequence===mockSpeechRequestSequence&&button)button.disabled=false;}
}
$('mock-speak-question').onclick=()=>speakCurrentMockQuestion(false);
$('mock-answer-playback')?.addEventListener('play',()=>{cancelMockQuestionSpeech();clearMockQuestionAudio();});
$('mock-auto-speak').onchange=e=>{localStorage.setItem('interviewos.autoSpeak',e.target.checked?'1':'0');if(e.target.checked&&state.view==='mock'){mockSpokenQuestionKey='';renderMock();}};
$('mock-auto-speak').checked=localStorage.getItem('interviewos.autoSpeak')!=='0';
$('mock-answer').addEventListener('input',()=>{mockAnswerEditRevision+=1;});
function pauseContinuousListeningForHiddenPage(){
  const sessionId=state.sessionId;
  const pendingGuard=pendingLiveCaptureStart?.sessionId===sessionId&&pendingLiveCaptureStart.captureRole==='guard';
  if((!liveContinuousMode&&!pendingGuard&&!liveContinuousGuardId)||state.session?.live_interview?.status!=='active'||desktopShutdownPreparing)return;
  visibilityPauseRequested=true;
  // Invalidate a permission prompt immediately. If this start was nested in a
  // resume transition, wait for that transition before issuing the compensating
  // pause so no hidden tab can become an active microphone owner afterwards.
  liveCaptureGeneration+=1;
  if(pendingGuard){
    const pending=pendingLiveCaptureStart;
    pendingLiveCaptureStart=null;
    void cancelLiveAudioCapture(pending.sessionId,pending.recordingId).catch(()=>{});
  }
  visibilityPausePromise=visibilityPausePromise.catch(()=>{}).then(async()=>{
    await mediaTransitionPromise.catch(()=>{});
    await mediaFlushTail.catch(()=>{});
    if(sessionId!==state.sessionId||state.session?.live_interview?.status!=='active'||desktopShutdownPreparing)return;
    const pauseHandler=$('live-pause')?.onclick;
    if(typeof pauseHandler==='function')await pauseHandler();
  }).catch(error=>toast(`页面进入后台后无法确认安全暂停：${error.message}`,true));
}
document.addEventListener('visibilitychange',()=>{
  if(document.visibilityState!=='visible'){
    cancelMockQuestionSpeech();
    clearMockQuestionAudio();
    pauseMockAnswerPlayback();
    pauseContinuousListeningForHiddenPage();
    return;
  }
  if(visibilityPauseRequested){visibilityPauseRequested=false;toast('页面进入后台时已请求安全暂停连续监听；请确认状态后手动恢复。');}
  void recoverPageBoundary();
  void renewPendingLiveCaptureLeases();
});
window.addEventListener('pageshow',()=>{void recoverPageBoundary();void renewPendingLiveCaptureLeases();});
function renderSpeechFeedback(feedback){const node=$('mock-speech-feedback');if(!node)return;if(!feedback){node.classList.add('hidden');node.innerHTML='';return;}const rows=[['语速',feedback.pace],['停顿',feedback.pauses],['填充词',feedback.fillers],['音量稳定',feedback.volume],['语调',feedback.intonation],['清晰度',feedback.clarity]].filter(([,value])=>value&&value!=='无法判断');node.classList.remove('hidden');node.innerHTML=`<div class="review-subtitle">语音表达辅导 · ${feedback.source==='audio_model'?'音频模型':'本地指标'}</div>${rows.map(([name,value])=>`<p><strong>${esc(name)}</strong><span>${esc(value)}</span></p>`).join('')}${list('可执行改进',feedback.improvements||[])}<small>${esc(feedback.disclaimer||'仅用于表达训练，不进入录用评价。')}</small>`;}
async function pollMockSpeechFeedback(sessionId,recordingId,generation){const pollGeneration=++mockSpeechFeedbackPollGeneration;for(let attempt=0;attempt<20;attempt+=1){await new Promise(resolve=>setTimeout(resolve,1500));if(pollGeneration!==mockSpeechFeedbackPollGeneration||generation!==mockVoiceGeneration||sessionId!==state.sessionId||mockPendingRecordingId!==recordingId)return;try{const data=await api(`/api/mock-interviews/${sessionId}/recordings/${recordingId}/speech-feedback`);if(pollGeneration!==mockSpeechFeedbackPollGeneration||mockPendingRecordingId!==recordingId)return;renderSpeechFeedback(data.speech_feedback);if(data.status==='completed'){mockVoiceNote('语音表达分析已完成；可回放、修改文字或提交回答');return;}}catch(error){if(attempt>2){mockVoiceNote(`表达分析暂不可用：${error.message}`);return;}}}mockVoiceNote('转写已完成；深度语音分析仍在后台处理');}
$('mock-voice').onclick=async()=>{
  if(mediaTransitionInProgress||mediaFlushInProgress||authenticationSuspended||!runtimeSettingsReady){toast('录音正在保存、配置未加载或需要重新登录，请稍后重试',true);return;}
  if(!capturePageIsVisible()){toast('页面在后台，已拒绝启动麦克风；请回到当前页面后重试',true);return;}
  if(liveMicrophoneCaptureActive()){toast('实时面试仍在使用麦克风，请先暂停并保存录音',true);return;}
  if(!ensureMicrophoneAvailable())return;
  const intentSessionId=state.sessionId;
  if(!await discardMockDraftIfNeeded())return;
  if(intentSessionId!==state.sessionId||mediaTransitionInProgress||mediaFlushInProgress||authenticationSuspended||!runtimeSettingsReady||desktopShutdownPreparing||liveMicrophoneCaptureActive()){toast('会话或录音状态已变化，本次麦克风启动已取消',true);return;}
  const recordingSessionId=state.sessionId;
  const generation=++mockVoiceGeneration;
  const mockSession=state.session?.mock_session;
  const currentQuestion=state.session?.mock_interview?.questions?.[mockSession?.current_question_index];
  if(!currentQuestion){toast('当前没有可回答的问题',true);return;}
  cancelMockQuestionSpeech();
  clearMockQuestionAudio();
  pauseMockAnswerPlayback();
  const retryVoiceTarget=mockRetry?(mockSession.responses||[]).find(response=>response.id===mockRetryResponseId):null;
  const voiceContext={
    questionId:mockSession.pending_parent_question_id||currentQuestion.id,
    question:retryVoiceTarget?.question||mockSession.pending_follow_up||currentQuestion.question,
    retry:!!retryVoiceTarget,
    retryResponseId:retryVoiceTarget?.id||'',
    answerEditRevision:mockAnswerEditRevision,
  };
  clearMockAnswerExperience();
  mockVoiceSessionId=recordingSessionId;
  mockVoiceContext=voiceContext;
  $('mock-voice').disabled=true;
  mockVoiceStartPending=true;
  renderMock();
  const startPromise=(async()=>{
    try{
      const stream=await navigator.mediaDevices.getUserMedia(microphoneConstraints());
      if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||!runtimeSettingsReady||authenticationSuspended||desktopShutdownPreparing||liveMicrophoneCaptureActive()){
        stream.getTracks().forEach(track=>track.stop());
        return;
      }
      mockVoiceStream=stream;
      const preferred=liveAudioType();
      const recorder=new MediaRecorder(stream,preferred?{mimeType:preferred}:undefined);
      mockVoiceRecorder=recorder;
      mockVoiceChunks=[];
      recorder.ondataavailable=e=>{
        if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId||!e.data.size)return;
        mockVoiceChunks.push(e.data);
        previewRecordedChunks(mockVoiceChunks,recorder.mimeType||preferred,text=>{
          if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId||mockAnswerEditRevision!==voiceContext.answerEditRevision)return;
          $('mock-answer').value=text;
          mockVoiceNote('实时转写草稿 · 停止后确认最终文本');
        },recordingSessionId);
      };
      mockVoiceRecorderDone=new Promise(resolve=>recorder.addEventListener('stop',resolve,{once:true}));
      recorder.start(1000);
      clearMockVoiceRecordingTimers();
      mockVoiceWarningTimer=setTimeout(()=>{if(generation===mockVoiceGeneration&&mockVoiceRecorder===recorder)mockVoiceNote('录音已接近 10 分钟上限，将自动停止并保存。');},MOCK_RECORDING_WARNING_MS);
      mockVoiceLimitTimer=setTimeout(()=>{if(generation===mockVoiceGeneration&&mockVoiceRecorder===recorder){mockVoiceNote('已达 10 分钟安全上限，正在自动停止并生成完整转写…');void finalizeMockVoiceRecording();}},MOCK_RECORDING_LIMIT_MS);
      $('mock-stop-voice').disabled=false;
      mockVoiceNote('正在录音并显示实时转写草稿…');
      renderMock();
    }catch(error){
      if(generation!==mockVoiceGeneration)return;
      if(mockVoiceStream){mockVoiceStream.getTracks().forEach(track=>track.stop());mockVoiceStream=null;}
      mockVoiceRecorder=null;
      mockVoiceRecorderDone=Promise.resolve();
      mockVoiceChunks=[];
      mockVoiceSessionId='';
      mockVoiceContext=null;
      if(!desktopShutdownPreparing)toast(`无法使用麦克风：${error.message}`,true);
    }
  })();
  mockVoiceStartPromise=startPromise;
  try{await startPromise;}finally{if(generation===mockVoiceGeneration)mockVoiceStartPending=false;renderMock();}
};

async function persistMockVoicePendingUpload({quiet=false}={}){
  const pending=mockVoicePendingUpload;
  if(!pending)return;
  const {blob,recordingSessionId,voiceContext,generation}=pending;
  const stillCurrent=()=>mockVoicePendingUpload===pending&&recordingSessionId===state.sessionId;
  try{
    const wav=await encodeBlobAsWav(blob);
    if(wav.size>MOCK_AUDIO_UPLOAD_LIMIT_BYTES)throw new Error('录音转换后超过 25 MB；原录音仍可回放，请删除草稿后分段重录');
    const form=new FormData();
    form.append('file',wav,wav.type==='audio/wav'?'answer.wav':liveAudioFilename(wav.type,'answer'));
    form.append('question_id',voiceContext?.questionId||'');
    form.append('question',voiceContext?.question||'');
    form.append('retry',voiceContext?.retry?'true':'false');
    if(voiceContext?.retryResponseId)form.append('retry_response_id',voiceContext.retryResponseId);
    const data=await mediaApi(`/api/mock-interviews/${recordingSessionId}/transcribe`,{method:'POST',body:form});
    if(mockVoicePendingUpload===pending)mockVoicePendingUpload=null;
    if(recordingSessionId!==state.sessionId)return data;
    mockPendingRecordingId=data.recording_id||'';
    const answerWasEdited=mockAnswerEditRevision!==voiceContext?.answerEditRevision;
    if(data.text?.trim()&&!answerWasEdited)$('mock-answer').value=data.text.trim();
    if(state.session?.mock_session&&data.draft){state.session.mock_session.answer_draft=data.draft;mockRestoredDraftId=data.draft.recording_id||mockRestoredDraftId;}
    renderSpeechFeedback(data.speech_feedback);
    $('mock-stop-voice').disabled=true;
    if(!desktopShutdownPreparing){
      mockVoiceNote(data.transcription_status==='failed'?(data.transcription_error||'录音已保存，请回放后手动补充文字。'):answerWasEdited?'最终转写已保存；已保留你在等待期间手动修改的文字。':'最终转写已确认，可立即修改或提交；深度语音分析在后台继续');
      pollMockSpeechFeedback(recordingSessionId,mockPendingRecordingId,generation);
    }
    return data;
  }catch(error){
    if(stillCurrent()){
      mockPendingRecordingId='';
      $('mock-stop-voice').disabled=false;
      if(!quiet){
        mockVoiceNote('录音尚未保存；点击“停止并转写”可重试，或明确删除后继续');
        toast(`语音保存失败：${error.message}`,true);
      }
    }
    throw error;
  }
}

async function finalizeMockVoiceRecording({quiet=false}={}){
  if(mockVoiceFinalizePromise)return mockVoiceFinalizePromise;
  mockVoiceFinalizePromise=(async()=>{
    clearMockVoiceRecordingTimers();
    if(mockVoicePendingUpload)return persistMockVoicePendingUpload({quiet});
    if(mockVoiceStartPending){
      // A system permission prompt cannot be cancelled. Wait briefly for an
      // already-approved capture, then invalidate a late result during exit.
      await Promise.race([
        mockVoiceStartPromise.catch(()=>{}),
        new Promise(resolve=>setTimeout(resolve,3000))
      ]);
    }
    const recorder=mockVoiceRecorder;
    const recorderDone=mockVoiceRecorderDone;
    const stream=mockVoiceStream;
    const recordingSessionId=mockVoiceSessionId;
    const voiceContext=mockVoiceContext;
    const generation=mockVoiceGeneration;
    if(!recorder){
      if(mockVoiceStartPending){mockVoiceGeneration+=1;mockVoiceSessionId='';mockVoiceContext=null;}
      mockVoiceStartPending=false;
      if(!quiet)mockVoiceNote('');
      return;
    }
    $('mock-stop-voice').disabled=true;
    if(!quiet)mockVoiceNote('正在确认最终转写…');
    resetAsrPreview();
    if(recorder.state!=='inactive'){
      try{recorder.requestData();}catch{}
      recorder.stop();
    }
    await recorderDone;
    await Promise.resolve();
    const chunks=[...mockVoiceChunks];
    if(mockVoiceRecorder===recorder)mockVoiceRecorder=null;
    if(mockVoiceStream===stream)mockVoiceStream=null;
    if(mockVoiceSessionId===recordingSessionId)mockVoiceSessionId='';
    if(mockVoiceContext===voiceContext)mockVoiceContext=null;
    mockVoiceChunks=[];
    stream?.getTracks().forEach(track=>track.stop());
    if(generation!==mockVoiceGeneration||recordingSessionId!==state.sessionId||!chunks.length)return;
    const blob=new Blob(chunks,{type:recorder.mimeType||'audio/webm'});
    mockVoicePendingUpload={blob,recordingSessionId,voiceContext,generation};
    clearObjectUrl('answer');
    clearAudioElement('mock-answer-playback');
    renderSpeechFeedback(null);
    mockAnswerAudioUrl=URL.createObjectURL(blob);
    mockAnswerAudioResponseId='';
    const playback=$('mock-answer-playback');
    playback.src=mockAnswerAudioUrl;
    playback.classList.remove('hidden');
    return persistMockVoicePendingUpload({quiet});
  })();
  try{return await mockVoiceFinalizePromise;}finally{mockVoiceFinalizePromise=null;renderMock();}
}

$('mock-stop-voice').onclick=()=>finalizeMockVoiceRecording();
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

function sessionPartQueueBytes(sessionId='') {
  return sessionPartQueue.reduce((total,part)=>total+(!sessionId||part.sessionId===sessionId?part.blob.size:0),0);
}

function sessionPartsPending(sessionId=state.sessionId) {
  return sessionPartQueue.some(part=>part.sessionId===sessionId);
}

function clearSessionRotationTimer() {
  if(sessionRotationTimer)clearTimeout(sessionRotationTimer);
  sessionRotationTimer=null;
}

function disableSessionRecorderRotation() {
  sessionRotationEnabled=false;
  clearSessionRotationTimer();
}

function stopSessionMediaStream(stream=sessionMediaStream) {
  if(!stream)return;
  stream.getTracks().forEach(track=>track.stop());
  if(sessionMediaStream===stream)sessionMediaStream=null;
}

function clearSessionRecordingIdentityIfIdle() {
  if(sessionCurrentPart||sessionMediaStream||sessionPartQueue.length||sessionRecorderStartPending)return;
  sessionRecordingSessionId='';
  sessionRecordingPartId='';
  sessionRecordingArchiveRevision=0;
  sessionRecordingCaptureEpoch=0;
  sessionPartSequence=0;
  sessionNextUploadSequence=1;
  sessionFinalizedPartSequences.clear();
  sessionChunks=[];
  sessionRecorderDone=Promise.resolve();
}

function sessionQueueAtHighWater(extraPart=null) {
  const count=sessionPartQueue.length+(extraPart?1:0);
  const bytes=sessionPartQueueBytes()+(extraPart?extraPart.bytes:0);
  return count>=SESSION_PART_QUEUE_HIGH_WATER||bytes>=SESSION_PART_QUEUE_BYTES_HIGH_WATER;
}

function enqueueSessionRecordingPart(part) {
  if(part.enqueued)return;
  part.enqueued=true;
  sessionFinalizedPartSequences.add(part.sequence);
  if(part.bytes){
    const blob=new Blob(part.chunks,{type:part.mimeType});
    sessionPartQueue.push({
      blob,
      type:part.mimeType,
      sessionId:part.sessionId,
      recordingId:part.recordingId,
      archiveRevision:part.archiveRevision,
      captureEpoch:part.captureEpoch,
      sequence:part.sequence,
    });
    // A successor can finish before its predecessor during a boundary race.
    // Sorting plus the upload sequence fence preserves archive chronology.
    sessionPartQueue.sort((left,right)=>left.sequence-right.sequence);
  }
  part.chunks=[];
  if(!sessionPartUploadsSuppressed&&!part.deferUpload)void uploadSessionAudio().catch(()=>{});
}

function createSessionRecordingPart(stream,metadata) {
  const preferred=liveAudioType();
  const recorder=new MediaRecorder(stream,preferred?{mimeType:preferred}:undefined);
  let resolveDone;
  let resolveStarted;
  let rejectStarted;
  let doneSettled=false;
  const part={
    recorder,
    chunks:[],
    bytes:0,
    mimeType:recorder.mimeType||preferred||'audio/webm',
    sessionId:metadata.sessionId,
    recordingId:newRecordingPartId(),
    archiveRevision:metadata.archiveRevision,
    captureEpoch:metadata.captureEpoch,
    sequence:++sessionPartSequence,
    stopIntent:'',
    enqueued:false,
    deferUpload:Boolean(metadata.deferUpload),
  };
  part.done=new Promise(resolve=>{resolveDone=resolve;});
  part.settleDone=()=>{if(!doneSettled){doneSettled=true;resolveDone();}};
  part.started=new Promise((resolve,reject)=>{resolveStarted=resolve;rejectStarted=reject;});
  recorder.addEventListener('start',()=>resolveStarted(),{once:true});
  recorder.addEventListener('error',event=>{
    const error=event.error||new Error('全场录音器运行失败');
    rejectStarted(error);
    if(!part.stopIntent)showSessionRecordingError(`全场录音意外中断：${error.message||error}`,part.sessionId);
  });
  recorder.ondataavailable=event=>{
    if(!event.data.size)return;
    part.chunks.push(event.data);
    part.bytes+=event.data.size;
    if(sessionCurrentPart===part){
      sessionChunks=part.chunks;
      if(part.bytes>=SESSION_RECORDING_PART_BYTES)queueSessionRecorderRotation('size',part);
    }
  };
  recorder.addEventListener('stop',()=>{
    enqueueSessionRecordingPart(part);
    if(sessionCurrentPart===part){
      sessionCurrentPart=null;
      sessionRecorder=null;
      sessionChunks=[];
      sessionRecordingPartId='';
      sessionRecordingActive=false;
      if(!part.stopIntent){
        disableSessionRecorderRotation();
        showSessionRecordingError('全场录音意外停止；已录分片仍保留并等待保存',part.sessionId);
      }
      if(!sessionRotationEnabled)stopSessionMediaStream();
    }
    part.settleDone();
    renderLive();
  },{once:true});
  return part;
}

async function startSessionRecordingPart(part) {
  let startTimer;
  try{
    part.recorder.start(1000);
    await Promise.race([
      part.started,
      new Promise((_,reject)=>{startTimer=setTimeout(()=>reject(new Error('全场录音器启动超时')),3000);}),
    ]);
    return part;
  }catch(error){
    // A synchronous start failure never emits `stop`; settle the local waiter
    // explicitly so shutdown and session switching cannot hang forever.
    if(part.recorder.state==='inactive'){
      part.stopIntent=part.stopIntent||'startup-failed';
      part.enqueued=true;
      part.settleDone();
    }
    throw error;
  }finally{clearTimeout(startTimer);}
}

function scheduleSessionRecorderRotation(part=sessionCurrentPart) {
  clearSessionRotationTimer();
  if(!sessionRotationEnabled||!part)return;
  sessionRotationTimer=setTimeout(()=>queueSessionRecorderRotation('duration',part),SESSION_RECORDING_PART_MS);
}

function activateSessionRecordingPart(part) {
  sessionCurrentPart=part;
  sessionRecorder=part.recorder;
  sessionRecorderDone=part.done;
  sessionChunks=part.chunks;
  sessionRecordingMimeType=part.mimeType;
  sessionRecordingPartId=part.recordingId;
  sessionRecordingActive=true;
  scheduleSessionRecorderRotation(part);
  renderLive();
}

function requestSessionPartStop(part,intent) {
  if(!part)return;
  part.stopIntent=part.stopIntent||intent;
  if(part.recorder.state==='inactive')return;
  try{part.recorder.requestData();}catch{}
  try{part.recorder.stop();}catch{}
}

async function stopSessionCaptureForBackpressure(part,message) {
  disableSessionRecorderRotation();
  requestSessionPartStop(part,'backpressure');
  await part.done;
  stopSessionMediaStream();
  sessionRecordingActive=false;
  showSessionRecordingError(message,part.sessionId);
}

async function performSessionRecorderRotation(trigger,expectedPart) {
  const oldPart=sessionCurrentPart;
  if(!sessionRotationEnabled||!oldPart||oldPart!==expectedPart||oldPart.recorder.state!=='recording')return;
  clearSessionRotationTimer();
  if(sessionQueueAtHighWater(oldPart)){
    await stopSessionCaptureForBackpressure(oldPart,'全场录音待保存队列已达安全上限，已停止继续收音；原始分片仍保留');
    return;
  }
  const stream=sessionMediaStream;
  if(!stream)throw new Error('全场录音流已丢失');
  const successor=createSessionRecordingPart(stream,{
    sessionId:oldPart.sessionId,
    archiveRevision:oldPart.archiveRevision,
    captureEpoch:oldPart.captureEpoch,
    deferUpload:true,
  });
  // During the two-recorder overlap either stop event may arrive first. Hold
  // uploads until their order is known, then drain the sequence-sorted queue.
  oldPart.deferUpload=true;
  try{
    // Start the successor on the same MediaStream before sealing the old part.
    // This intentionally allows a tiny overlap instead of dropping speech.
    await startSessionRecordingPart(successor);
  }catch(error){
    requestSessionPartStop(successor,'rotation-failed');
    await successor.done.catch(()=>{});
    await stopSessionCaptureForBackpressure(oldPart,`全场录音分片轮换失败，已停止继续收音：${error.message}`);
    if(!sessionPartUploadsSuppressed)void uploadSessionAudio().catch(()=>{});
    throw error;
  }
  if(!sessionRotationEnabled||sessionCurrentPart!==oldPart){
    requestSessionPartStop(successor,'boundary');
    requestSessionPartStop(oldPart,'boundary');
    await Promise.all([successor.done,oldPart.done]);
    stopSessionMediaStream(stream);
    if(!sessionPartUploadsSuppressed)void uploadSessionAudio().catch(()=>{});
    return;
  }
  oldPart.deferUpload=false;
  successor.deferUpload=false;
  activateSessionRecordingPart(successor);
  requestSessionPartStop(oldPart,`rotate-${trigger}`);
  await oldPart.done;
  // A pause/finish can arrive while the two recorders overlap. Its rotation
  // disable flag wins and the newly activated successor is sealed immediately.
  if(!sessionRotationEnabled&&sessionCurrentPart===successor){
    requestSessionPartStop(successor,'boundary');
    await successor.done;
    stopSessionMediaStream(stream);
  }
}

function queueSessionRecorderRotation(trigger,part=sessionCurrentPart) {
  if(!sessionRotationEnabled||sessionRotationInProgress||!part||part!==sessionCurrentPart)return sessionRotationPromise;
  sessionRotationInProgress=true;
  sessionRotationPromise=performSessionRecorderRotation(trigger,part)
    .catch(error=>{
      if(part.sessionId===state.sessionId)toast(`全场录音分片失败：${error.message}`,true);
    })
    .finally(()=>{sessionRotationInProgress=false;renderLive();});
  return sessionRotationPromise;
}

function requestSessionRecorderStop(intent='boundary') {
  disableSessionRecorderRotation();
  sessionRecordingActive=false;
  requestSessionPartStop(sessionCurrentPart,intent);
  if(!sessionCurrentPart&&!sessionRotationInProgress)stopSessionMediaStream();
}

async function startSessionRecorder() {
  if(mediaFlushInProgress||authenticationSuspended||!runtimeSettingsReady||liveStatusUncertainSessionId===state.sessionId)return false;
  if(!capturePageIsVisible()){showSessionRecordingError('页面在后台，已拒绝启动全场录音');return false;}
  if(mockVoiceCaptureBusy()){
    showSessionRecordingError('模拟回答仍在录制或保存，不能同时启动全场录音');
    return false;
  }
  cancelMockQuestionSpeech();clearMockQuestionAudio();pauseMockAnswerPlayback();
  const unavailable=microphoneAvailabilityMessage();
  const pendingParts=sessionPartQueue.length>0;
  if(sessionRecorder||sessionMediaStream||sessionRecorderStartPending||sessionStopInProgress||pendingParts||sessionAudioUploading||unavailable){
    showSessionRecordingError(unavailable||(pendingParts||sessionAudioUploading?'上一批全场录音尚未保存':'全场录音已在运行'));
    return false;
  }
  const recordingSessionId=state.sessionId;
  const recordingCaptureEpoch=Number(state.session?.live_interview?.capture_epoch);
  if(!Number.isInteger(recordingCaptureEpoch))return false;
  const generation=++sessionRecorderStartGeneration;
  sessionRecorderStartPending=true;
  sessionRecorderStartSessionId=recordingSessionId;
  sessionRecorderStartCaptureEpoch=recordingCaptureEpoch;
  sessionRecordingError='';
  sessionRecordingErrorSessionId='';
  renderLive();
  let pendingStream=null;
  sessionRecorderStartPromise=navigator.mediaDevices.getUserMedia(microphoneConstraints()).then(async stream=>{
    pendingStream=stream;
    if(generation!==sessionRecorderStartGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||!runtimeSettingsReady||authenticationSuspended||liveStatusUncertainSessionId===recordingSessionId||state.session?.live_interview?.status!=='active'||desktopShutdownPreparing){
      stopSessionMediaStream(stream);
      return false;
    }
    sessionPendingMediaStream=stream;
    if(!await confirmRemoteLiveCaptureAuthority(recordingSessionId)||generation!==sessionRecorderStartGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||state.session?.live_interview?.status!=='active'||state.session?.live_interview?.capture_epoch!==recordingCaptureEpoch||!runtimeSettingsReady||authenticationSuspended||desktopShutdownPreparing){
      stopSessionMediaStream(stream);
      if(sessionPendingMediaStream===stream)sessionPendingMediaStream=null;
      throw new Error('实时会话已在另一窗口停止');
    }
    if(sessionPendingMediaStream===stream)sessionPendingMediaStream=null;
    sessionMediaStream=stream;
    sessionRecordingSessionId=recordingSessionId;
    sessionRecordingArchiveRevision=Number(state.session?.live_interview?.audio_archive_revision||0);
    sessionRecordingCaptureEpoch=recordingCaptureEpoch;
    sessionPartSequence=0;
    sessionNextUploadSequence=1;
    sessionFinalizedPartSequences.clear();
    sessionAudioSaveError='';
    sessionAudioErrorSessionId='';
    sessionRecordingError='';
    sessionRecordingErrorSessionId='';
    sessionRotationEnabled=true;
    const part=createSessionRecordingPart(stream,{
      sessionId:recordingSessionId,
      archiveRevision:sessionRecordingArchiveRevision,
      captureEpoch:recordingCaptureEpoch,
    });
    try{
      await startSessionRecordingPart(part);
      activateSessionRecordingPart(part);
    }catch(error){
      disableSessionRecorderRotation();
      requestSessionPartStop(part,'startup-failed');
      await part.done.catch(()=>{});
      stopSessionMediaStream(stream);
      clearSessionRecordingIdentityIfIdle();
      throw error;
    }
    return true;
  }).catch(error=>{
    stopSessionMediaStream(pendingStream);
    if(sessionPendingMediaStream===pendingStream)sessionPendingMediaStream=null;
    if(generation===sessionRecorderStartGeneration&&recordingSessionId===state.sessionId){
      showSessionRecordingError(error?.message||'无法访问麦克风，未在录制全场录音（转写不受影响）',recordingSessionId);
      renderLive();
    }
    return false;
  }).finally(()=>{
    if(generation===sessionRecorderStartGeneration){sessionRecorderStartPending=false;sessionRecorderStartSessionId='';sessionRecorderStartCaptureEpoch=0;}
    renderLive();
  });
  return sessionRecorderStartPromise;
}

function showSessionRecordingError(message,sessionId=sessionRecordingSessionId||state.sessionId) {
  sessionRecordingActive=false;
  sessionRecordingError=message;
  sessionRecordingErrorSessionId=sessionId;
  const status=$('session-recording-status');
  if(status){status.textContent=message;status.classList.add('error');}
}

function stopSessionRecorder() {
  if(sessionStopInProgress)return sessionStopPromise;
  sessionStopInProgress=true;
  sessionStopPromise=(async()=>{
    requestSessionRecorderStop('boundary');
    await sessionRotationPromise.catch(()=>{});
    const part=sessionCurrentPart;
    if(part){requestSessionPartStop(part,'boundary');await part.done;}
    stopSessionMediaStream();
    sessionRecordingActive=false;
    clearSessionRecordingIdentityIfIdle();
  })().finally(()=>{sessionStopInProgress=false;renderLive();});
  return sessionStopPromise;
}

function uploadSessionAudio() {
  if(sessionAudioUploading)return sessionAudioUploadPromise;
  if(!sessionPartQueue.length)return Promise.resolve();
  sessionAudioUploading=true;
  sessionAudioSaveError='';
  sessionAudioErrorSessionId='';
  renderLive();
  sessionAudioUploadPromise=persistSessionAudio().finally(()=>{sessionAudioUploading=false;clearSessionRecordingIdentityIfIdle();renderLive();});
  return sessionAudioUploadPromise;
}

function mergeSessionAudioArchiveState(incomingState,sessionId,expectedArchiveRevision) {
  if(sessionId!==state.sessionId||!incomingState?.live_interview)return;
  if(!state.session){state.session=incomingState;return;}
  const incomingLive=incomingState.live_interview;
  const currentLive=state.session.live_interview||{};
  const incomingRevision=Number(incomingLive.audio_archive_revision||0);
  const currentRevision=Number(currentLive.audio_archive_revision||0);
  if(incomingRevision!==expectedArchiveRevision||incomingRevision<currentRevision)return;
  // Whole-session uploads now run in the background while transcript and
  // suggestion mutations continue. Merge only the append-only audio manifest;
  // assigning the response's whole state could roll those newer mutations back.
  const partsById=new Map();
  for(const part of [...(currentLive.audio_parts||[]),...(incomingLive.audio_parts||[])])partsById.set(String(part.id),part);
  const audioParts=[...partsById.values()].sort((left,right)=>String(left.audio_saved_at||'').localeCompare(String(right.audio_saved_at||'')));
  const latest=audioParts.at(-1);
  state.session={
    ...state.session,
    live_interview:{
      ...currentLive,
      audio_archive_revision:incomingRevision,
      audio_parts:audioParts,
      audio_file:latest?.audio_file||incomingLive.audio_file||currentLive.audio_file||'',
      audio_size_bytes:latest?.audio_size_bytes||incomingLive.audio_size_bytes||currentLive.audio_size_bytes||0,
      audio_payload_sha256:latest?.payload_sha256||incomingLive.audio_payload_sha256||currentLive.audio_payload_sha256||'',
      audio_saved_at:latest?.audio_saved_at||incomingLive.audio_saved_at||currentLive.audio_saved_at||null,
    },
  };
}

function replaceSessionAudioArchiveState(incomingState,sessionId) {
  if(sessionId!==state.sessionId||!incomingState?.live_interview)return;
  if(!state.session){state.session=incomingState;return;}
  const incomingLive=incomingState.live_interview;
  const currentLive=state.session.live_interview||{};
  state.session={
    ...state.session,
    live_interview:{
      ...currentLive,
      audio_archive_revision:Number(incomingLive.audio_archive_revision||0),
      audio_parts:[...(incomingLive.audio_parts||[])],
      audio_file:incomingLive.audio_file||'',
      audio_size_bytes:Number(incomingLive.audio_size_bytes||0),
      audio_payload_sha256:incomingLive.audio_payload_sha256||'',
      audio_saved_at:incomingLive.audio_saved_at||null,
      last_audio_delete_operation_id:incomingLive.last_audio_delete_operation_id||'',
    },
  };
}

async function persistSessionAudio() {
  while(sessionPartQueue.length){
    while(sessionFinalizedPartSequences.has(sessionNextUploadSequence)&&!sessionPartQueue.some(item=>item.sequence===sessionNextUploadSequence)){
      sessionFinalizedPartSequences.delete(sessionNextUploadSequence);
      sessionNextUploadSequence+=1;
    }
    const part=sessionPartQueue.find(item=>item.sequence===sessionNextUploadSequence);
    // A later overlap part stopped first. It stays in memory until the earlier
    // recorder emits its final data and stop event.
    if(!part)return;
    const recordingSessionId=part.sessionId;
    if(!recordingSessionId)throw new Error('全场录音缺少所属会话，已停止保存以避免串写');
    if(!part.recordingId)throw new Error('全场录音缺少分片标识，已停止保存以避免覆盖');
    try{
      const form=new FormData();
      const filename=liveAudioFilename(part.blob.type,`session-${recordingSessionId.slice(0,8)}-${part.sequence}`);
      form.append('file',part.blob,filename);
      form.append('recording_id',part.recordingId);
      // This is the delete generation captured once when the listening round
      // starts. Successful appends do not advance it, so every part uses it.
      form.append('expected_audio_revision',String(part.archiveRevision));
      const data=await mediaApi(`/api/live-interviews/${recordingSessionId}/audio/final`,{method:'POST',body:form});
      mergeSessionAudioArchiveState(data.state,recordingSessionId,part.archiveRevision);
      if(sessionPartQueue[0]===part)sessionPartQueue.shift();
      else sessionPartQueue=sessionPartQueue.filter(item=>item!==part);
      sessionFinalizedPartSequences.delete(part.sequence);
      sessionNextUploadSequence+=1;
      sessionAudioSaveError='';
      sessionAudioErrorSessionId='';
    }catch(error){
      sessionAudioSaveError=`全场录音第 ${part.sequence} 段尚未保存：${error.message}`;
      sessionAudioErrorSessionId=recordingSessionId;
      disableSessionRecorderRotation();
      requestSessionRecorderStop('upload-failed');
      await sessionRotationPromise.catch(()=>{});
      const activePart=sessionCurrentPart;
      if(activePart){requestSessionPartStop(activePart,'upload-failed');await activePart.done;}
      stopSessionMediaStream();
      showSessionRecordingError('全场录音因分片保存失败已停止；原始分片仍保留，请重试',recordingSessionId);
      if(recordingSessionId===state.sessionId)toast(`全场录音保存失败：${error.message}`,true);
      throw error;
    }
  }
  clearSessionRecordingIdentityIfIdle();
}

async function startLiveInterview(){
  if(!await ensureSession())return;
  if(!runtimeSettingsReady){toast('运行配置尚未成功加载，不能启动麦克风；请重新登录或在设置页重试',true);return;}
  if(mockVoiceCaptureBusy()){toast('请先停止并保存当前模拟回答录音，再启动实时面试',true);return;}
  const operationSessionId=state.sessionId;
  const operationGeneration=snapshotLiveGeneration();
  const operationId=newRecordingPartId();
  const expectedActiveGeneration=transitionedLiveGeneration(operationGeneration,'active',operationId);
  let fullRecordingStarted=false;
  try{
    await runMediaTransition(async()=>{
      const consentConfirmed=$('live-consent').checked;
      await preflightLiveGeneration(operationSessionId,operationGeneration,'启动');
      const data=await requestLiveStart(operationSessionId,consentConfirmed,operationGeneration,operationId);
      if(operationSessionId!==state.sessionId)throw new Error('会话已变更，实时面试未启动');
      if(!liveTransitionMatches(data.state?.live_interview,operationGeneration,'active',operationId))throw liveOwnershipChangedError(data,'启动新的监听轮次');
      clearLiveStatusUncertain(operationSessionId);
      state.session=data.state;
      if(consentConfirmed)fullRecordingStarted=await waitForCaptureStartup(startSessionRecorder());
    });
    renderLive();
    toast(fullRecordingStarted?'实时面试与全场录音已开始':'实时面试已开始；全场录音未启动，请检查麦克风');
  }catch(error){
    if(error.liveState&&operationSessionId===state.sessionId)state.session=error.liveState;
    if(error.liveState&&!liveGenerationMatches(error.liveState.live_interview,operationGeneration)&&!liveGenerationMatches(error.liveState.live_interview,expectedActiveGeneration))error.remoteOwnershipChanged=true;
    if(error.remoteOwnershipChanged){
      clearLiveStatusUncertain(operationSessionId);
      toast('远端已进入新的监听轮次；本窗口未启动或中断远端会话。',true);
      renderLive();
      return;
    }
    if(error.liveStatusUncertain){
      try{
        const paused=await enforceFailClosedPause(operationSessionId,expectedActiveGeneration,operationGeneration);
        if(operationSessionId===state.sessionId)state.session=paused.state;
      }catch(pauseError){
        if(pauseError.remoteOwnershipChanged){
          if(pauseError.liveState&&operationSessionId===state.sessionId)state.session=pauseError.liveState;
          clearLiveStatusUncertain(operationSessionId);
          toast('远端已进入新的监听轮次；本窗口未启动或中断远端会话。',true);
          renderLive();
          return;
        }
      }
    }
    const safetyNote=error.liveStatusUncertain?'；为避免无提示监听，本机麦克风保持停止':'';
    toast(`${error.message}${safetyNote}`,true);
    renderLive();
  }
}
$('live-start').onclick=startLiveInterview;
$('live-pause').onclick=async()=>{
  const operationSessionId=state.sessionId;
  const pausing=state.session?.live_interview?.status!=='paused';
  const operationGeneration=snapshotLiveGeneration();
  const resumeOperationId=pausing?'':newRecordingPartId();
  const expectedActiveGeneration=transitionedLiveGeneration(operationGeneration,'active',resumeOperationId);
  let fullRecordingReady=true;
  try{
    await runMediaTransition(async()=>{
      if(operationSessionId!==state.sessionId)throw new Error('会话已变更');
      if(pausing){
        freezeLocalCapture({includeMock:true,sessionId:operationSessionId});
        const localDrain=flushActiveMedia({includeMock:true,includeSessionRecording:true});
        let drainError=null;
        try{await localDrain;}catch(error){drainError=error;}
        await preflightLiveGeneration(operationSessionId,operationGeneration,'暂停');
        let data;
        try{data=await requestLiveStatus(operationSessionId,'paused',operationGeneration.statusRevision);}
        catch(error){
          if(error.liveState&&operationSessionId===state.sessionId)state.session=error.liveState;
          if(error.liveState&&!liveGenerationMatches(error.liveState.live_interview,operationGeneration))throw liveOwnershipChangedError({state:error.liveState},'暂停新的监听轮次');
          try{data=await enforceFailClosedPause(operationSessionId,operationGeneration);}
          catch(pauseError){
            if(pauseError.remoteOwnershipChanged)throw pauseError;
            markLiveStatusUncertain(operationSessionId,operationGeneration);
            throw drainError||error;
          }
        }
        if(operationSessionId!==state.sessionId)throw new Error('会话已变更');
        if(!liveTransitionMatches(data.state?.live_interview,operationGeneration,'paused'))throw liveOwnershipChangedError(data,'暂停新的监听轮次');
        clearLiveStatusUncertain(operationSessionId);
        state.session=data.state;
        if(drainError)throw drainError;
        return;
      }

      if(mockVoiceCaptureBusy())throw new Error('模拟回答仍在录制或保存，不能恢复实时面试');
      if(liveAudioChunks.length||liveRecordingUploading||liveContinuousQueue.length||liveContinuousUploading){
        await flushActiveMedia({includeMock:false,includeSessionRecording:false});
      }
      if(sessionPartQueue.length||sessionAudioUploading)await uploadSessionAudio();
      await preflightLiveGeneration(operationSessionId,operationGeneration,'恢复');
      let data;
      try{data=await requestLiveStatus(operationSessionId,'active',operationGeneration.statusRevision,resumeOperationId,operationGeneration);}
      catch(error){
        if(error.liveState&&operationSessionId===state.sessionId)state.session=error.liveState;
        if(error.liveState&&!liveGenerationMatches(error.liveState.live_interview,operationGeneration)&&!liveGenerationMatches(error.liveState.live_interview,expectedActiveGeneration))throw liveOwnershipChangedError({state:error.liveState},'恢复新的监听轮次');
        if(error.liveStatusUncertain){try{
          const paused=await enforceFailClosedPause(operationSessionId,expectedActiveGeneration,operationGeneration);
          if(operationSessionId===state.sessionId)state.session=paused.state;
        }catch(pauseError){if(pauseError.remoteOwnershipChanged)throw pauseError;}}
        throw error;
      }
      if(operationSessionId!==state.sessionId)throw new Error('会话已变更');
      if(!liveTransitionMatches(data.state?.live_interview,operationGeneration,'active',resumeOperationId))throw liveOwnershipChangedError(data,'恢复新的监听轮次');
      clearLiveStatusUncertain(operationSessionId);
      state.session=data.state;
      if(!sessionRecorder&&!sessionMediaStream&&!sessionPartQueue.length&&state.session?.live_interview?.consent_confirmed){
        fullRecordingReady=await waitForCaptureStartup(startSessionRecorder());
      }
      if(liveContinuousResumeSessionId===operationSessionId){
        await waitForCaptureStartup(startLiveContinuousVad({fromTransition:true}));
      }
    });
    renderLive();
    toast(pausing?'实时会话与录音已暂停':fullRecordingReady?'实时会话与录音已恢复':'实时会话已恢复；全场录音未恢复');
  }catch(error){
    if(error.remoteOwnershipChanged){
      if(error.liveState&&operationSessionId===state.sessionId)state.session=error.liveState;
      clearLiveStatusUncertain(operationSessionId);
      toast('远端已进入新的监听轮次；本机旧录音已处理，未中断远端会话。',true);
      renderLive();
      return;
    }
    const paused=state.session?.live_interview?.status==='paused';
    toast(pausing&&paused?`会话已暂停，但待处理音频尚未保存：${error.message}`:`${pausing?'暂停':'恢复'}失败：${error.message}`,true);
    renderLive();
  }
};
$('live-finish').onclick=async()=>{
  const operationSessionId=state.sessionId;
  const operationGeneration=snapshotLiveGeneration();
  try{
    await runMediaTransition(async()=>{
      freezeLocalCapture({includeMock:true,sessionId:operationSessionId});
      liveContinuousResumeSessionId='';
      await flushActiveMedia({includeMock:true,includeSessionRecording:true});
      if(operationSessionId!==state.sessionId)throw new Error('会话已变更');
      await preflightLiveGeneration(operationSessionId,operationGeneration,'结束');
      const data=await requestLiveStatus(operationSessionId,'completed',operationGeneration.statusRevision);
      if(operationSessionId!==state.sessionId)throw new Error('会话已变更');
      if(!liveTransitionMatches(data.state?.live_interview,operationGeneration,'completed'))throw liveOwnershipChangedError(data,'结束新的监听轮次');
      clearLiveStatusUncertain(operationSessionId);
      state.session=data.state;
    });
    renderLive();
    toast('实时面试已结束，请审阅转写');
  }catch(error){
    if(error.liveState&&operationSessionId===state.sessionId)state.session=error.liveState;
    if(error.liveState&&!liveGenerationMatches(error.liveState.live_interview,operationGeneration))error.remoteOwnershipChanged=true;
    if(error.remoteOwnershipChanged){
      clearLiveStatusUncertain(operationSessionId);
      toast('远端已进入新的监听轮次；本机旧录音已处理，未结束远端会话。',true);
      renderLive();
      return;
    }
    if(operationSessionId===state.sessionId){
      try{
        const paused=await enforceFailClosedPause(operationSessionId,operationGeneration);
        if(operationSessionId===state.sessionId)state.session=paused.state;
      }catch(pauseError){
        if(pauseError.remoteOwnershipChanged){
          if(pauseError.liveState&&operationSessionId===state.sessionId)state.session=pauseError.liveState;
          clearLiveStatusUncertain(operationSessionId);
          toast('远端已进入新的监听轮次；本机旧录音已处理，未结束远端会话。',true);
          renderLive();
          return;
        }
        markLiveStatusUncertain(operationSessionId,operationGeneration);
      }
    }
    toast(`结束未确认：${error.message}。实时会话已安全暂停或正在确认，本机麦克风已停止，原始录音仍保留。`,true);
    renderLive();
  }
};
$('live-text-form').onsubmit=async e=>{e.preventDefault();const text=$('live-text').value.trim();if(!text)return;const form=e.currentTarget;busy(form,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/segments`,{method:'POST',body:JSON.stringify({text,speaker:$('live-speaker').value})});state.session=data.state;$('live-text').value='';renderLive();toast('发言已加入实时对话');}catch(error){toast(error.message,true)}finally{busy(form,false)}};

function liveAudioType(){return ['audio/webm;codecs=opus','audio/webm','audio/mp4'].find(type=>MediaRecorder.isTypeSupported(type))||'';}
function liveAudioFilename(type,prefix='speech'){return type.includes('wav')?`${prefix}.wav`:type.includes('mp4')?`${prefix}.m4a`:`${prefix}.webm`;}
function cancelLiveRecordingRollover({clearLimit=true}={}){
  if(liveRecordingRolloverTimer)clearTimeout(liveRecordingRolloverTimer);
  if(liveRecordingWarningTimer)clearTimeout(liveRecordingWarningTimer);
  liveRecordingRolloverTimer=null;
  liveRecordingWarningTimer=null;
  if(clearLimit)liveRecordingLimitReachedId='';
}
function newRecordingPartId(){
  if(crypto.randomUUID)return crypto.randomUUID();
  const bytes=crypto.getRandomValues(new Uint8Array(16));
  bytes[6]=(bytes[6]&0x0f)|0x40;
  bytes[8]=(bytes[8]&0x3f)|0x80;
  const hex=[...bytes].map(value=>value.toString(16).padStart(2,'0')).join('');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}
async function encodeBlobAsWav(blob){
  // Convert MediaRecorder webm/opus to a 16 kHz mono 16-bit PCM WAV so the
  // LAN ASR (Qwen3-ASR on 8007) can decode it — it rejects webm. Returns a
  // Blob; falls back to the original blob if decoding is unavailable.
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx || !blob.size) return blob;
  let decoder=null;
  try {
    const arrayBuffer = await blob.arrayBuffer();
    decoder=new AudioCtx();
    const audioBuffer = await decoder.decodeAudioData(arrayBuffer);
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
  } finally {
    if(decoder&&decoder.state!=='closed')await decoder.close().catch(()=>{});
  }
}
function showLiveAsrPreview(text=''){
  const box=$('live-asr-preview');const copy=$('live-asr-preview-text');if(!box||!copy)return;
  copy.textContent=text;box.classList.toggle('hidden',!text);
}
function resetAsrPreview(){asrPreviewSequence+=1;asrPreviewLastAt=0;asrPreviewFailureNotified=false;asrPreviewAbort?.abort();asrPreviewAbort=null;asrPreviewBusy=false;showLiveAsrPreview('');}
function boundedAsrPreviewChunks(chunks,timesliceMs){
  const maxDataChunks=Math.max(2,Math.ceil(ASR_PREVIEW_WINDOW_MS/Math.max(1,timesliceMs)));
  if(chunks.length<=maxDataChunks+1)return{chunks:[...chunks],truncated:false};
  return{chunks:[chunks[0],...chunks.slice(-maxDataChunks)],truncated:true};
}
async function previewRecordedChunks(chunks,type,onText,recordingSessionId=state.sessionId,timesliceMs=1000){
  const now=performance.now();const previewSessionId=recordingSessionId;
  if(asrPreviewBusy||chunks.length<2||now-asrPreviewLastAt<ASR_PREVIEW_INTERVAL_MS||!previewSessionId)return;
  const preview=boundedAsrPreviewChunks(chunks,timesliceMs);
  const isMockPreview=onText!==showLiveAsrPreview;
  if(preview.truncated&&isMockPreview){
    if(!asrPreviewFailureNotified)mockVoiceNote('回答较长，实时草稿已停止更新；录音仍在继续，停止后会生成完整转写。');
    asrPreviewFailureNotified=true;
    return;
  }
  asrPreviewBusy=true;asrPreviewLastAt=now;const requestSequence=++asrPreviewSequence;const controller=new AbortController();asrPreviewAbort=controller;
  const snapshot=new Blob(preview.chunks,{type:type||'audio/webm'});
  try{
    const wav=await encodeBlobAsWav(snapshot);const form=new FormData();
    form.append('file',wav,wav.type==='audio/wav'?'preview.wav':liveAudioFilename(wav.type,'preview'));form.append('language','zh');
    const data=await boundedApi(`/api/live-interviews/${previewSessionId}/audio/preview`,{method:'POST',body:form,signal:controller.signal},15000,'实时转写预览超时');
    if(requestSequence===asrPreviewSequence&&previewSessionId===state.sessionId&&data.text?.trim()){asrPreviewFailureNotified=false;onText(preview.truncated?`最近 90 秒：${data.text.trim()}`:data.text.trim());}
  }catch(error){
    if(requestSequence===asrPreviewSequence&&previewSessionId===state.sessionId&&!asrPreviewFailureNotified){
      asrPreviewFailureNotified=true;
      if(isMockPreview)mockVoiceNote('实时转写暂不可用；录音仍在继续，停止后会重试完整转写。');
      else $('live-recording-note').textContent='实时转写暂不可用；录音仍在继续，最终分段会正常保存并转写。';
    }
  }finally{if(asrPreviewAbort===controller){asrPreviewAbort=null;asrPreviewBusy=false;}}
}
async function uploadLiveAudioBlob(sessionId,blob,type,speaker,prefix='speech',mode='single',recordingId=''){
  if(!recordingId)throw new Error('语音片段未登记，已拒绝上传');
  const wavBlob = await encodeBlobAsWav(blob);
  const isWav = wavBlob.type === 'audio/wav';
  const form=new FormData();
  form.append('file', wavBlob, isWav ? `${prefix}.wav` : liveAudioFilename(type,prefix));
  form.append('speaker',speaker);
  form.append('language','zh');
  if(mode==='dialogue')form.append('mode','dialogue');
  form.append('recording_id',recordingId);
  const data=await mediaApi(`/api/live-interviews/${sessionId}/audio`,{method:'POST',body:form});if(sessionId===state.sessionId)state.session=data.state;return data;
}

function currentAudioSettingsSnapshot(){
  const revision=state.settings?.audio_settings_revision;
  const etag=state.settings?.audio_settings_etag;
  const captureEpoch=state.session?.live_interview?.capture_epoch;
  return Number.isInteger(revision)&&etag&&Number.isInteger(captureEpoch)?{revision,etag,captureEpoch}:null;
}

async function registerLiveAudioCapture(sessionId,recordingId,parentRecordingId='',captureRole='utterance',settingsSnapshot=null){
  const snapshot=settingsSnapshot||currentAudioSettingsSnapshot();
  const expectedRevision=snapshot?.revision;
  const expectedEtag=snapshot?.etag;
  const expectedCaptureEpoch=snapshot?.captureEpoch;
  if(!Number.isInteger(expectedRevision)||!expectedEtag||!Number.isInteger(expectedCaptureEpoch))throw new Error('音频配置或监听状态尚未加载，请刷新后重试');
  try{
    await boundedApi(`/api/live-interviews/${sessionId}/audio/captures/${recordingId}`,{method:'POST',body:JSON.stringify({expected_capture_epoch:expectedCaptureEpoch,expected_settings_revision:expectedRevision,expected_settings_etag:expectedEtag,parent_recording_id:parentRecordingId||null,capture_role:captureRole})},12000,'录音登记超时，请重试');
  }catch(error){
    if(error.status===409&&(/Audio settings changed|earlier settings generation|capture epoch changed|旧的音频配置/.test(String(error.message)))){
      // Stop/drain decisions must not wait on a settings GET that may itself be
      // unavailable. Hydration continues once in the background while capture
      // owners receive the stale-generation signal immediately.
      if(!staleAudioSettingsRefresh){
        staleAudioSettingsRefresh=requireRuntimeSettings().catch(()=>{}).finally(()=>{staleAudioSettingsRefresh=null;});
      }
      const staleError=new Error('音频配置已在另一窗口更改；这段旧配置录音不会自动改用新模型，请重新录制或放弃');
      staleError.audioSettingsChanged=true;
      throw staleError;
    }
    throw error;
  }
}

async function cancelLiveAudioCapture(sessionId,recordingId){
  if(!sessionId||!recordingId)return;
  await api(`/api/live-interviews/${sessionId}/audio/captures/${recordingId}`,{method:'DELETE'});
}

function settledLiveAudioCaptureRegistration(sessionId,recordingId,parentRecordingId='',settingsSnapshot=null){
  return registerLiveAudioCapture(sessionId,recordingId,parentRecordingId,'utterance',settingsSnapshot).then(()=>null,error=>error);
}

function cancelSettledLiveAudioCapture(sessionId,recordingId,registration){
  if(!sessionId||!recordingId)return;
  void Promise.resolve(registration).then(()=>cancelLiveAudioCapture(sessionId,recordingId)).catch(()=>cancelLiveAudioCapture(sessionId,recordingId)).catch(()=>{});
}

async function renewPendingLiveCaptureLeases(){
  const liveStatus=state.session?.live_interview?.status;
  if(!state.token||!runtimeSettingsReady||!state.sessionId||!['active','paused','completed'].includes(liveStatus))return;
  const sessionId=state.sessionId;
  const generation=liveCaptureGeneration;
  const captures=new Map();
  const add=(id,role='utterance',parent='',settingsSnapshot=null)=>{if(id&&!captures.has(id))captures.set(id,{id,role,parent,settingsSnapshot});};
  if(pendingLiveCaptureStart?.sessionId===sessionId)add(pendingLiveCaptureStart.recordingId,pendingLiveCaptureStart.captureRole,pendingLiveCaptureStart.parentRecordingId,pendingLiveCaptureStart.settingsSnapshot);
  if(liveRecordingSessionId===sessionId&&liveRecordingId&&(liveRecorder||liveAudioChunks.length||liveRecordingUploading))add(liveRecordingId,'utterance','',liveRecordingSettingsSnapshot);
  if(liveStatus==='active'&&liveContinuousSessionId===sessionId&&liveContinuousGuardId)add(liveContinuousGuardId,'guard','',liveContinuousSettingsSnapshot);
  if(liveVadRecordingId&&liveContinuousSessionId===sessionId)add(liveVadRecordingId,'utterance',liveContinuousGuardId,liveContinuousSettingsSnapshot);
  for(const chunk of liveContinuousQueue){if(chunk.sessionId===sessionId){add(chunk.recordingId,'utterance',chunk.parentRecordingId,chunk.settingsSnapshot);if(liveStatus==='active'&&chunk.parentRecordingId)add(chunk.parentRecordingId,'guard','',chunk.settingsSnapshot);}}
  const captureEntries=[...captures.values()];
  const results=await Promise.allSettled(captureEntries.map(item=>registerLiveAudioCapture(sessionId,item.id,item.parent,item.role,item.settingsSnapshot)));
  if(generation!==liveCaptureGeneration||sessionId!==state.sessionId)return;
  const stillLocallyOwned=item=>Boolean(
    pendingLiveCaptureStart?.recordingId===item.id||
    (liveRecordingSessionId===sessionId&&liveRecordingId===item.id&&(liveRecorder||liveAudioChunks.length||liveRecordingUploading))||
    (liveContinuousSessionId===sessionId&&liveContinuousGuardId===item.id&&(liveContinuousMode||liveRecorder||liveMediaStream))||
    (liveContinuousSessionId===sessionId&&liveVadRecordingId===item.id)||
    liveContinuousQueue.some(chunk=>chunk.sessionId===sessionId&&(chunk.recordingId===item.id||chunk.parentRecordingId===item.id))
  );
  const authorityLost=results.some((result,index)=>result.status==='rejected'&&(
    result.reason?.audioSettingsChanged||
    (stillLocallyOwned(captureEntries[index])&&result.reason?.status===409&&/must be active|already cancelled|capture epoch changed|role does not match/i.test(String(result.reason?.message)))
  ));
  if(authorityLost){
    freezeLocalCapture({sessionId});
    toast('音频配置、监听轮次或远端状态已变化，本机麦克风已停止；未保存录音仍保留，请刷新后处理。',true);
  }
}

async function startLiveRecording(dialogue=false){
  if(mediaTransitionInProgress||mediaFlushInProgress||authenticationSuspended||!runtimeSettingsReady||liveStatusUncertainSessionId===state.sessionId){toast('录音正在保存、配置未加载、状态确认中或需要重新登录，请稍后重试',true);return;}
  if(!capturePageIsVisible()){toast('页面在后台，已拒绝启动麦克风；请回到当前页面后重试',true);return;}
  if(mockVoiceCaptureBusy()){toast('请先停止并保存当前模拟回答录音，再开始实时收音',true);return;}
  cancelMockQuestionSpeech();clearMockQuestionAudio();pauseMockAnswerPlayback();
  if(!ensureMicrophoneAvailable())return;
  if(liveCaptureStartPending||liveRecorder)return;
  if(liveContinuousQueue.length||liveContinuousUploading){toast('请先保存连续监听队列中的音频',true);return;}
  if(liveAudioChunks.length||liveRecordingUploading){
    try{await scheduleLiveRecordingUpload();toast('上一段录音已保存；请再次点击开始新的录音');}catch{}
    return;
  }
  const recordingSessionId=state.sessionId;
  const generation=++liveCaptureGeneration;
  const settingsSnapshot=currentAudioSettingsSnapshot();
  let recordingId='';
  let stream=null;
  liveCaptureStartPending=true;
  renderLive();
  try{
    recordingId=newRecordingPartId();
    pendingLiveCaptureStart={sessionId:recordingSessionId,recordingId,captureRole:'utterance',parentRecordingId:'',settingsSnapshot};
    await registerLiveAudioCapture(recordingSessionId,recordingId,'','utterance',settingsSnapshot);
    if(generation!==liveCaptureGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||!runtimeSettingsReady||authenticationSuspended||liveStatusUncertainSessionId===recordingSessionId||state.session?.live_interview?.status!=='active'||desktopShutdownPreparing){
      if(pendingLiveCaptureStart?.recordingId===recordingId)pendingLiveCaptureStart=null;
      void cancelLiveAudioCapture(recordingSessionId,recordingId).catch(()=>{});
      return;
    }
    stream=await navigator.mediaDevices.getUserMedia(microphoneConstraints());
    if(generation!==liveCaptureGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||!runtimeSettingsReady||authenticationSuspended||liveStatusUncertainSessionId===recordingSessionId||state.session?.live_interview?.status!=='active'||desktopShutdownPreparing){
      stream.getTracks().forEach(track=>track.stop());
      if(pendingLiveCaptureStart?.recordingId===recordingId)pendingLiveCaptureStart=null;
      void cancelLiveAudioCapture(recordingSessionId,recordingId).catch(()=>{});
      return;
    }
    liveMediaStream=stream;
    if(!await confirmRemoteLiveCaptureAuthority(recordingSessionId)||generation!==liveCaptureGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||state.session?.live_interview?.status!=='active'||state.session?.live_interview?.capture_epoch!==settingsSnapshot?.captureEpoch||!runtimeSettingsReady||authenticationSuspended||desktopShutdownPreparing){
      stream.getTracks().forEach(track=>track.stop());
      void cancelLiveAudioCapture(recordingSessionId,recordingId).catch(()=>{});
      throw new Error('实时会话已在另一窗口停止');
    }
    if(pendingLiveCaptureStart?.recordingId===recordingId)pendingLiveCaptureStart=null;
    liveRecordingSessionId=recordingSessionId;
    liveRecordingId=recordingId;
    liveRecordingSettingsSnapshot=settingsSnapshot;
    const preferred=liveAudioType();
    const recorder=new MediaRecorder(stream,preferred?{mimeType:preferred}:undefined);
    liveRecorder=recorder;
    liveRecordingMimeType=recorder.mimeType||preferred||'audio/webm';
    liveRecordingSpeaker=dialogue?'unknown':$('live-speaker').value;
    liveRecordingMode=dialogue?'dialogue':'single';
    liveAudioChunks=[];
    resetAsrPreview();
    liveRecorderDone=new Promise(resolve=>recorder.addEventListener('stop',resolve,{once:true}));
    recorder.ondataavailable=event=>{if(!event.data.size)return;liveAudioChunks.push(event.data);previewRecordedChunks(liveAudioChunks,liveRecordingMimeType,showLiveAsrPreview,recordingSessionId);};
    recorder.onstop=()=>{
      cancelLiveRecordingRollover({clearLimit:false});
      void scheduleLiveRecordingUpload().catch(()=>{});
    };
    recorder.start(1000);
    cancelLiveRecordingRollover();
    liveRecordingWarningTimer=setTimeout(()=>{
      if(liveRecorder===recorder&&recorder.state==='recording')toast('本段录音将在 1 分钟后达到安全上限；长时间访谈请改用连续监听模式',true);
    },LIVE_RECORDING_WARNING_MS);
    liveRecordingRolloverTimer=setTimeout(()=>{
      if(liveRecorder!==recorder||recorder.state!=='recording')return;
      liveRecordingLimitReachedId=recordingId;
      liveRecordingRolloverTimer=null;
      try{recorder.requestData();}catch{}
      recorder.stop();
      $('live-stop-record').disabled=true;
      $('live-recording-note').textContent='本段已达到 10 分钟安全上限，正在保存；需要继续时请再次点击录音，或改用连续监听。';
    },LIVE_RECORDING_ROLLOVER_MS);
    liveDialogueMode=dialogue;
    $('live-stop-record').disabled=false;
    $('live-recording-note').textContent=dialogue?'对话模式：实时草稿仅展示文字，最终结果再自动区分说话人。':`正在录制${$('live-speaker').value==='candidate'?'候选人':$('live-speaker').value==='interviewer'?'面试官':'待确认'}发言并实时转写…`;
    if(dialogue)toast('对话模式开始，请录完整段对话');
  }catch(error){
    // DELETE is idempotent. It also covers the case where the server committed
    // registration but the API freshness check rejected the late response.
    if(recordingId)void cancelLiveAudioCapture(recordingSessionId,recordingId).catch(()=>{});
    if(pendingLiveCaptureStart?.recordingId===recordingId)pendingLiveCaptureStart=null;
    stream?.getTracks().forEach(track=>track.stop());
    if(liveMediaStream===stream)liveMediaStream=null;
    if(liveRecordingSessionId===recordingSessionId&&liveRecordingId===recordingId){liveRecordingSessionId='';liveRecordingId='';}
    if(liveRecorder?.stream===stream)liveRecorder=null;
    if(generation===liveCaptureGeneration){
      cancelLiveRecordingRollover();
      liveRecorderDone=Promise.resolve();liveAudioChunks=[];liveRecordingSpeaker='unknown';liveRecordingMode='single';liveRecordingSettingsSnapshot=null;liveDialogueMode=false;
      toast(`无法使用麦克风：${error.message}`,true);
    }
  }
  finally{if(generation===liveCaptureGeneration)liveCaptureStartPending=false;renderLive();}
}
$('live-record').onclick=()=>startLiveRecording(false);
$('live-dialogue').onclick=()=>startLiveRecording(true);
$('live-stop-record').onclick=()=>{if(liveRecorder?.state==='recording'){cancelLiveRecordingRollover();resetAsrPreview();liveRecorder.stop();$('live-stop-record').disabled=true;$('live-recording-note').textContent='录制结束，正在确认最终转写…'}};

let liveDialogueMode = false;
function scheduleLiveRecordingUpload(){
  if(liveRecordingUploading)return liveRecordingUploadPromise;
  liveRecordingUploading=true;
  liveRecordingUploadPromise=persistLiveRecording().finally(()=>{liveRecordingUploading=false;});
  return liveRecordingUploadPromise;
}
async function persistLiveRecording(){
  const recorder=liveRecorder;
  const stream=liveMediaStream;
  const recordingSessionId=liveRecordingSessionId;
  const recordingId=liveRecordingId;
  const type=recorder?.mimeType||liveRecordingMimeType||'audio/webm';
  const wasDialogue=liveDialogueMode;
  const speaker=liveRecordingSpeaker;
  const mode=liveRecordingMode;
  const reachedDurationLimit=liveRecordingLimitReachedId===recordingId;
  if(recorder){liveRecorder=null;liveMediaStream=null;stream?.getTracks().forEach(track=>track.stop());}
  const blob=new Blob([...liveAudioChunks],{type});
  if(!blob.size){if(recordingSessionId&&recordingId)void cancelLiveAudioCapture(recordingSessionId,recordingId).catch(()=>{});if(liveRecordingLimitReachedId===recordingId)liveRecordingLimitReachedId='';liveAudioChunks=[];liveRecordingSessionId='';liveRecordingId='';liveRecordingSpeaker='unknown';liveRecordingMode='single';liveRecordingSettingsSnapshot=null;liveDialogueMode=false;renderLive();return;}
  if(!recordingSessionId)throw new Error('录音缺少所属会话，已停止保存以避免串写');
  if(!recordingId)throw new Error('录音缺少幂等标识，已停止保存以避免重复转写');
  try{
    await uploadLiveAudioBlob(recordingSessionId,blob,type,speaker,'speech',mode,recordingId);
    liveAudioChunks=[];
    liveRecordingSessionId='';
    liveRecordingId='';
    liveRecordingSpeaker='unknown';
    liveRecordingMode='single';
    liveRecordingSettingsSnapshot=null;
    liveDialogueMode=false;
    if(reachedDurationLimit)liveRecordingLimitReachedId='';
    resetAsrPreview();
    if(recordingSessionId===state.sessionId){$('live-recording-note').textContent=reachedDurationLimit?'本段录音已安全保存；请点击继续录音，或开启连续监听避免长访谈遗漏。':wasDialogue?'最终对话已转写并自动区分说话人；候选人回答可确认。':'最终转写已确认；候选人回答结束后可生成下一问题。';toast(reachedDurationLimit?'长录音已分段保存，继续前请再次点击录音':wasDialogue?'对话已自动分说话人':'音频已转为文字');}
  }catch(error){if(recordingSessionId===state.sessionId){$('live-recording-note').textContent='最终转写失败；原始音频仍保留，保存成功前不能切换会话或退出。';toast(error.message,true);}throw error;}
  finally{renderLive();}
}

$('live-continuous').onclick=startLiveContinuousVad;
$('live-stop-continuous').onclick=()=>{if(liveContinuousResumeSessionId===state.sessionId)liveContinuousResumeSessionId='';liveContinuousMode=false;if(liveRecorder?.state==='recording')liveRecorder.stop();$('live-recording-note').textContent='连续监听停止中；正在处理已录到的音频。';renderLive();};
$('live-retry-pending-audio').onclick=async()=>{
  try{
    const restartSessionRecorder=sessionRecordingError&&sessionRecordingErrorSessionId===state.sessionId&&state.session?.live_interview?.status==='active';
    if(liveAudioChunks.length||liveRecordingUploading)await scheduleLiveRecordingUpload();
    await processLiveContinuousQueue();
    if(sessionPartQueue.length||sessionAudioUploading)await uploadSessionAudio();
    if(restartSessionRecorder&&!sessionPartQueue.length&&!sessionAudioUploading){
      const started=await startSessionRecorder();
      if(!started)throw new Error(sessionRecordingError||'全场录音仍未启动');
      toast('待处理音频已保存，全场录音已重新启动');
      return;
    }
    if(liveAudioChunks.length||liveContinuousQueue.length||sessionPartQueue.length)toast('仍未保存成功，请检查本地模型服务后重试',true);
    else toast('待处理音频已全部保存');
  }catch(error){toast(`待处理音频处理失败：${error.message}`,true);}
};
$('live-discard-pending-audio').onclick=async()=>{
  if(!state.sessionId||!window.confirm('放弃本窗口尚未保存的发言音频？这些音频将无法恢复，已保存的全场录音和其他窗口的录音不受影响。'))return;
  const sessionId=state.sessionId;
  const ids=localPendingLiveCaptureIds(sessionId);
  try{
    await Promise.all([...ids].map(id=>cancelLiveAudioCapture(sessionId,id)));
    cancelLiveRecordingRollover();
    liveAudioChunks=[];liveRecordingSessionId='';liveRecordingId='';liveRecordingSpeaker='unknown';liveRecordingMode='single';liveRecordingSettingsSnapshot=null;liveDialogueMode=false;
    liveContinuousQueue=liveContinuousQueue.filter(chunk=>chunk.sessionId!==sessionId);
    if(liveContinuousResumeSessionId===sessionId)liveContinuousResumeSessionId='';
    resetAsrPreview();
    const data=await api(`/api/live-interviews/${sessionId}`);
    if(sessionId===state.sessionId)state.session=data.state;
    renderLive();toast('未保存的发言音频已放弃');
  }catch(error){toast(`放弃待处理音频失败：${error.message}`,true);}
};
function enqueueLiveContinuousChunk(blob,recordingId=newRecordingPartId(),registration=null,parentRecordingId=liveContinuousGuardId,settingsSnapshot=liveContinuousSettingsSnapshot){if(!blob.size)return;const chunkSessionId=liveContinuousSessionId;liveContinuousQueue.push({blob,type:blob.type||'audio/webm',index:++liveContinuousChunkIndex,sessionId:chunkSessionId,mode:liveContinuousUploadMode(),recordingId,parentRecordingId,settingsSnapshot,registration:registration||settledLiveAudioCaptureRegistration(chunkSessionId,recordingId,parentRecordingId,settingsSnapshot)});if(liveContinuousQueue.length>=LIVE_CONTINUOUS_QUEUE_HIGH_WATER){freezeLocalCapture({sessionId:chunkSessionId});toast('转写队列积压，已自动停止继续收音；原始音频仍保留，可在服务恢复后重试。',true);}void processLiveContinuousQueue();renderLive();}
async function processLiveContinuousQueue(){
  if(liveContinuousUploading)return;
  liveContinuousUploading=true;
  try{
    while(liveContinuousQueue.length){
      const chunk=liveContinuousQueue[0];
      if(!liveContinuousMode)$('live-recording-note').textContent=`正在转写连续监听第 ${chunk.index} 段；剩余 ${liveContinuousQueue.length-1} 段。`;else updateLiveVadNote();
      try{
        let registrationError=await chunk.registration;
        if(registrationError&&!registrationError.audioSettingsChanged){chunk.registration=settledLiveAudioCaptureRegistration(chunk.sessionId,chunk.recordingId,chunk.parentRecordingId,chunk.settingsSnapshot);registrationError=await chunk.registration;}
        // Always try the idempotent upload. If the server committed the first
        // attempt but its response was lost, the durable receipt succeeds even
        // when re-registration is no longer allowed after completion.
        try{
          await uploadLiveAudioBlob(chunk.sessionId,chunk.blob,chunk.type,'unknown',`continuous-${chunk.index}`,chunk.mode,chunk.recordingId);
        }catch(uploadError){
          if(registrationError?.audioSettingsChanged)throw registrationError;
          throw uploadError;
        }
        liveContinuousQueue.shift();
        if(chunk.parentRecordingId&&liveContinuousGuardId!==chunk.parentRecordingId&&!liveContinuousQueue.some(item=>item.parentRecordingId===chunk.parentRecordingId))void cancelLiveAudioCapture(chunk.sessionId,chunk.parentRecordingId).catch(()=>{});
        resetAsrPreview();
        renderLive();
      }catch(error){
        // Browser memory is not an offline recording journal. Stop producing
        // new blobs on the first persistent failure and preserve this bounded
        // raw queue for an explicit retry.
        freezeLocalCapture({sessionId:chunk.sessionId});
        $('live-recording-note').textContent=`连续监听第 ${chunk.index} 段最终转写失败；原始音频仍保留，稍后会重试。`;
        toast(`连续监听第 ${chunk.index} 段转写失败：${error.message}`,true);
        break;
      }
    }
  }finally{
    liveContinuousUploading=false;
    if(!liveContinuousMode&&!liveContinuousQueue.length)$('live-recording-note').textContent='连续监听队列已处理完成；请审阅待确认片段。';else if(liveContinuousMode)updateLiveVadNote();
    renderLive();
  }
}
async function startLiveContinuousVad({fromTransition=false}={}){
  if((mediaTransitionInProgress&&!fromTransition)||mediaFlushInProgress||authenticationSuspended||!runtimeSettingsReady||liveStatusUncertainSessionId===state.sessionId){toast('录音正在保存、配置未加载、状态确认中或需要重新登录，请稍后重试',true);return;}
  if(!capturePageIsVisible()){toast('页面在后台，已拒绝启动连续监听；请回到当前页面后重试',true);return;}
  if(mockVoiceCaptureBusy()){toast('请先停止并保存当前模拟回答录音，再开始连续监听',true);return;}
  cancelMockQuestionSpeech();clearMockQuestionAudio();pauseMockAnswerPlayback();
  if(!ensureMicrophoneAvailable())return;
  if(liveCaptureStartPending||liveRecorder||liveRecordingUploading||liveAudioChunks.length||liveContinuousQueue.length)return;
  const recordingSessionId=state.sessionId;
  const generation=++liveCaptureGeneration;
  const guardId=newRecordingPartId();
  const settingsSnapshot=currentAudioSettingsSnapshot();
  let stream=null;
  const ownsStart=()=>liveContinuousGuardId===guardId&&liveContinuousSessionId===recordingSessionId;
  const cleanupThisStart=({teardown=false}={})=>{
    stream?.getTracks().forEach(track=>track.stop());
    if(pendingLiveCaptureStart?.recordingId===guardId)pendingLiveCaptureStart=null;
    void cancelLiveAudioCapture(recordingSessionId,guardId).catch(()=>{});
    if(!ownsStart())return;
    if(teardown){teardownLiveContinuous();return;}
    if(liveMediaStream===stream)liveMediaStream=null;
    liveContinuousGuardId='';liveContinuousSessionId='';liveContinuousSettingsSnapshot=null;
  };
  liveCaptureStartPending=true;
  renderLive();
  try{
    pendingLiveCaptureStart={sessionId:recordingSessionId,recordingId:guardId,captureRole:'guard',parentRecordingId:'',settingsSnapshot};
    await registerLiveAudioCapture(recordingSessionId,guardId,'','guard',settingsSnapshot);
    if(generation!==liveCaptureGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||!runtimeSettingsReady||authenticationSuspended||liveStatusUncertainSessionId===recordingSessionId||state.session?.live_interview?.status!=='active'||desktopShutdownPreparing){cleanupThisStart();return;}
    liveContinuousGuardId=guardId;
    liveContinuousSessionId=recordingSessionId;
    liveContinuousSettingsSnapshot=settingsSnapshot;
    stream=await navigator.mediaDevices.getUserMedia(microphoneConstraints());
    if(generation!==liveCaptureGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||!runtimeSettingsReady||authenticationSuspended||liveStatusUncertainSessionId===recordingSessionId||state.session?.live_interview?.status!=='active'||desktopShutdownPreparing){cleanupThisStart();return;}
    liveMediaStream=stream;
    if(!await confirmRemoteLiveCaptureAuthority(recordingSessionId)||generation!==liveCaptureGeneration||recordingSessionId!==state.sessionId||!capturePageIsVisible()||state.session?.live_interview?.status!=='active'||state.session?.live_interview?.capture_epoch!==settingsSnapshot?.captureEpoch||!runtimeSettingsReady||authenticationSuspended||desktopShutdownPreparing){cleanupThisStart();throw new Error('实时会话已在另一窗口停止或重新开始');}
    if(pendingLiveCaptureStart?.recordingId===guardId)pendingLiveCaptureStart=null;
    liveContinuousMode=true;
    if(liveContinuousResumeSessionId===recordingSessionId)liveContinuousResumeSessionId='';
    liveContinuousChunkIndex=0;
    liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;liveVadHeaderChunk=null;liveVadState='idle';liveVadSpeechStart=0;liveVadLastSpeech=0;liveVadVoicedMs=0;liveVadLastAnalysisAt=0;liveVadLastFinalizeAt=0;liveVadFinalizing=false;
    resetAsrPreview();
    liveVadAudioContext=new(window.AudioContext||window.webkitAudioContext)();
    liveVadAnalyser=liveVadAudioContext.createAnalyser();liveVadAnalyser.fftSize=1024;liveVadAnalyser.smoothingTimeConstant=0.3;
    const source=liveVadAudioContext.createMediaStreamSource(liveMediaStream);source.connect(liveVadAnalyser);
    liveVadData=new Float32Array(liveVadAnalyser.fftSize);liveVadTimer=setInterval(analyzeLiveVadLevel,VAD_LEVEL_INTERVAL_MS);
    startContinuousRecorder(liveAudioType());updateLiveVadNote();toast('语音分段监听已开始：静音约 0.6 秒自动结束一段');
  }catch(error){cleanupThisStart({teardown:true});if(generation===liveCaptureGeneration)toast(`无法启动连续监听：${error.message}`,true);}
  finally{if(generation===liveCaptureGeneration)liveCaptureStartPending=false;renderLive();}
}
function startContinuousRecorder(preferred){
  const stream=liveMediaStream;
  if(!stream)return;
  let recorder=null;
  liveVadHeaderChunk=null;
  try{
    recorder=new MediaRecorder(stream,preferred?{mimeType:preferred}:undefined);
    liveRecorder=recorder;
    liveRecorderDone=new Promise(resolve=>recorder.addEventListener('stop',resolve,{once:true}));
    recorder.ondataavailable=event=>{
      if(!event.data.size)return;
      if(!liveVadHeaderChunk){
        liveVadHeaderChunk=event.data;
        if(!['in-speech','finalizing'].includes(liveVadState))return;
      }
      if(!['in-speech','finalizing'].includes(liveVadState))return;
      if(!liveVadUtteranceChunks.length&&liveVadHeaderChunk!==event.data){
        liveVadUtteranceChunks.push(liveVadHeaderChunk);
        liveVadUtteranceBytes+=liveVadHeaderChunk.size;
      }
      liveVadUtteranceChunks.push(event.data);
      liveVadUtteranceBytes+=event.data.size;
      previewRecordedChunks(liveVadUtteranceChunks,recorder.mimeType||preferred,showLiveAsrPreview,liveContinuousSessionId,VAD_TIMESLICE_MS);
      if(liveVadUtteranceBytes>20*1024*1024&&recorder.state==='recording'){
        liveVadState='finalizing';
        finalizeLiveUtterance();
      }
    };
    recorder.onstop=stopLiveContinuousVad;
    recorder.start(VAD_TIMESLICE_MS);
  }catch(error){
    if(recorder){recorder.ondataavailable=null;recorder.onstop=null;}
    if(liveRecorder===recorder)liveRecorder=null;
    liveRecorderDone=Promise.resolve();
    throw error;
  }
}
function analyzeLiveVadLevel(){
  if(!liveVadAnalyser||!liveVadData||!liveContinuousMode)return;
  if(liveVadAudioContext?.state==='suspended'){void liveVadAudioContext.resume().catch(()=>{});return;}
  liveVadAnalyser.getFloatTimeDomainData(liveVadData);
  let sum=0;
  for(let i=0;i<liveVadData.length;i++)sum+=liveVadData[i]*liveVadData[i];
  const rms=Math.sqrt(sum/liveVadData.length);
  const now=performance.now();
  const analysisElapsed=liveVadLastAnalysisAt?Math.min(VAD_LEVEL_INTERVAL_MS*5,Math.max(0,now-liveVadLastAnalysisAt)):VAD_LEVEL_INTERVAL_MS;
  liveVadLastAnalysisAt=now;
  const speaking=rms>=VAD_SPEECH_RMS;
  if(liveVadState==='idle'){
    if(now-liveVadLastFinalizeAt<VAD_GRACE_MS)return;
    if(speaking){
      resetAsrPreview();liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;
      liveVadRecordingId=newRecordingPartId();
      liveVadCaptureRegistration=settledLiveAudioCaptureRegistration(liveContinuousSessionId,liveVadRecordingId,liveContinuousGuardId,liveContinuousSettingsSnapshot);
      liveVadState='in-speech';liveVadSpeechStart=now;liveVadLastSpeech=now;liveVadVoicedMs=analysisElapsed;
    }
  }else if(liveVadState==='in-speech'){
    if(speaking){liveVadLastSpeech=now;liveVadVoicedMs+=analysisElapsed;}
    else if(now-liveVadLastSpeech>=VAD_SILENCE_MS){
      if(liveVadVoicedMs>=VAD_MIN_SPEECH_MS&&liveVadUtteranceChunks.length){liveVadState='finalizing';finalizeLiveUtterance();}
      else{
        const emptySessionId=liveContinuousSessionId;const emptyRecordingId=liveVadRecordingId;const emptyRegistration=liveVadCaptureRegistration;
        liveVadState='idle';liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;liveVadRecordingId='';liveVadCaptureRegistration=null;liveVadSpeechStart=0;liveVadLastSpeech=0;liveVadVoicedMs=0;resetAsrPreview();
        cancelSettledLiveAudioCapture(emptySessionId,emptyRecordingId,emptyRegistration);
      }
    }
  }
  updateLiveVadNote();
}
function finalizeLiveUtterance(){if(liveVadFinalizing)return;liveVadFinalizing=true;liveVadLastFinalizeAt=performance.now();if(liveVadState!=='finalizing')liveVadState='finalizing';if(liveRecorder?.state==='recording')liveRecorder.stop();}
function stopLiveContinuousVad(){
  const hadSpeech=liveVadVoicedMs>=VAD_MIN_SPEECH_MS&&liveVadUtteranceChunks.length>0;
  const type=liveRecorder?.mimeType||'audio/webm';
  if(hadSpeech){
    const blob=new Blob(liveVadUtteranceChunks,{type});
    liveVadUtteranceChunks=[];
    liveVadUtteranceBytes=0;
    const recordingId=liveVadRecordingId||newRecordingPartId();
    const registration=liveVadCaptureRegistration||settledLiveAudioCaptureRegistration(liveContinuousSessionId,recordingId,liveContinuousGuardId,liveContinuousSettingsSnapshot);
    enqueueLiveContinuousChunk(blob,recordingId,registration,liveContinuousGuardId,liveContinuousSettingsSnapshot);
  }else if(liveVadRecordingId&&liveVadCaptureRegistration){
    const emptySessionId=liveContinuousSessionId;
    const emptyRecordingId=liveVadRecordingId;
    // DELETE is idempotent and leaves a short tombstone, so it is also safe
    // when registration committed but its response was lost.
    cancelSettledLiveAudioCapture(emptySessionId,emptyRecordingId,liveVadCaptureRegistration);
  }
  liveVadRecordingId='';
  liveVadCaptureRegistration=null;
  liveVadFinalizing=false;
  liveVadState='idle';
  liveVadSpeechStart=0;
  liveVadLastSpeech=0;
  liveVadVoicedMs=0;
  if(liveContinuousMode&&liveMediaStream){
    try{
      startContinuousRecorder(liveAudioType());
      updateLiveVadNote();
      renderLive();
    }catch(error){
      teardownLiveContinuous();
      $('live-recording-note').textContent='连续监听重启失败；已录音频仍保留在待处理队列。';
      toast(`连续监听重启失败：${error.message}`,true);
      renderLive();
    }
  }else{
    teardownLiveContinuous();
    $('live-recording-note').textContent=liveContinuousQueue.length||liveContinuousUploading?'连续监听已停止；剩余音频正在转写。':'连续监听已停止。';
    renderLive();
  }
}
function teardownLiveContinuous(){const guardSessionId=liveContinuousSessionId;const guardId=liveContinuousGuardId;const guardHasPendingAudio=!!guardId&&liveContinuousQueue.some(chunk=>chunk.sessionId===guardSessionId&&chunk.parentRecordingId===guardId);liveContinuousMode=false;if(liveVadTimer){clearInterval(liveVadTimer);liveVadTimer=null;}if(liveVadAudioContext){liveVadAudioContext.close().catch(()=>{});liveVadAudioContext=null;}liveVadAnalyser=null;liveVadData=null;liveVadState='idle';liveVadUtteranceChunks=[];liveVadUtteranceBytes=0;liveVadHeaderChunk=null;liveVadRecordingId='';liveVadCaptureRegistration=null;liveVadSpeechStart=0;liveVadLastSpeech=0;liveVadVoicedMs=0;liveVadLastAnalysisAt=0;liveVadLastFinalizeAt=0;liveVadFinalizing=false;if(liveMediaStream){liveMediaStream.getTracks().forEach(track=>track.stop());}liveMediaStream=null;liveRecorder=null;liveContinuousSessionId='';liveContinuousGuardId='';liveContinuousSettingsSnapshot=null;if(guardSessionId&&guardId&&!guardHasPendingAudio)void cancelLiveAudioCapture(guardSessionId,guardId).catch(()=>{});}
function liveContinuousUploadMode(){const mode=state.settings?.live_audio?.mode||document.querySelector('#setting-live-audio-mode')?.value||'asr_text';return mode==='audio_direct'?'dialogue':'single';}
function updateLiveVadNote(){const note=$('live-recording-note');if(!note)return;if(liveVadState==='in-speech'){note.textContent=`正在听取 (${Math.max(0,Math.floor((performance.now()-liveVadSpeechStart)/1000))}s)`;note.classList.add('listening');}else{note.textContent='正在聆听…';note.classList.remove('listening');}if(liveContinuousUploading||liveContinuousQueue.length){note.textContent+=` ｜ 队列 ${liveContinuousQueue.length} 段${liveContinuousUploading?'，正在转写 1 段':''}`;}}
function activeLocalCaptureEpoch(sessionId){if(sessionRecorderStartSessionId===sessionId&&sessionRecorderStartPending)return sessionRecorderStartCaptureEpoch;if(sessionRecordingSessionId===sessionId&&(sessionRecorder||sessionMediaStream||sessionRotationInProgress))return sessionRecordingCaptureEpoch;if(pendingLiveCaptureStart?.sessionId===sessionId)return pendingLiveCaptureStart.settingsSnapshot?.captureEpoch??null;if(liveContinuousSessionId===sessionId&&(liveContinuousMode||liveRecorder||liveMediaStream))return liveContinuousSettingsSnapshot?.captureEpoch??null;if(liveRecordingSessionId===sessionId&&(liveCaptureStartPending||liveRecorder||liveMediaStream))return liveRecordingSettingsSnapshot?.captureEpoch??null;return null;}
function hasLocalLiveCapture(sessionId){return Boolean((pendingLiveCaptureStart?.sessionId===sessionId)||(sessionRecorderStartSessionId===sessionId&&(sessionRecorderStartPending||sessionPendingMediaStream))||(sessionRecordingSessionId===sessionId&&(sessionRecorder||sessionMediaStream||sessionRotationInProgress))||(liveRecordingSessionId===sessionId&&(liveCaptureStartPending||liveRecorder||liveMediaStream))||(liveContinuousSessionId===sessionId&&(liveContinuousMode||liveRecorder||liveMediaStream)));}
function hasUnsavedAudio(){return Boolean(mockVoiceCaptureBusy()||mockVoiceChunks.length||sessionRecorderStartPending||sessionPendingMediaStream||sessionRecordingActive||sessionMediaStream||sessionRecorder||sessionRotationInProgress||sessionChunks.length||sessionPartQueue.length||sessionAudioUploading||pendingLiveCaptureStart||liveCaptureStartPending||liveRecorder||liveMediaStream||liveRecordingUploading||liveAudioChunks.length||liveContinuousMode||liveVadFinalizing||liveVadUtteranceChunks.length||liveContinuousQueue.length||liveContinuousUploading);}
window.addEventListener('beforeunload',event=>{if(runtime.mode==='browser'&&hasUnsavedAudio()){event.preventDefault();event.returnValue='关闭页面会丢失尚未保存的原始录音；请先暂停或结束面试。';}});
window.addEventListener('pagehide',()=>{
  if(runtime.mode!=='browser')return;
  const sessionId=state.sessionId;
  const hadLocalCapture=hasLocalLiveCapture(sessionId);
  if(hadLocalCapture&&sessionId)markLivePauseRequired(sessionId);
  freezeLocalCapture({includeMock:true,sessionId});
  if(hadLocalCapture&&sessionId&&state.token&&state.session?.live_interview?.status==='active'){
    // keepalive is intentionally used only for the tiny status mutation. Whole
    // recordings exceed browser keepalive/sendBeacon limits and must be saved
    // through the explicit pause/finish flow guarded by beforeunload.
    const expectedRevision=Number(state.session?.live_interview?.status_revision||0);
    void api.raw(`/api/live-interviews/${sessionId}/status`,{method:'POST',body:JSON.stringify({status:'paused',expected_revision:expectedRevision}),keepalive:true}).catch(()=>{});
  }
});

async function recoverPageBoundary(){
  const sessionId=livePauseRequiredSessionId;
  if(pageBoundaryRecoveryInProgress||!sessionId||!state.token||state.sessionId!==sessionId||!state.session)return;
  if(mediaTransitionInProgress||mediaFlushInProgress){setTimeout(()=>void recoverPageBoundary(),500);return;}
  pageBoundaryRecoveryInProgress=true;
  try{
    const result=await runMediaTransition(()=>prepareCurrentSessionBoundary());
    toast(result?.remoteOwnershipChanged?'页面离开期间远端已进入新监听轮次；本机旧录音已处理，未中断远端会话。':'页面恢复后已保存待处理录音，并确认实时会话已暂停');
  }catch(error){
    toast(`页面恢复后仍无法确认安全暂停：${error.message}`,true);
  }finally{pageBoundaryRecoveryInProgress=false;}
}

function flushActiveMedia(options={}){
  const queued=mediaFlushTail.catch(()=>{}).then(()=>performMediaFlush(options));
  mediaFlushTail=queued;
  return queued;
}

async function performMediaFlush(options={}){
  mediaFlushInProgress=true;
  renderAll();
  try{return await executeMediaFlush(options);}
  finally{mediaFlushInProgress=false;renderAll();}
}

async function executeMediaFlush({includeMock=true,includeSessionRecording=true}={}){
  liveCaptureGeneration+=1;
  liveCaptureStartPending=false;
  const tasks=[];
  if(includeMock)tasks.push(finalizeMockVoiceRecording({quiet:true}));
  if(includeSessionRecording){
    sessionRecorderStartGeneration+=1;
    sessionRecorderStartPending=false;
    tasks.push(stopSessionRecorder());
  }
  const recorder=liveRecorder;
  if(liveContinuousMode)liveContinuousMode=false;
  if(recorder){
    if(recorder.state==='recording')recorder.stop();
    tasks.push(liveRecorderDone);
  }
  const results=await Promise.allSettled(tasks);
  const failures=results.filter(result=>result.status==='rejected').map(result=>result.reason);
  await Promise.resolve();
  try{
    if(liveAudioChunks.length||liveRecordingUploading)await scheduleLiveRecordingUpload();
  }catch(error){failures.push(error);}
  if(liveContinuousQueue.length&&!liveContinuousUploading)await processLiveContinuousQueue();
  const queueDeadline=Date.now()+90000;
  while((liveContinuousQueue.length||liveContinuousUploading)&&Date.now()<queueDeadline){
    if(liveContinuousQueue.length&&!liveContinuousUploading)await processLiveContinuousQueue();
    if(liveContinuousQueue.length&&!liveContinuousUploading)break;
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  if(liveContinuousQueue.length||liveContinuousUploading)failures.push(new Error('连续监听仍有未保存音频'));
  if(includeSessionRecording){try{await uploadSessionAudio();}catch(error){failures.push(error);}}
  if(failures.length)throw failures[0];
}

async function waitForDesktopShutdownStep(promise,timeoutMs,message){
  let timer;
  const timeoutError=new Error(message);
  timeoutError.desktopShutdownTimedOut=true;
  try{
    return await Promise.race([
      promise,
      new Promise((_,reject)=>{timer=setTimeout(()=>reject(timeoutError),timeoutMs);})
    ]);
  }catch(error){
    if(error.desktopShutdownTimedOut){
      // Do not return control to a window while its old transition can still
      // mutate the server or keep the shared media lock held in the background.
      desktopShutdownAbort?.abort(error);
      api.abortActive(error);
      await promise.catch(()=>{});
    }
    throw error;
  }finally{clearTimeout(timer);}
}

async function prepareDesktopShutdown(){
  if(desktopShutdownPreparing)return;
  desktopShutdownPreparing=true;
  desktopShutdownAbort=new AbortController();
  api.setGlobalAbortSignal(desktopShutdownAbort.signal);
  cancelLiveSuggestionStream();
  const shutdownSessionId=state.sessionId;
  if(shutdownSessionId&&hasLocalLiveCapture(shutdownSessionId)){
    markLivePauseRequired(shutdownSessionId);
  }
  // Invalidate pending permission prompts before waiting on another transition.
  // A cancelled close must never let a late getUserMedia result reopen the mic.
  freezeLocalCapture({includeMock:true,sessionId:state.sessionId});
  try{
    await waitForDesktopShutdownStep(mediaTransitionPromise,30000,'正在执行的会话操作超时');
    const shutdownTransition=runMediaTransition(()=>prepareCurrentSessionBoundary());
    await waitForDesktopShutdownStep(shutdownTransition,90000,'保存录音超时');
    await window.__TAURI__.core.invoke('desktop_renderer_ready_to_exit');
  }catch(error){
    desktopShutdownAbort?.abort(error);
    api.abortActive(error);
    api.setGlobalAbortSignal(null);
    desktopShutdownAbort=null;
    desktopShutdownPreparing=false;
    toast(`关闭已取消：${error.message}。原始录音仍保留，请检查服务后重试。`,true);
    await window.__TAURI__.core.invoke('desktop_renderer_cancel_exit');
    renderAll();
  }
}

async function installDesktopShutdownBridge(){
  if(runtime.mode!=='desktop')return;
  if(!window.__TAURI__)throw new Error('桌面运行时接口不可用');
  const cleanup=[];
  try{
    cleanup.push(await window.__TAURI__.event.listen('desktop-shutdown-requested',prepareDesktopShutdown));
    cleanup.push(await window.__TAURI__.window.getCurrentWindow().onCloseRequested(event=>{
      event.preventDefault();
      void prepareDesktopShutdown();
    }));
    await window.__TAURI__.core.invoke('desktop_shutdown_bridge_ready');
  }catch(error){
    cleanup.reverse().forEach(dispose=>{try{dispose();}catch{}});
    throw error;
  }
}

async function initializeDesktopShutdownBridge(){
  if(desktopShutdownBridgeReady||runtime.mode!=='desktop')return true;
  if(desktopShutdownBridgeInstalling)return desktopShutdownBridgeInstalling;
  desktopShutdownBridgeInstalling=(async()=>{
    try{
      await installDesktopShutdownBridge();
      desktopShutdownBridgeReady=true;
      desktopShutdownBridgeError='';
      renderAll();
      return true;
    }catch(error){
      desktopShutdownBridgeReady=false;
      desktopShutdownBridgeError=`桌面安全关闭保护启动失败：${error.message||error}`;
      renderAll();
      toast(`${desktopShutdownBridgeError}。为避免丢失未保存录音，麦克风功能已停用；再次点击录音可重试。`,true);
      return false;
    }finally{desktopShutdownBridgeInstalling=null;}
  })();
  return desktopShutdownBridgeInstalling;
}

$('live-plan').onclick=async()=>{const button=$('live-plan');busy(button,true);try{const data=await api(`/api/live-interviews/${state.sessionId}/suggestions`,{method:'POST'});state.session=data.state;renderLive();toast('下一问题已准备');}catch(error){toast(error.message,true)}finally{busy(button,false)}};
$('live-export').onclick=async()=>{if(!state.sessionId){toast('请先创建会话',true);return;}try{const text=await api.text(`/api/interviews/sessions/${state.sessionId}/transcript`);const blob=new Blob([text],{type:'text/plain;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`interview-${state.sessionId.slice(0,8)}.txt`;document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);toast('转写文本已导出');}catch(error){toast(error.message,true)}};
async function downloadSessionAudio(recordingId='',storedName=''){
  if(!state.sessionId)return;
  const sessionId=state.sessionId;
  try{
    const suffix=recordingId?`/${encodeURIComponent(recordingId)}`:'';
    const blob=await api.blob(`/api/live-interviews/${sessionId}/audio${suffix}`);
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    const fallback=state.session?.live_interview?.audio_file||'';
    const extension=(storedName||fallback).match(/\.(wav|webm|m4a)$/i)?.[1]?.toLowerCase()||'wav';
    a.href=url;a.download=`interview-${sessionId.slice(0,8)}${recordingId?`-${recordingId.slice(0,8)}`:''}.${extension}`;
    document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);toast('全场录音已下载');
  }catch(error){toast(error.message,true);}
}
$('session-audio-download')?.addEventListener('click',()=>downloadSessionAudio());
document.addEventListener('click',event=>{const button=event.target.closest('[data-session-audio-id]');if(button)void downloadSessionAudio(button.dataset.sessionAudioId,button.dataset.sessionAudioFile||'');});
$('session-audio-delete')?.addEventListener('click',async()=>{
  if(!state.sessionId||!window.confirm('删除这场面试的全部原始录音？转写和评价不会删除。'))return;
  const sessionId=state.sessionId;
  const operationGeneration=snapshotLiveGeneration();
  sessionPartUploadsSuppressed=true;
  try{await runMediaTransition(async()=>{
    freezeLocalCapture({sessionId});
    let utteranceDrainError=null;
    try{await flushActiveMedia({includeMock:false,includeSessionRecording:false});}
    catch(error){utteranceDrainError=error;}
    if(sessionRecorderStartPending){sessionRecorderStartGeneration+=1;await waitForCaptureStartup(sessionRecorderStartPromise);}
    if(sessionAudioUploading)await sessionAudioUploadPromise.catch(()=>{});
    await stopSessionRecorder();
    if(state.session?.live_interview?.status==='active'){
      try{
        const paused=await requestLiveStatus(sessionId,'paused',operationGeneration.statusRevision);
        if(sessionId===state.sessionId)state.session=paused.state;
      }catch{
        const paused=await enforceFailClosedPause(sessionId,operationGeneration);
        if(sessionId===state.sessionId)state.session=paused.state;
      }
    }
    if(utteranceDrainError)throw utteranceDrainError;
    const deleteOperationId=newRecordingPartId();
    const expectedAudioRevision=Number(state.session?.live_interview?.audio_archive_revision||0);
    let data;
    try{
      data=await api(`/api/live-interviews/${sessionId}/audio`,{method:'DELETE',body:JSON.stringify({expected_audio_revision:expectedAudioRevision,operation_id:deleteOperationId})});
    }catch(error){
      const current=await api(`/api/live-interviews/${sessionId}`);
      replaceSessionAudioArchiveState(current.state,sessionId);
      if(String(current.state?.live_interview?.last_audio_delete_operation_id||'')!==deleteOperationId){
        throw new Error('录音集合已在其他窗口发生变化，请再次确认删除');
      }
      data=current;
    }
    // Explicit deletion is the only path that discards queued raw parts. Wait
    // for the server CAS delete before dropping the local recovery copy.
    sessionPartQueue=sessionPartQueue.filter(part=>part.sessionId!==sessionId);
    if(sessionRecordingSessionId===sessionId){sessionChunks=[];sessionRecordingSessionId='';sessionRecordingPartId='';sessionRecordingArchiveRevision=0;sessionRecordingCaptureEpoch=0;sessionPartSequence=0;sessionNextUploadSequence=1;sessionFinalizedPartSequences.clear();}
    sessionRecordingActive=false;sessionAudioSaveError='';sessionAudioErrorSessionId='';sessionRecordingError='';sessionRecordingErrorSessionId='';
    if(sessionId===state.sessionId)state.session=data.state;
  });renderLive();toast('全部原始录音（含未上传分片）已删除');}
  catch(error){toast(`录音删除失败：${error.message}`,true);}
  finally{sessionPartUploadsSuppressed=false;}
});

let liveSuggestionStream = null;
let liveSuggestionAbort = null;
let liveSuggestionGeneration = 0;
function cancelLiveSuggestionStream(){liveSuggestionGeneration+=1;liveSuggestionAbort?.abort();liveSuggestionStream=null;liveSuggestionAbort=null;}
async function streamNextSuggestion() {
  if (!state.sessionId || liveSuggestionStream) return;
  const requestedSessionId=state.sessionId;
  const generation=++liveSuggestionGeneration;
  liveSuggestionStream = true;
  const controller=new AbortController();
  liveSuggestionAbort = controller;
  const node = $('live-suggestion');
  if (node) {
    node.className = 'suggestion-card streaming';
    node.innerHTML = '<p class="streaming-hint">AI 正在生成下一问…</p><div id="streaming-text" class="streaming-text"></div>';
  }
  try {
    const resp = await api.raw(`/api/live-interviews/${requestedSessionId}/suggestions/stream`, {method:'POST', signal: controller.signal});
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    const box = $('streaming-text');
    let buffer = '';
    let full = '';
    let finished = false;
    while (!finished) {
      const { done, value } = await reader.read();
      if(generation!==liveSuggestionGeneration||requestedSessionId!==state.sessionId){
        await reader.cancel();
        throw new DOMException('流式请求所属会话已变更', 'AbortError');
      }
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
            if (box&&generation===liveSuggestionGeneration) box.textContent = full;
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
      if (box&&generation===liveSuggestionGeneration) box.textContent = full;
    }
    // Directly refresh state, bypassing the poll's busy guard.
    try {
      const data = await api(`/api/live-interviews/${requestedSessionId}`);
      if(generation!==liveSuggestionGeneration||requestedSessionId!==state.sessionId)return;
      state.session = data.state;
      renderLive();
    } catch { /* ignore refresh errors */ }
    if (node) node.classList.remove('streaming');
    toast('下一问已流式生成');
  } catch (error) {
    if (error.name !== 'AbortError') {
      toast(error.message, true);
      try {
        const data = await api(`/api/live-interviews/${requestedSessionId}`);
        if(generation!==liveSuggestionGeneration||requestedSessionId!==state.sessionId)return;
        state.session = data.state; renderLive();
      } catch { /* ignore */ }
    }
  } finally {
    if(generation===liveSuggestionGeneration)liveSuggestionStream = null;
    if(liveSuggestionAbort===controller)liveSuggestionAbort = null;
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

async function loadSettings({required=false}={}){const generation=++settingsLoadGeneration;if(required){runtimeSettingsReady=false;state.settings=null;renderLive();}try{const s=await api('/api/settings');if(generation!==settingsLoadGeneration)return null;state.settings=s;runtimeSettingsReady=true;$('setting-provider').value=s.search.selected;$('setting-base-url').value=s.llm.base_url||'';$('setting-model').value=s.llm.model||'';$('setting-input-cost').value=s.llm.input_cost_per_million||0;$('setting-output-cost').value=s.llm.output_cost_per_million||0;$('setting-search-cost').value=s.search.search_request_cost_usd||0;$('setting-asr-url').value=s.asr?.base_url||'';$('setting-asr-path').value=s.asr?.transcription_path||'/v1/audio/transcriptions';$('setting-asr-model').value=s.asr?.model||'whisper-1';$('setting-asr-timeout').value=s.asr?.timeout_seconds||90;const la=s.live_audio||{};const liveAudioMode=document.querySelector('#setting-live-audio-mode');if(liveAudioMode&&s.live_audio!==undefined){liveAudioMode.value=s.live_audio.mode||'asr_text';}
if($('setting-live-audio-name'))$('setting-live-audio-name').value=la.name||'音频直连';
if($('setting-live-audio-url'))$('setting-live-audio-url').value=la.base_url||'http://192.168.1.97:8004/v1';
if($('setting-live-audio-model'))$('setting-live-audio-model').value=la.model||'';
const tts=s.tts||{};if($('setting-tts-url'))$('setting-tts-url').value=tts.base_url||'http://192.168.1.97:8002/v1';if($('setting-tts-path'))$('setting-tts-path').value=tts.speech_path||'/audio/speech';if($('setting-tts-model'))$('setting-tts-model').value=tts.model||'qwen3-tts';if($('setting-tts-voice'))$('setting-tts-voice').value=tts.voice||'温和、专业、清晰的中文声音';
const rl=s.resume_llm||{};if($('setting-resume-llm-url'))$('setting-resume-llm-url').value=rl.base_url||'http://192.168.1.97:8004/v1';if($('setting-resume-llm-model'))$('setting-resume-llm-model').value=rl.model||'';
renderLiveAudioCapability(la.capability);renderLive();return s;}catch(error){if(generation!==settingsLoadGeneration)return null;if(required){runtimeSettingsReady=false;state.settings=null;renderLive();throw error;}toast(error.message,true);return null;}}
async function requireRuntimeSettings(){await loadSettings({required:true});if(!runtimeSettingsReady)throw new Error('运行配置未完成加载，请重试');}
function renderLiveAudioCapability(cap){
  const node=$('live-audio-capability');if(!node)return;
  if(!cap){node.textContent='尚未探测能力。';return;}
  const latency=Number.isFinite(Number(cap.latency_ms))?`（${Number(cap.latency_ms)}ms）`:'';
  node.textContent=cap.ok?`能力可用${latency}${cap.sample?` · 样例「${cap.sample}」`:''}`:`能力不可用：${cap.error||'端点未返回可用建议'}${latency}`;
  node.className=cap.ok?'security-note ok':'security-note error';
}
$('probe-live-audio').onclick=async()=>{const btn=$('probe-live-audio');btn.disabled=true;try{const r=await api(`/api/settings/probe-live-audio`,{method:'POST'});renderLiveAudioCapability(r.capability);toast(r.capability.ok?'音频直连能力探测成功':'音频直连不可用',!r.capability.ok);}catch(error){toast(error.message,true)}finally{btn.disabled=false}};
$('settings-form').onsubmit=async e=>{e.preventDefault();if(mediaTransitionInProgress||hasUnsavedAudio()||state.session?.live_interview?.status==='active'){toast('请先暂停实时面试并保存所有录音，再切换模型或音频配置',true);return;}const form=e.currentTarget;settingsLoadGeneration+=1;runtimeSettingsReady=false;renderAll();busy(form,true);try{await api('/api/settings',{method:'PUT',body:JSON.stringify({search:{provider:$('setting-provider').value,tavily_api_key:providerSecret('setting-tavily','clear-tavily-key'),searxng_base_url:optional('setting-searxng'),brave_api_key:providerSecret('setting-brave','clear-brave-key'),search_request_cost_usd:Number($('setting-search-cost').value)||0},llm:{base_url:optional('setting-base-url'),model:optional('setting-model'),api_key:providerSecret('setting-llm-key','clear-llm-key'),input_cost_per_million:Number($('setting-input-cost').value)||0,output_cost_per_million:Number($('setting-output-cost').value)||0},asr:{base_url:optional('setting-asr-url'),transcription_path:optional('setting-asr-path'),model:optional('setting-asr-model'),timeout_seconds:Number($('setting-asr-timeout').value)||90,api_key:providerSecret('setting-asr-key','clear-asr-key')},tts:{base_url:optional('setting-tts-url'),speech_path:optional('setting-tts-path'),model:optional('setting-tts-model'),voice:optional('setting-tts-voice'),api_key:providerSecret('setting-tts-key','clear-tts-key')},live_audio:{mode:$('setting-live-audio-mode').value,name:optional('setting-live-audio-name'),base_url:optional('setting-live-audio-url'),model:optional('setting-live-audio-model'),api_key:providerSecret('setting-live-audio-key','clear-live-audio-key')},resume_llm:{base_url:optional('setting-resume-llm-url'),model:optional('setting-resume-llm-model'),api_key:providerSecret('setting-resume-llm-key','clear-resume-llm-key')}})});await requireRuntimeSettings();clearAuthenticationEditors();toast('设置已持久化并立即应用');}catch(error){if(!runtimeSettingsReady){try{await requireRuntimeSettings();}catch{}}toast(error.message,true)}finally{busy(form,false)}};

async function refreshDebug(){try{const [status,events,sessions]=await Promise.all([api('/api/debug/status'),api('/api/debug/events?limit=100'),api('/api/debug/sessions')]);const metrics=status.llm.metrics||{};const cache=status.search.cache||{};$('debug-status').innerHTML=[['应用',status.application.status],['模型',status.llm.model||'managed'],['模型地址',status.llm.base_url||'—'],['ASR',status.asr?.model||'—'],['ASR 地址',status.asr?.base_url||'—'],['TTS',status.tts?.model||'—'],['TTS 地址',status.tts?.base_url||'—'],['模型请求 / 失败',`${metrics.requests||0} / ${metrics.failures||0}`],['Token 输入 / 输出',`${metrics.prompt_tokens||0} / ${metrics.completion_tokens||0}`],['模型累计成本',`$${metrics.estimated_cost_usd||0}`],['平均耗时',`${metrics.average_latency_ms||0} ms`],['搜索',status.search.selected],['搜索请求 / 成本',`${status.search.provider_requests||0} / $${status.search.estimated_cost_usd||0}`],['缓存命中 / 未命中',`${cache.hits||0} / ${cache.misses||0}`],['缓存容量',`${cache.size||0} / ${cache.capacity||0}`],['来源筛选',status.search.source_filter_policy||'实体精确匹配 / 可信别名'],['持久缓存',cache.persistent?'已启用':'仅运行时'],['事件持久化',status.events_persistent?'已启用':'仅运行时'],['事件容量',status.event_capacity]].map(([k,v])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');$('debug-sessions').innerHTML=sessions.sessions.map(s=>`<div class="session-row"><span>${esc(s.candidate_name||'未命名')}<small>${esc(s.job_title||'未指定岗位')}</small></span><code>${esc(s.id.slice(0,8))}</code></div>`).join('')||'<div class="empty-state">暂无会话</div>';renderEvents(events.events);}catch(error){toast(error.message,true)}}
function renderEvents(events){$('event-list').innerHTML=events.map(e=>`<div class="event-row"><span>${new Date(e.timestamp).toLocaleTimeString()}</span><span class="${esc(e.level)}">${esc(e.level)}</span><span>${esc(e.agent||e.category)} · ${esc(e.action)}${e.detail?` · ${esc(e.detail)}`:''}</span><span>${e.duration_ms?`${e.duration_ms} ms`:'—'}</span></div>`).join('')||'<div class="empty-state">暂无运行事件</div>';}
$('refresh-events').onclick=refreshDebug;$('probe-llm').onclick=async()=>{try{const d=await api('/api/debug/probes/llm',{method:'POST'});toast(`模型连接正常 · ${d.latency_ms||0} ms`);await refreshDebug();}catch(error){toast(error.message,true)}};

async function refreshLive() {
  if (!state.sessionId || !state.session?.live_interview || mediaTransitionInProgress || mediaFlushInProgress || liveRefreshAbort) return;
  const requestedSessionId=state.sessionId;
  const stateBeforeRequest=state.session;
  const generation=++liveRefreshGeneration;
  const controller=new AbortController();
  liveRefreshAbort=controller;
  const timeout=setTimeout(()=>controller.abort(),12000);
  try {
    let data = await api(`/api/live-interviews/${requestedSessionId}`,{signal:controller.signal});
    if(generation!==liveRefreshGeneration||requestedSessionId!==state.sessionId||requestedSessionId!==activeLoadedSessionId||state.session!==stateBeforeRequest)return;
    if(liveStatusUncertainSessionId===requestedSessionId){
      const expectedGeneration=liveStatusUncertainExpectedGeneration;
      const alternateGeneration=liveStatusUncertainAlternateGeneration;
      try{
        if(!expectedGeneration){clearLiveStatusUncertain(requestedSessionId);throw new Error('缺少原监听轮次，已拒绝暂停远端会话');}
        data=await enforceFailClosedPause(requestedSessionId,expectedGeneration,alternateGeneration);
        if(generation!==liveRefreshGeneration||requestedSessionId!==state.sessionId||requestedSessionId!==activeLoadedSessionId||state.session!==stateBeforeRequest)return;
      }
      catch(error){
        if(generation!==liveRefreshGeneration||requestedSessionId!==state.sessionId||requestedSessionId!==activeLoadedSessionId||state.session!==stateBeforeRequest)return;
        state.session=error.liveState||data.state;
        renderLive();
        return;
      }
    }
    const nextStatus=data.state?.live_interview?.status;
    const nextCaptureEpoch=Number(data.state?.live_interview?.capture_epoch);
    const localCaptureEpoch=activeLocalCaptureEpoch(requestedSessionId);
    const captureEpochChanged=Number.isInteger(localCaptureEpoch)&&Number.isInteger(nextCaptureEpoch)&&localCaptureEpoch!==nextCaptureEpoch;
    if(['paused','completed'].includes(nextStatus))clearLiveStatusUncertain(requestedSessionId);
    const shouldDrain=(['paused','completed'].includes(nextStatus)||captureEpochChanged)&&hasLocalLiveCapture(requestedSessionId);
    state.session = data.state;
    renderLive();
    if(shouldDrain){
      void runMediaTransition(async()=>{freezeLocalCapture({sessionId:requestedSessionId});await flushActiveMedia({includeMock:false,includeSessionRecording:true});})
        .then(()=>{if(captureEpochChanged&&requestedSessionId===state.sessionId)toast('远端已重新开始监听；本机旧轮次收音已停止并保存，请手动重新启动本机语音监听。',true);})
        .catch(error=>toast(`检测到远端已${captureEpochChanged?'重新开始':nextStatus==='paused'?'暂停':'结束'}；本机录音已停止，但保存失败：${error.message}`,true));
    }
  } catch { /* Abort/stale/transient poll failures are retried on the next tick. */ }
  finally { clearTimeout(timeout);if(liveRefreshAbort===controller)liveRefreshAbort=null; }
}

void initializeDesktopShutdownBridge();
setInterval(()=>{if(state.view==='debug')refreshDebug()},4000);
setInterval(()=>{if(state.view==='live'||hasLocalLiveCapture(state.sessionId)||liveStatusUncertainSessionId===state.sessionId)refreshLive()},3000);
setInterval(()=>{void renewPendingLiveCaptureLeases()},5*60*1000);
const requestedView = (location.hash || '').slice(1);
if (navigation.interviewer.some(([view]) => view === requestedView)) state.role = 'interviewer';
if (navigation.candidate.some(([view]) => view === requestedView)) state.role = 'candidate';
document.documentElement.dataset.role = state.role;
setView(viewMeta[requestedView] ? requestedView : `${state.role}-home`);
checkHealth();
async function restoreAuthenticatedSession() {
  try {
    const user = await api('/api/auth/me');
    $('user-chip').textContent = user.username;
    await requireRuntimeSettings();
    await loadSessions();
    await recoverPageBoundary();
  } catch (error) { showLogin(); toast(`运行配置加载失败，已暂停进入工作区：${error.message}`, true); }
}
if (!state.token) { showLogin(); } else { restoreAuthenticatedSession(); }
