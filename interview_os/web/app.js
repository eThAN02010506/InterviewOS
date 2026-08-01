const state = { sessionId: localStorage.getItem('interviewos.session') || '', session: null, view: 'overview' };
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const optional = id => $(id).value.trim() || null;
const safeUrl = value => { try { const url = new URL(value); return ['http:','https:'].includes(url.protocol) ? url.href : '#'; } catch { return '#'; } };

async function api(path, options = {}) {
  const response = await fetch(path, { headers: {'Content-Type':'application/json', ...(options.headers || {})}, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
  return data;
}

function toast(message, error = false) {
  const node = $('toast'); node.textContent = message; node.className = error ? 'show error' : 'show';
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.className = '', 3200);
}

function busy(form, active) { form.classList.toggle('loading', active); }
function tags(items = []) { return `<div class="tag-list">${items.map(v => `<span class="tag">${esc(v)}</span>`).join('')}</div>`; }
function list(title, items = []) { return items.length ? `<div class="result-block"><h4>${esc(title)}</h4><ul>${items.map(v => `<li>${esc(v)}</li>`).join('')}</ul></div>` : ''; }

const viewMeta = {
  overview:['LOCAL INTELLIGENCE','面试智能工作台'], candidate:['CANDIDATE INTELLIGENCE','候选人准备'],
  enterprise:['HIRING INTELLIGENCE','企业面试设计'], mock:['PRACTICE & EVIDENCE','模拟面试'],
  settings:['RUNTIME CONFIGURATION','模型与搜索'], debug:['LOCAL OBSERVABILITY','Debug Console']
};

function setView(view) {
  state.view = view;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${view}`));
  document.querySelectorAll('.nav-item').forEach(v => v.classList.toggle('active', v.dataset.view === view));
  [$('view-eyebrow').textContent, $('view-title').textContent] = viewMeta[view];
  location.hash = view;
  if (view === 'debug') refreshDebug();
  if (view === 'settings') loadSettings();
  if (view === 'mock') refreshMock();
}

async function checkHealth() {
  try { await api('/health'); $('health-dot').classList.add('ok'); $('health-label').textContent = '本地服务已连接'; }
  catch { $('health-dot').classList.remove('ok'); $('health-label').textContent = '本地服务不可用'; }
}

async function loadSessions() {
  try {
    const data = await api('/api/debug/sessions'); const select = $('session-select');
    select.innerHTML = '<option value="">选择会话</option>' + data.sessions.map(s => `<option value="${esc(s.id)}">${esc(s.candidate_name || '未命名')} · ${esc(s.job_title || '未指定岗位')}</option>`).join('');
    if (state.sessionId && data.sessions.some(s => s.id === state.sessionId)) select.value = state.sessionId;
    else if (data.sessions[0]) select.value = state.sessionId = data.sessions[0].id;
    if (state.sessionId) await loadSession();
  } catch (error) { toast(error.message, true); }
}

async function loadSession() {
  if (!state.sessionId) { state.session = null; renderState(); return; }
  try { const data = await api(`/api/interviews/sessions/${state.sessionId}`); state.session = data.state; localStorage.setItem('interviewos.session', state.sessionId); renderState(); }
  catch (error) { toast(error.message, true); }
}

function renderState() {
  const s = state.session;
  $('metric-workflow').textContent = s?.workflow?.status || '未开始';
  $('metric-step').textContent = s?.workflow?.current_step || '等待输入资料';
  const sources = (s?.company?.public_sources?.length || 0) + (s?.interviewer?.public_expressions?.length || 0);
  $('metric-sources').textContent = sources; $('metric-questions').textContent = s?.mock_interview?.questions?.length || 0; $('metric-evidence').textContent = s?.evidence?.length || 0;
  $('next-action').textContent = s?.next_action || '创建一个会话，然后选择候选人准备或企业面试设计。';
  const profiles = [['候选人', !!s?.candidate?.skills?.length],['岗位',!!s?.job?.competencies?.length],['公司',!!s?.company?.dna],['面试官',!!s?.interviewer?.name]];
  $('profile-progress').innerHTML = profiles.map(([name, done]) => `<div class="progress-item"><span>${name}</span><div class="progress-track"><i style="width:${done?100:8}%"></i></div><b>${done?'完成':'待分析'}</b></div>`).join('');
  renderStrategy(); renderBlueprint(); renderSources(); renderMock();
}

function renderStrategy() {
  const strategy = state.session?.strategy; const node = $('strategy-result');
  if (!strategy?.summary && !strategy?.key_risks?.length) { node.className='empty-state'; node.textContent='完成分析后，这里会显示风险、回答框架和重点话题。'; return; }
  node.className=''; node.innerHTML=`<p>${esc(strategy.summary)}</p>${list('关键风险',strategy.key_risks)}${list('回答框架',strategy.answer_framework)}${list('重点强调',strategy.topics_to_emphasize)}${list('可能问题',strategy.likely_questions)}`;
}

function renderSources() {
  const s=state.session; const sources=[...(s?.company?.public_sources||[]),...(s?.interviewer?.public_expressions||[])]; const node=$('source-result');
  if (!sources.length) { node.className='source-list empty-state'; node.textContent='尚未检索。'; return; }
  node.className='source-list'; node.innerHTML=sources.map(x=>`<a href="${esc(safeUrl(x.url))}" target="_blank" rel="noreferrer"><strong>${esc(x.title||x.url||'公开资料')}</strong><small>${esc((x.snippet||x.text||'').slice(0,150))}</small></a>`).join('');
}

function renderBlueprint() {
  const blueprint=state.session?.blueprint; const node=$('blueprint-result');
  if (!blueprint?.rounds?.length) { node.className='empty-state'; node.textContent='工作流完成后显示每轮目标、问题和强信号。'; return; }
  node.className=''; node.innerHTML=blueprint.rounds.map((round,i)=>`<div class="round-card"><p class="eyebrow">ROUND ${i+1}</p><h4>${esc(round.name)}</h4><p>${esc(round.goal)}</p>${tags(round.evaluation_criteria)}${(round.questions||[]).map(q=>`<div class="question-item"><strong>${esc(q.question)}</strong><small>${esc(q.competency)}</small></div>`).join('')}</div>`).join('');
}

function renderMock() {
  const plan=state.session?.mock_interview?.questions||[]; const session=state.session?.mock_session; const index=session?.current_question_index||0;
  $('mock-progress').textContent=`${Math.min(index,plan.length)} / ${plan.length}`; $('mock-status').textContent=session?.status==='active'?'面试进行中':session?.status==='completed'?'本轮已完成':'准备开始';
  const current=session?.status==='active'?plan[index]:null; $('start-mock').style.display=session?.status==='idle'||!session?'inline-block':'none'; $('answer-form').style.display=current?'block':'none';
  $('mock-question').className=current?'question-copy':'question-copy empty-state'; $('mock-question').innerHTML=current?`<small>${esc(current.competency||'综合能力')}</small>${esc(current.question)}`:(session?.status==='completed'?'所有问题均已完成，可以进入证据评价。':'先完成候选人准备工作流，生成个性化问题。');
  const last=session?.responses?.at(-1); const node=$('coach-result');
  if (!last) { node.className='empty-state'; node.textContent='提交回答后显示内容、深度、结构和影响力评分。'; return; }
  const e=last.evaluation; node.className=''; node.innerHTML=`<div class="score-grid">${[['内容',e.content],['深度',e.technical_depth],['结构',e.structure],['影响',e.impact]].map(([n,v])=>`<div class="score"><span>${n}</span><strong>${Math.round(v*100)}</strong></div>`).join('')}</div>${list('改进建议',e.feedback)}<div class="result-block"><h4>优化回答</h4><p>${esc(e.improved_answer)}</p></div>`;
}

async function ensureSession() { if (state.sessionId) return true; $('session-dialog').showModal(); toast('请先创建一个会话'); return false; }

$('nav').addEventListener('click', e => { const button=e.target.closest('[data-view]'); if(button) setView(button.dataset.view); });
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>setView(b.dataset.go));
$('new-session').onclick=()=> $('session-dialog').showModal();
$('close-session-dialog').onclick=()=> $('session-dialog').close();
$('session-select').onchange=async e=>{state.sessionId=e.target.value;await loadSession();};
$('session-form').onsubmit=async e=>{e.preventDefault();try{const data=await api('/api/interviews/sessions',{method:'POST',body:JSON.stringify({candidate_name:$('new-candidate').value,job_title:$('new-job').value,company_name:$('new-company').value})});state.sessionId=data.id;$('session-dialog').close();await loadSessions();toast('会话已创建');}catch(error){toast(error.message,true)}};

$('candidate-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);try{const data=await api('/api/workflows/candidate-prep',{method:'POST',body:JSON.stringify({session_id:state.sessionId,resume_text:$('candidate-resume').value,job_description:$('candidate-jd').value,company_name:$('candidate-company').value,company_context:$('candidate-company-context').value,interviewer_name:$('interviewer-name').value,interviewer_position:$('interviewer-position').value})});state.session=data.state;renderState();toast('候选人策略已生成');}catch(error){toast(error.message,true)}finally{busy(form,false)}};
$('enterprise-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);try{const data=await api('/api/workflows/enterprise-design',{method:'POST',body:JSON.stringify({session_id:state.sessionId,resume_text:$('enterprise-resume').value,job_description:$('enterprise-jd').value,company_name:$('enterprise-company').value,company_context:$('enterprise-context').value})});state.session=data.state;renderState();toast('面试 Blueprint 已生成');}catch(error){toast(error.message,true)}finally{busy(form,false)}};

$('start-mock').onclick=async()=>{if(!await ensureSession())return;try{await api(`/api/mock-interviews/${state.sessionId}/start`,{method:'POST'});await loadSession();toast('模拟面试已开始');}catch(error){toast(error.message,true)}};
$('answer-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;const question=state.session?.mock_interview?.questions?.[state.session.mock_session.current_question_index];if(!question)return;busy(form,true);try{await api(`/api/mock-interviews/${state.sessionId}/answers`,{method:'POST',body:JSON.stringify({question_id:question.id,answer:$('mock-answer').value})});$('mock-answer').value='';await loadSession();toast('回答已评分');}catch(error){toast(error.message,true)}finally{busy(form,false)}};

async function loadSettings(){try{const s=await api('/api/settings');$('setting-provider').value=s.search.selected;$('setting-base-url').value=s.llm.base_url||'';$('setting-model').value=s.llm.model||'';}catch(error){toast(error.message,true)}}
$('settings-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;busy(form,true);try{await api('/api/settings',{method:'PUT',body:JSON.stringify({search:{provider:$('setting-provider').value,tavily_api_key:optional('setting-tavily'),searxng_base_url:optional('setting-searxng'),brave_api_key:optional('setting-brave')},llm:{base_url:optional('setting-base-url'),model:optional('setting-model'),api_key:optional('setting-llm-key')}})});toast('设置已立即应用');await loadSettings();}catch(error){toast(error.message,true)}finally{busy(form,false)}};

async function refreshDebug(){try{const [status,events,sessions]=await Promise.all([api('/api/debug/status'),api('/api/debug/events?limit=100'),api('/api/debug/sessions')]);$('debug-status').innerHTML=[['应用',status.application.status],['模型',status.llm.model||'managed'],['模型地址',status.llm.base_url||'—'],['搜索',status.search.selected],['事件容量',status.event_capacity]].map(([k,v])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');$('debug-sessions').innerHTML=sessions.sessions.map(s=>`<div class="session-row"><span>${esc(s.candidate_name||'未命名')}<small>${esc(s.job_title||'未指定岗位')}</small></span><code>${esc(s.id.slice(0,8))}</code></div>`).join('')||'<div class="empty-state">暂无会话</div>';renderEvents(events.events);}catch(error){toast(error.message,true)}}
function renderEvents(events){$('event-list').innerHTML=events.map(e=>`<div class="event-row"><span>${new Date(e.timestamp).toLocaleTimeString()}</span><span class="${esc(e.level)}">${esc(e.level)}</span><span>${esc(e.agent||e.category)} · ${esc(e.action)}${e.detail?` · ${esc(e.detail)}`:''}</span><span>${e.duration_ms?`${e.duration_ms} ms`:'—'}</span></div>`).join('')||'<div class="empty-state">暂无运行事件</div>';}
$('refresh-events').onclick=refreshDebug;$('probe-llm').onclick=async()=>{try{const d=await api('/api/debug/probes/llm',{method:'POST'});toast(`模型连接正常 · ${d.latency_ms||0} ms`);await refreshDebug();}catch(error){toast(error.message,true)}};

setInterval(()=>{if(state.view==='debug')refreshDebug()},4000);
setView((location.hash||'#overview').slice(1) in viewMeta?(location.hash||'#overview').slice(1):'overview');checkHealth();loadSessions();
