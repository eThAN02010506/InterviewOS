const state = {
  sessionId: localStorage.getItem('interviewos.session') || '',
  role: localStorage.getItem('interviewos.role') || 'candidate',
  session: null,
  view: ''
};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const optional = id => $(id).value.trim() || null;
const safeUrl = value => { try { const url = new URL(value); return ['http:','https:'].includes(url.protocol) ? url.href : '#'; } catch { return '#'; } };

async function api(path, options = {}) {
  const headers = options.body instanceof FormData ? {...(options.headers || {})} : {'Content-Type':'application/json', ...(options.headers || {})};
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
  return data;
}

function toast(message, error = false) {
  const node = $('toast'); node.textContent = message; node.className = error ? 'show error' : 'show';
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.className = '', 3200);
}

function busy(form, active) { form.classList.toggle('loading', active); }
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
    ['interviewer-home', '总览'], ['enterprise', '面试设计'], ['evaluation', '候选人评估']
  ]
};
const globalViews = new Set(['settings', 'debug']);
const viewMeta = {
  'candidate-home':['CANDIDATE WORKSPACE','候选人工作台'],
  candidate:['PERSONAL STRATEGY','面试准备'], mock:['PRACTICE & EVIDENCE','模拟面试'],
  'candidate-report':['GROWTH & REVIEW','个人改进报告'],
  'interviewer-home':['INTERVIEWER WORKSPACE','面试官工作台'],
  enterprise:['INTERVIEW ARCHITECTURE','面试设计'], evaluation:['EVIDENCE REVIEW','候选人评估'],
  settings:['RUNTIME CONFIGURATION','模型与搜索'], debug:['LOCAL OBSERVABILITY','Debug Console']
};

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
  state.view = view;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${view}`));
  [$('view-eyebrow').textContent, $('view-title').textContent] = viewMeta[view];
  renderNavigation();
  location.hash = view;
  if (view === 'debug') refreshDebug();
  if (view === 'settings') loadSettings();
  if (view === 'mock') renderMock();
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
  try { const data = await api(`/api/interviews/sessions/${state.sessionId}`); state.session = data.state; localStorage.setItem('interviewos.session', state.sessionId); hydrateSessionForms(); renderState(); }
  catch (error) { toast(error.message, true); }
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
  $('metric-workflow').textContent = s?.workflow?.status || '未开始';
  $('metric-step').textContent = s?.autopilot?.enabled ? `AI · ${s.autopilot.phase || s.autopilot.status}` : (s?.workflow?.current_step || '等待输入资料');
  const sources = (s?.company?.public_sources?.length || 0) + (s?.interviewer?.public_expressions?.length || 0);
  $('metric-sources').textContent = sources; $('metric-questions').textContent = s?.mock_interview?.questions?.length || 0; $('metric-evidence').textContent = s?.evidence?.length || 0;
  $('next-action').textContent = s?.next_action || '创建一个会话，然后选择候选人准备或企业面试设计。';
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
  renderStrategy(); renderBlueprint(); renderSources(); renderMock(); renderReports();
}

function renderResumeReview() {
  const review = state.session?.resume_review;
  const targets = [$('candidate-resume-review'), $('enterprise-resume-review')];
  targets.forEach(node => {
    if (!review?.metadata?.filename) { node.innerHTML = ''; node.classList.remove('visible'); return; }
    const unresolved = (review.claims || []).filter(claim => claim.status === 'unverified');
    node.classList.add('visible');
    node.innerHTML = `<div class="review-summary"><strong>${esc(review.metadata.filename)}</strong><span>${review.metadata.character_count} 字 · ${review.issues.length} 项提示 · ${unresolved.length} 项待确认</span></div>
      ${(review.issues || []).map(issue => `<div class="review-issue ${esc(issue.severity)}"><b>${esc(issue.severity === 'warning' ? '请检查' : '提示')}</b><span>${esc(issue.message)}</span></div>`).join('')}
      ${unresolved.slice(0, 8).map(claim => `<div class="review-claim"><div><small>${esc(claim.category)}</small><span>${esc(claim.statement)}</span></div><div class="claim-actions"><button type="button" data-claim-action="confirmed" data-claim-id="${esc(claim.id)}">确认</button><button type="button" data-claim-action="modified" data-claim-id="${esc(claim.id)}">修改</button><button type="button" data-claim-action="needs_documents" data-claim-id="${esc(claim.id)}">要材料</button><button type="button" data-claim-action="ignored" data-claim-id="${esc(claim.id)}">忽略</button></div></div>`).join('')}
      ${unresolved.length > 8 ? `<p class="review-more">另有 ${unresolved.length - 8} 项，可在后续审阅中处理。</p>` : ''}`;
  });
}

function renderJDReview() {
  const review=state.session?.job_review;
  [$('candidate-jd-review'),$('enterprise-jd-review')].forEach(node=>{
    if(!review?.requirements?.length && !review?.warnings?.length){node.innerHTML='';return;}
    const explicit=(review.requirements||[]).filter(x=>x.origin==='explicit');
    const inferred=(review.requirements||[]).filter(x=>x.origin==='inferred');
    node.innerHTML=`<div class="review-summary"><strong>JD 完整度 ${Math.round((review.completeness_score||0)*100)}%</strong><span>${review.is_title_only?'仅职位名称，无法生成岗位专属问题':'已完成结构检查'}</span></div>${review.missing_sections?.length?`<p class="jd-warning">请补充：${esc(review.missing_sections.join('、'))}</p>`:''}${list('明确要求',explicit.map(x=>x.text))}${list('AI 推测（需确认）',inferred.map(x=>x.text))}`;
  });
}

function renderFacts(){
  const node=$('fact-result'); const cards=state.session?.fact_cards||[];
  if(!cards.length){node.className='fact-list empty-state';node.textContent='尚未形成事实卡。';return;}
  const labels={verified:'已验证',inferred:'推测',conflict:'冲突'};
  node.className='fact-list';node.innerHTML=cards.map(card=>`<article class="fact-card ${esc(card.status)}"><div><span>${esc(card.category)}</span><b>${esc(labels[card.status]||card.status)}</b></div><strong>${esc(card.subject)}</strong><p>${esc(card.claim)}</p><small>${esc(card.note||'')}</small>${(card.source_urls||[]).map((url,i)=>`<a href="${esc(safeUrl(url))}" target="_blank" rel="noreferrer">来源 ${i+1}</a>`).join('')}</article>`).join('');
}

function maybePromptEntityResolution(){
  const dialog=$('entity-dialog'); if(dialog.open)return;
  const item=(state.session?.entity_resolutions||[]).find(x=>x.status==='pending'); if(!item)return;
  dialog.dataset.resolutionId=item.id;$('entity-copy').textContent=`搜索结果显示“${item.proposed_name}”可能是“${item.input_name}”的正确实体。是否将“${item.input_name}”更正为“${item.proposed_name}”？`;
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
  const evidenceHtml = competencyItems.map(item => `<div class="evidence-row"><div><strong>${esc(item.competency)}</strong><small>${esc((item.supporting_evidence || [item.signal]).filter(Boolean).join(' · '))}</small>${item.gaps?.length?`<em>缺口：${esc(item.gaps.join(' · '))}</em>`:''}</div><b>${Math.round((item.score ?? item.confidence) * 100)}</b></div>`).join('');
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
  const s=state.session; const sources=[...(s?.company?.public_sources||[]),...(s?.interviewer?.public_expressions||[])]; const node=$('source-result');
  if (!sources.length) {
    const statuses=[s?.company?.public_research_status,s?.interviewer?.public_research_status];
    node.className='source-list empty-state';
    node.textContent=statuses.includes('no_reliable_sources')?'已检索，但没有找到可可靠归属于该公司或人物的公开资料。':statuses.includes('failed')?'公开检索失败，请检查搜索设置后重试。':'尚未检索。';
    return;
  }
  const qualityLabels={official:'官方',high:'高可信',secondary:'二手来源',unrated:'未评级'};
  node.className='source-list'; node.innerHTML=sources.map(x=>{const alias=x.identity_match==='corroborated_alias'?`名称近似匹配：输入“${x.input_identity}”，来源“${x.matched_identity}” · `:'';return `<a href="${esc(safeUrl(x.url))}" target="_blank" rel="noreferrer"><strong>${esc(x.title||x.url||'公开资料')}</strong><small>${esc(alias)}${esc(qualityLabels[x.source_quality]||'未评级')} · ${esc((x.snippet||x.text||'').slice(0,150))}</small></a>`}).join('');
}

function renderBlueprint() {
  const blueprint=state.session?.blueprint; const node=$('blueprint-result');
  if (!blueprint?.rounds?.length) { node.className='empty-state'; node.textContent='工作流完成后显示每轮目标、问题和强信号。'; return; }
  node.className=''; node.innerHTML=blueprint.rounds.map((round,i)=>`<div class="round-card"><p class="eyebrow">ROUND ${i+1}</p><h4>${esc(round.name)}</h4><p>${esc(round.goal)}</p>${tags(round.evaluation_criteria)}${(round.questions||[]).map(q=>`<div class="question-item"><strong>${esc(q.question)}</strong><small>${esc(q.competency)}</small></div>`).join('')}</div>`).join('');
}

function renderMock() {
  const plan=state.session?.mock_interview?.questions||[]; const session=state.session?.mock_session; const index=session?.current_question_index||0;
  $('mock-progress').textContent=`${Math.min(index,plan.length)} / ${plan.length}`; $('mock-status').textContent=session?.status==='active'?'面试进行中':session?.status==='completed'?'本轮已完成':'准备开始';
  const current=session?.status==='active'?plan[index]:null; const questionText=session?.pending_follow_up||current?.question; $('start-mock').style.display=session?.status==='idle'||!session?'inline-block':'none'; $('answer-form').style.display=current?'block':'none';
  $('mock-question').className=current?'question-copy':'question-copy empty-state'; $('mock-question').innerHTML=current?`<small>${session?.pending_follow_up?'证据追问':esc(current.competency||'综合能力')}</small>${esc(questionText)}`:(session?.status==='completed'?'所有问题均已完成，最终报告已自动生成。':'先完成候选人准备工作流，生成个性化问题。');
  const last=session?.responses?.at(-1); const node=$('coach-result');
  if (!last) { node.className='empty-state'; node.textContent='提交回答后显示内容、深度、结构和影响力评分。'; return; }
  const e=last.evaluation; node.className=''; node.innerHTML=`<div class="score-grid">${[['内容',e.content],['深度',e.technical_depth],['结构',e.structure],['影响',e.impact]].map(([n,v])=>`<div class="score"><span>${n}</span><strong>${Math.round(v*100)}</strong></div>`).join('')}</div>${list('改进建议',e.feedback)}<div class="result-block"><h4>优化回答</h4><p>${esc(e.improved_answer)}</p></div>`;
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
  busy(upload, true);
  try {
    const data = await api(`/api/resumes/${state.sessionId}/upload`, {method:'POST', body:form});
    state.session = data.state;
    $(upload.dataset.resumeTarget).value = state.session.candidate.raw_resume_text;
    renderState(); toast('简历已解析，请先查看校验项');
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
    state.session = data.state; renderState(); toast({confirmed:'已确认，后续 Agent 可使用',needs_documents:'已标记为需要材料',ignored:'已忽略，后续 Agent 不会使用'}[button.dataset.claimAction]||'已更新');
  } catch (error) { toast(error.message, true); }
});

$('cancel-claim').onclick=()=>$('claim-dialog').close();
$('save-claim').onclick=async()=>{const dialog=$('claim-dialog');try{const data=await api(`/api/resumes/${state.sessionId}/claims/${dialog.dataset.claimId}`,{method:'PATCH',body:JSON.stringify({status:'modified',statement:$('claim-statement').value,note:$('claim-note').value})});state.session=data.state;dialog.close();renderState();toast('修改后的事实已确认，后续 Agent 将使用新表述');}catch(error){toast(error.message,true)}};

async function resolveEntity(accept){const dialog=$('entity-dialog');try{const data=await api(`/api/intelligence/${state.sessionId}/entities/${dialog.dataset.resolutionId}`,{method:'PATCH',body:JSON.stringify({accept})});state.session=data.state;dialog.close();hydrateSessionForms();renderState();toast(accept?'实体名称已更正':'已保留原名称');}catch(error){toast(error.message,true)}}
$('accept-entity').onclick=()=>resolveEntity(true);$('reject-entity').onclick=()=>resolveEntity(false);

$('candidate-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);showWorkflowStarting();try{const payload={role:'candidate',resume_text:$('candidate-resume').value,job_description:$('candidate-jd').value,company_name:$('candidate-company').value,company_context:$('candidate-company-context').value,interviewer_name:$('interviewer-name').value,interviewer_position:$('interviewer-position').value,authorized_public_research:$('candidate-research-consent').checked};const endpoint=$('candidate-autopilot').checked?`/api/autopilot/${state.sessionId}/run`:'/api/workflows/candidate-prep';if(!$('candidate-autopilot').checked)payload.session_id=state.sessionId;const data=await api(endpoint,{method:'POST',body:JSON.stringify(payload)});state.session=data.state;await loadSessions();toast(state.session.autopilot?.enabled?'AI 已推进到需要你回答的阶段':'候选人策略已生成');}catch(error){await loadSession();toast(`运行失败：${error.message}`,true)}finally{busy(form,false)}};
$('enterprise-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;if(!await ensureSession())return;busy(form,true);showWorkflowStarting();try{const payload={role:'interviewer',resume_text:$('enterprise-resume').value,job_description:$('enterprise-jd').value,company_name:$('enterprise-company').value,company_context:$('enterprise-context').value,authorized_public_research:$('enterprise-research-consent').checked};const endpoint=$('enterprise-autopilot').checked?`/api/autopilot/${state.sessionId}/run`:'/api/workflows/enterprise-design';if(!$('enterprise-autopilot').checked)payload.session_id=state.sessionId;const data=await api(endpoint,{method:'POST',body:JSON.stringify(payload)});state.session=data.state;await loadSessions();toast(state.session.autopilot?.enabled?'AI 已完成设计，等待采集真实面试证据':'面试 Blueprint 已生成');}catch(error){await loadSession();toast(`运行失败：${error.message}`,true)}finally{busy(form,false)}};

$('start-mock').onclick=async()=>{if(!await ensureSession())return;try{await api(`/api/mock-interviews/${state.sessionId}/start`,{method:'POST'});await loadSession();toast('模拟面试已开始');}catch(error){toast(error.message,true)}};
$('answer-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;const mockSession=state.session?.mock_session;const question=state.session?.mock_interview?.questions?.[mockSession?.current_question_index];if(!question)return;busy(form,true);try{await api(`/api/mock-interviews/${state.sessionId}/answers`,{method:'POST',body:JSON.stringify({question_id:mockSession.pending_parent_question_id||question.id,answer:$('mock-answer').value})});$('mock-answer').value='';await loadSession();toast(state.session?.mock_session?.pending_follow_up?'回答已评分，AI 正在追问缺失证据':'回答已评分');}catch(error){toast(error.message,true)}finally{busy(form,false)}};
document.querySelectorAll('.evaluation-trigger').forEach(button => button.onclick = async () => {
  if (!await ensureSession()) return;
  busy(button, true);
  try {
    const data = await api(`/api/evaluations/${state.sessionId}`, {method:'POST'});
    state.session = data.state;
    renderState();
    toast('最终评价已生成');
  } catch (error) { toast(error.message, true); }
  finally { busy(button, false); }
});

async function submitTranscript(){if(!await ensureSession())return;const form=$('transcript-form');const lines=$('transcript-input').value.split('\n').map(x=>x.trim()).filter(Boolean);const entries=lines.map(line=>{const parts=line.split('|').map(x=>x.trim());return {competency:parts[0]||'综合能力',question:parts[1]||'',answer:parts.slice(2).join(' | ')};});if(!entries.length||entries.some(x=>!x.question||!x.answer)){toast('每行必须包含“能力 | 问题 | 回答”三部分',true);return;}busy(form,true);try{const data=await api(`/api/evaluations/${state.sessionId}/transcript`,{method:'POST',body:JSON.stringify({entries,auto_evaluate:true})});state.session=data.state;renderState();toast(`已导入 ${entries.length} 条真实回答并生成评价`);}catch(error){toast(error.message,true)}finally{busy(form,false)}}
$('transcript-form').onsubmit=e=>{e.preventDefault();submitTranscript();};
$('import-transcript').onclick=submitTranscript;

async function loadSettings(){try{const s=await api('/api/settings');$('setting-provider').value=s.search.selected;$('setting-base-url').value=s.llm.base_url||'';$('setting-model').value=s.llm.model||'';$('setting-input-cost').value=s.llm.input_cost_per_million||0;$('setting-output-cost').value=s.llm.output_cost_per_million||0;$('setting-search-cost').value=s.search.search_request_cost_usd||0;}catch(error){toast(error.message,true)}}
$('settings-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget;busy(form,true);try{await api('/api/settings',{method:'PUT',body:JSON.stringify({search:{provider:$('setting-provider').value,tavily_api_key:optional('setting-tavily'),searxng_base_url:optional('setting-searxng'),brave_api_key:optional('setting-brave'),search_request_cost_usd:Number($('setting-search-cost').value)||0},llm:{base_url:optional('setting-base-url'),model:optional('setting-model'),api_key:optional('setting-llm-key'),input_cost_per_million:Number($('setting-input-cost').value)||0,output_cost_per_million:Number($('setting-output-cost').value)||0}})});toast('设置已持久化并立即应用');await loadSettings();}catch(error){toast(error.message,true)}finally{busy(form,false)}};

async function refreshDebug(){try{const [status,events,sessions]=await Promise.all([api('/api/debug/status'),api('/api/debug/events?limit=100'),api('/api/debug/sessions')]);const metrics=status.llm.metrics||{};const cache=status.search.cache||{};$('debug-status').innerHTML=[['应用',status.application.status],['模型',status.llm.model||'managed'],['模型地址',status.llm.base_url||'—'],['模型请求 / 失败',`${metrics.requests||0} / ${metrics.failures||0}`],['Token 输入 / 输出',`${metrics.prompt_tokens||0} / ${metrics.completion_tokens||0}`],['模型累计成本',`$${metrics.estimated_cost_usd||0}`],['平均耗时',`${metrics.average_latency_ms||0} ms`],['搜索',status.search.selected],['搜索请求 / 成本',`${status.search.provider_requests||0} / $${status.search.estimated_cost_usd||0}`],['缓存命中 / 未命中',`${cache.hits||0} / ${cache.misses||0}`],['持久缓存',cache.persistent?'已启用':'仅运行时'],['事件持久化',status.events_persistent?'已启用':'仅运行时'],['事件容量',status.event_capacity]].map(([k,v])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');$('debug-sessions').innerHTML=sessions.sessions.map(s=>`<div class="session-row"><span>${esc(s.candidate_name||'未命名')}<small>${esc(s.job_title||'未指定岗位')}</small></span><code>${esc(s.id.slice(0,8))}</code></div>`).join('')||'<div class="empty-state">暂无会话</div>';renderEvents(events.events);}catch(error){toast(error.message,true)}}
function renderEvents(events){$('event-list').innerHTML=events.map(e=>`<div class="event-row"><span>${new Date(e.timestamp).toLocaleTimeString()}</span><span class="${esc(e.level)}">${esc(e.level)}</span><span>${esc(e.agent||e.category)} · ${esc(e.action)}${e.detail?` · ${esc(e.detail)}`:''}</span><span>${e.duration_ms?`${e.duration_ms} ms`:'—'}</span></div>`).join('')||'<div class="empty-state">暂无运行事件</div>';}
$('refresh-events').onclick=refreshDebug;$('probe-llm').onclick=async()=>{try{const d=await api('/api/debug/probes/llm',{method:'POST'});toast(`模型连接正常 · ${d.latency_ms||0} ms`);await refreshDebug();}catch(error){toast(error.message,true)}};

setInterval(()=>{if(state.view==='debug')refreshDebug()},4000);
const requestedView = (location.hash || '').slice(1);
if (navigation.interviewer.some(([view]) => view === requestedView)) state.role = 'interviewer';
if (navigation.candidate.some(([view]) => view === requestedView)) state.role = 'candidate';
setView(viewMeta[requestedView] ? requestedView : `${state.role}-home`);
checkHealth(); loadSessions();
