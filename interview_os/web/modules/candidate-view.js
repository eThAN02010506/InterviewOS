import {state} from './state.js';
import {$, esc, formatDateTime, list, safeUrl, tags} from './ui.js';

export function renderResumeReview() {
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
export function renderJDReview() {
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
export function renderFacts(){
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
export function renderReports() {
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
export function renderStrategy() {
  const strategy = state.session?.strategy; const node = $('strategy-result');
  if (!strategy?.summary && !strategy?.key_risks?.length) { node.className='empty-state'; node.textContent='完成分析后，这里会显示风险、回答框架和重点话题。'; return; }
  node.className=''; node.innerHTML=`<p>${esc(strategy.summary)}</p>${list('关键风险',strategy.key_risks)}${list('回答框架',strategy.answer_framework)}${list('重点强调',strategy.topics_to_emphasize)}${list('可能问题',strategy.likely_questions)}`;
}
export function renderSources() {
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
export function renderBlueprint() {
  const blueprint=state.session?.blueprint; const node=$('blueprint-result');
  if (!blueprint?.rounds?.length) { node.className='empty-state'; node.textContent='工作流完成后显示每轮目标、问题和强信号。'; return; }
  node.className=''; node.innerHTML=blueprint.rounds.map((round,i)=>`<div class="round-card"><p class="eyebrow">ROUND ${i+1}</p><h4>${esc(round.name)}</h4><p>${esc(round.goal)}</p>${tags(round.evaluation_criteria)}${(round.questions||[]).map(q=>`<div class="question-item"><strong>${esc(q.question)}</strong><small>${esc(q.competency)}</small></div>`).join('')}</div>`).join('');
}
