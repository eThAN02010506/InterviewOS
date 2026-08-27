import {esc, list, tags} from './ui.js';

export function renderSpokenAnswerAnalysis(evaluation) {
  const analysis=evaluation?.spoken_analysis;
  if(!analysis?.question_coverage?.length&&!analysis?.semantic_steps?.length)return '';
  const typeLabels={behavioral_example:'\u884c\u4e3a\u6848\u4f8b',situational:'\u60c5\u666f\u63a8\u6f14',motivation:'\u52a8\u673a\u5339\u914d',methodology:'\u65b9\u6cd5\u8bba',general:'\u7efc\u5408\u56de\u7b54'};
  const steps=(analysis.semantic_steps||[]).map((item,index)=>`<li><strong>${index+1}. ${esc(item.label)}</strong><span>${esc(item.evidence)}</span></li>`).join('');
  const coverage=(analysis.question_coverage||[]).map(item=>`<div class="review-claim"><div><small>${esc({covered:'\u5df2\u8986\u76d6',partial:'\u90e8\u5206\u8986\u76d6',missing:'\u672a\u8986\u76d6'}[item.status]||item.status)}</small><span>${esc(item.requirement)}</span><small>${esc(item.evidence||item.suggestion)}</small></div></div>`).join('');
  const cleaned=analysis.cleaned_transcript&&analysis.cleaned_transcript!==analysis.raw_transcript?`<div class="result-block"><h4>\u6e05\u6d17\u540e\u8bed\u4e49\u7a3f\uff08\u539f\u8f6c\u5199\u4ecd\u4fdd\u7559\uff09</h4><p>${esc(analysis.cleaned_transcript)}</p></div>`:'';
  const calibrated=(analysis.calibration_notes||[]).length?list('\u8bc1\u636e\u4e00\u81f4\u6027\u6821\u51c6',analysis.calibration_notes):'';
  return `<div class="result-block"><h4>\u56de\u7b54\u7c7b\u578b\uff1a${esc(typeLabels[analysis.answer_type]||analysis.answer_type||'\u7efc\u5408\u56de\u7b54')}</h4>${steps?`<ol class="semantic-steps">${steps}</ol>`:''}</div>${cleaned}${coverage?`<div class="result-block"><h4>\u95ee\u9898\u8981\u6c42\u8986\u76d6</h4>${coverage}</div>`:''}${calibrated}`;
}
export function renderQuestionAlignment(evaluation) {
  const coverage=evaluation?.spoken_analysis?.question_coverage||[];
  if(!coverage.length)return '';
  const labels={covered:'已回答',partial:'回答了一部分',missing:'没有回答'};
  return `<section class="alignment-feedback"><div class="review-subtitle">这道题问了什么，你实际回答了什么</div>${coverage.map(item=>`<article class="alignment-row ${esc(item.status)}"><div><strong>${esc(item.requirement)}</strong><b>${esc(labels[item.status]||item.status)}</b></div><p>${item.evidence?`你的原话：“${esc(item.evidence)}”`:'你的回答中没有找到对应内容。'}</p>${item.status!=='covered'?`<small>下次这样补：${esc(item.suggestion)}</small>`:''}</article>`).join('')}</section>`;
}
export function renderRetryComparison(current, previous) {
  if(!current||!previous)return '';
  const dimensions=[['证据','content'],['深度','technical_depth'],['结构','structure'],['结果','impact']];
  const deltas=dimensions.map(([label,key])=>{const before=Math.round((previous.evaluation?.[key]||0)*100),after=Math.round((current.evaluation?.[key]||0)*100),delta=after-before;return `<div class="score"><span>${label}</span><strong>${delta>0?'+':''}${delta}</strong><small>${before} → ${after}</small></div>`}).join('');
  const beforeCoverage=new Map((previous.evaluation?.spoken_analysis?.question_coverage||[]).map(item=>[item.requirement,item.status]));
  const improved=(current.evaluation?.spoken_analysis?.question_coverage||[]).filter(item=>item.status==='covered'&&beforeCoverage.get(item.requirement)!=='covered').map(item=>item.requirement);
  return `<div class="result-block retry-comparison"><h4>本次重答对比</h4><div class="score-grid">${deltas}</div>${improved.length?list('新补齐的要求',improved):'<small>本次尚未新增完整覆盖项，可继续针对首要缺口练习。</small>'}<div class="claim-actions">${previous.audio_file?`<button type="button" data-play-mock-audio="${esc(previous.id)}">播放上一版</button><button type="button" data-delete-mock-audio="${esc(previous.id)}">删除上一版录音</button>`:''}</div></div>`;
}
export function renderQuestionDeepAnalysis(current, understanding) {
  const sourceLabels={explicit_jd:'明确 JD',title_inference:'仅岗位名推测',generic:'通用题型'};
  const levelLabels={strong:'优秀回答',acceptable:'合格回答',risk:'风险回答'};
  const stageLabels={foundation:'基础澄清',evidence:'证据验证',tradeoff:'取舍深挖',pressure:'压力迁移'};
  const levels=(understanding.answer_levels||[]).map(item=>`<article class="answer-level ${esc(item.level)}"><strong>${esc(levelLabels[item.level]||item.level)}</strong><p>${esc(item.description)}</p>${tags(item.observable_signals||[])}</article>`).join('');
  const stories=(understanding.candidate_story_options||[]).map(item=>`<article class="story-option"><small>已确认简历事实</small><strong>${esc(item.claim)}</strong><p>${esc(item.fit_reason)}</p><em>作答重点：${esc(item.adaptation_focus)}</em></article>`).join('');
  const storyGuidance=(understanding.story_selection_guidance||[]).length?`<div class="story-guidance"><small>${stories?'选材边界':'当前没有匹配的已确认经历，先按以下标准选材'}</small>${tags(understanding.story_selection_guidance)}</div>`:'';
  const probes=(understanding.probe_tree||[]).map(item=>`<article class="probe-node ${esc(item.stage)}"><small>${esc(stageLabels[item.stage]||item.stage)}</small><button type="button" data-related-question="${esc(item.question)}" data-related-competency="${esc(understanding.competency||current.competency||'')}">${esc(item.question)}</button><p>${esc(item.purpose)}</p><em>何时追问：${esc(item.entry_condition)}</em></article>`).join('');
  if(!understanding.role_relevance&&!levels&&!probes)return '';
  return `<details class="deep-question-analysis"><summary>展开深度解析 · 岗位关联、决策标准、选材与压力题树</summary><div class="deep-analysis-body"><section><div class="deep-analysis-title"><h4>为什么这个岗位会问</h4><span>${esc(sourceLabels[understanding.role_relevance_source]||'分析依据待确认')}</span></div><p>${esc(understanding.role_relevance)}</p>${(understanding.secondary_competencies||[]).length?`<small>同时观察</small>${tags(understanding.secondary_competencies)}`:''}</section>${(understanding.decision_criteria||[]).length?`<section><h4>面试官如何形成判断</h4><ol>${understanding.decision_criteria.map(item=>`<li>${esc(item)}</li>`).join('')}</ol></section>`:''}${levels?`<section><h4>回答质量分界</h4><div class="answer-levels">${levels}</div></section>`:''}<section><h4>从已确认经历中选材</h4>${stories||storyGuidance?`${stories}${storyGuidance}`:'<p>暂无已确认且与本题匹配的简历事实。请先完成简历事实确认，系统不会替你虚构案例。</p>'}</section>${probes?`<section><h4>由浅入深的追问题树</h4><p class="deep-analysis-help">按回答暴露的缺口逐级追问；点击任一问题可切换练习，不会自动加入题池。</p><div class="probe-tree">${probes}</div></section>`:''}</div></details>`;
}
