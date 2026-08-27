export const state = {
  sessionId: localStorage.getItem('interviewos.session') || '',
  role: localStorage.getItem('interviewos.role') || 'candidate',
  token: localStorage.getItem('interviewos.token') || '',
  session: null,
  view: '',
  settings: null
};

export const navigation = {
  candidate: [
    ['candidate-home', '总览'], ['candidate', '面试准备'], ['mock', '模拟面试'], ['candidate-report', '改进报告']
  ],
  interviewer: [
    ['interviewer-home', '总览'], ['enterprise', '面试设计'], ['live', '实时辅助'], ['evaluation', '候选人评估']
  ]
};

export const globalViews = new Set(['settings', 'debug']);

export const viewMeta = {
  'candidate-home':['CANDIDATE WORKSPACE','候选人工作台'],
  candidate:['PERSONAL STRATEGY','面试准备'], mock:['PRACTICE & EVIDENCE','模拟面试'],
  'candidate-report':['GROWTH & REVIEW','个人改进报告'],
  'interviewer-home':['INTERVIEWER WORKSPACE','面试官工作台'],
  enterprise:['INTERVIEW ARCHITECTURE','面试设计'], live:['LIVE INTERVIEW COPILOT','实时面试辅助'], evaluation:['EVIDENCE REVIEW','候选人评估'],
  settings:['RUNTIME CONFIGURATION','模型与搜索'], debug:['LOCAL OBSERVABILITY','Debug Console']
};

const nextActionLabels = {
  'Review resume checks, then continue the interview workflow':'先完成简历待确认项，再继续生成面试策略',
  'Start mock interview':'开始模拟面试',
  'Answer the current mock interview question':'回答当前模拟面试问题',
  'Answer the next mock interview question':'回答下一道模拟面试问题',
  'Answer the evidence-seeking follow-up question':'回答当前证据追问',
  'Generate evidence-based evaluation':'生成基于证据的面试评价',
  'Mock interview completed without answers':'本轮尚无回答，可返回重新练习',
  'Listen to the interview and prepare the next question':'监听面试并准备下一道问题',
  'Review the transcript before final evaluation':'在最终评价前审阅转写内容',
  'Interviewer reviews the suggested next question':'审阅并决定是否采用建议问题',
  'Review more live evidence or generate evaluation':'继续补充现场证据，或生成候选人评价',
  'Review updated live evidence or generate evaluation':'审阅更新后的证据，或生成候选人评价',
  'Review corrected live evidence before evaluation':'评价前审阅修正后的现场证据',
  'Regenerate candidate-dependent interview artifacts':'重新生成与候选人相关的面试材料',
  'Regenerate evaluation after human score review':'人工复核后重新生成最终评价',
  'Generate evidence-based hiring evaluation':'生成基于证据的招聘评价',
  'Review the final evaluation and feedback':'审阅最终评价与改进反馈',
  'Evaluation was interrupted; retry ending the interview':'评价过程曾中断，请重新结束面试以继续',
  'Answer the current custom mock interview question':'回答刚添加的自定义问题',
  'Review the custom question in the mock interview pool':'在模拟面试题库中查看自定义问题'
};

export function localizedNextAction(value) {
  return nextActionLabels[value] || value || '创建一个会话，然后选择候选人准备或企业面试设计。';
}

export function candidateHomeAction(session) {
  if (!session) return {view:'', label:'创建第一个会话'};
  if (session.mock_session?.status === 'completed') return {view:'candidate-report', label:'查看本轮改进报告'};
  if (session.mock_session?.status === 'active') return {view:'mock', label:'继续模拟面试'};
  if (session.strategy?.summary || session.mock_interview?.questions?.length) return {view:'mock', label:'开始模拟面试'};
  return {view:'candidate', label:'补充资料并生成策略'};
}
