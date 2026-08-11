# Agent Specifications

## CandidateAgent
- Role: Candidate Intelligence Analyst
- Input: Resume text
- Output: CandidateProfile (strengths, weaknesses, unique advantages)

## JobAgent
- Role: Job Intelligence Analyst
- Input: Job description text
- Output: JobDescription (competencies, required skills)

## CompanyAgent
- Role: Company Intelligence Analyst
- Input: Company name + info
- Output: CompanyInfo (DNA, culture, preferences)

## InterviewerAgent (CORE INNOVATION)
- Role: Interviewer Intelligence Analyst
- Input: Name, position, company, public info
- Output: InterviewerProfile (career pattern, communication style, likely preferences)

## InterviewStrategyAgent
- Role: Three-way fusion
- Input: Candidate + Job + Interviewer profiles
- Output: Personalized interview strategy

## MockInterviewAgent
- Input: All profiles
- Output: Personalized questions with evaluation signals

## EvaluationAgent
- Input: All evidence
- Output: Deterministic competency scores, evidence narratives, and hiring recommendation
- Trust boundary: final aggregation makes no LLM call; every conclusion is derived from persisted Evidence

## CoachAgent
- Input: Question + Answer
- Output: 4-dimension scoring + deterministic STAR completion scaffold
- Provenance: the model returns an untrusted draft; the service assigns scoring source and review status

## LiveInterviewAgent
- Input: Live transcript
- Output: Real-time suggestions for interviewer/candidate
