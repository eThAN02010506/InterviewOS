# Agent Specifications

## CandidateAgent
- Role: Candidate Intelligence Analyst
- Input: Resume text
- Output: CandidateProfile (strengths, weaknesses, unique advantages)

## JobAgent
- Role: Job Intelligence Analyst
- Input: Job description text; for title-only input, optional source-bound public JD context marked unconfirmed
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
- Output: 4-dimension scoring + per-dimension behavior anchors and concrete next actions + deterministic STAR completion scaffold
- Provenance: the model returns an untrusted draft; the service assigns scoring source and review status

## Speech Delivery Coach (service boundary)
- Input: Candidate practice audio + final ASR text
- Output: pace, pause, filler, volume-stability, intonation, and clarity coaching
- Trust boundary: no accent/personality/demographic inference; output never becomes Evidence or a hiring signal

## LiveInterviewAgent
- Input: Live transcript
- Output: Real-time suggestions for interviewer/candidate
