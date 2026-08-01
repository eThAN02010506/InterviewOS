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
- Output: Competency scores + hiring recommendation

## CoachAgent
- Input: Question + Answer
- Output: 4-dimension scoring + improved answer

## LiveInterviewAgent
- Input: Live transcript
- Output: Real-time suggestions for interviewer/candidate
