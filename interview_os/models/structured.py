"""Utilities for validating structured LLM output."""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, Field

ModelT = TypeVar("ModelT", bound=BaseModel)


class FrameworkItem(BaseModel):
    question_index: int
    answer_framework: str


class FrameworkMap(BaseModel):
    frameworks: list[FrameworkItem] = Field(default_factory=list)

    def index(self) -> dict[int, FrameworkItem]:
        return {item.question_index: item for item in self.frameworks}


class CompetencyNarrativeDraft(BaseModel):
    """Model-written interpretation anchored to numbered persisted evidence."""

    competency_id: str
    evidence_numbers: list[int]
    assessment: str
    next_probe: str


class EvaluationNarrativeDraft(BaseModel):
    """Narrative-only final report draft; scores and gaps are intentionally absent."""

    summary: str
    competency_reviews: list[CompetencyNarrativeDraft] = Field(default_factory=list)


class CustomQuestionAnalysisDraft(BaseModel):
    """Bounded semantic interpretation returned by the local model."""

    answer_type: str
    answer_type_label: str
    assessment_goal: str
    competency: str
    answer_boundary: list[str]
    common_mistakes: list[str]
    transfer_principle: str
    related_questions: list[str]
    likely_follow_ups: list[str]



def parse_model_output(raw: str, model: type[ModelT]) -> ModelT:
    """Parse a JSON object (optionally fenced) and validate it as ``model``.

    Local models frequently wrap otherwise valid JSON in Markdown.  Keeping this
    normalization in one place prevents every domain agent from growing its own
    subtly different parser.
    """
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)

    candidates: list[object] = []
    try:
        decoded = json.loads(text)
        if not isinstance(decoded, dict):
            raise TypeError("structured model output must be a JSON object")
        candidates.append(decoded)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"{", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
                candidates.append(payload)
            except json.JSONDecodeError:
                continue

    validation_error: Exception | None = None
    for payload in candidates:
        if not isinstance(payload, dict):
            continue
        try:
            return model.model_validate(payload)
        except (ValueError, TypeError) as exc:
            validation_error = exc
    if validation_error is not None:
        raise validation_error
    raise ValueError("structured model output did not contain a valid JSON object")
