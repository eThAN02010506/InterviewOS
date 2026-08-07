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

    def index(self) -> dict[int, str]:
        return {item.question_index: item.answer_framework for item in self.frameworks}



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
