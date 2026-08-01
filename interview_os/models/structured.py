"""Utilities for validating structured LLM output."""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


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

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, dict):
        raise TypeError("structured model output must be a JSON object")
    return model.model_validate(payload)
