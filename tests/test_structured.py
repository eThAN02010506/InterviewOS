import pytest
from pydantic import BaseModel

from interview_os.models.structured import parse_model_output


class Example(BaseModel):
    name: str


def test_parse_fenced_model_output():
    parsed = parse_model_output('```json\n{"name": "Ada"}\n```', Example)
    assert parsed.name == "Ada"


def test_parse_json_surrounded_by_explanation():
    parsed = parse_model_output('Result follows: {"name": "Ada"} done.', Example)
    assert parsed.name == "Ada"


def test_parse_rejects_non_object():
    with pytest.raises(TypeError):
        parse_model_output('[{"name": "Ada"}]', Example)
