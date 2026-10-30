from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearchTask(Model):
    query: str = Field(min_length=3, max_length=500)
    objective: str = Field(min_length=3, max_length=1000)


class Plan(Model):
    tasks: list[ResearchTask] = Field(min_length=1, max_length=8)


class Document(Model):
    name: str = Field(min_length=1, max_length=250)
    text: str = Field(min_length=1, max_length=100_000)


class Source(Model):
    id: str
    title: str
    uri: str
    kind: Literal["web", "document"]
    content: str


class Finding(Model):
    source_id: str
    quote: str = Field(min_length=12, max_length=1500)
    claim: str = Field(min_length=3, max_length=1500)


class Findings(Model):
    findings: list[Finding] = Field(max_length=6)


class Evidence(Finding):
    id: str
    task_id: str


class Claim(Model):
    text: str = Field(min_length=3, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class Report(Model):
    title: str = Field(min_length=3, max_length=200)
    answer: Claim | None = None
    claims: list[Claim] = Field(min_length=1, max_length=16)
    limitations: list[str] = Field(min_length=1, max_length=12)


class Critique(Model):
    verdict: Literal["pass", "revise"]
    issues: list[str] = Field(max_length=12)
    suggestions: list[str] = Field(max_length=12)
