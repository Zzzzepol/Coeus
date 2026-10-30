from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


Domain = Annotated[
    str, Field(pattern=r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}$", max_length=253)
]


class ResearchTask(Model):
    preferred_domains: list[Domain] = Field(default_factory=list, max_length=4)
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
    domain: str = ""
    published_date: str | None = None
    retrieved_at: str | None = None
    authority_hint: str = "unverified"


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
    claims: list[Claim] = Field(default_factory=list, max_length=16)
    limitations: list[str] = Field(min_length=1, max_length=12)


class ClaimReview(Model):
    claim_id: str = Field(min_length=1, max_length=40)
    supported: bool
    reason: str = Field(min_length=3, max_length=500)


class Critique(Model):
    claim_reviews: list[ClaimReview] = Field(default_factory=list, max_length=17)
    verdict: Literal["pass", "revise"]
    issues: list[str] = Field(max_length=12)
    suggestions: list[str] = Field(max_length=12)
