"""Deterministic providers used only by automated tests."""

import asyncio

from coeus.models import Claim, ClaimReview, Critique, Finding, Findings, Plan, Report, ResearchTask


class FixtureProvider:
    """Deterministic fixtures, not live research or a factual research service."""

    async def close(self):
        pass

    async def search(self, query):
        await asyncio.sleep(0)
        return [
            {
                "title": "Research test fixture",
                "url": "https://example.com/coeus-demo",
                "content": "A bounded worker pool limits the number of concurrent research tasks. "
                "SQLite stores sources and evidence locally for later inspection.",
            }
        ]

    async def generate(self, operation, instructions, data, schema):
        await asyncio.sleep(0)
        if schema is Plan:
            return Plan(
                tasks=[
                    ResearchTask(
                        query="bounded research workers",
                        objective="Inspect concurrency constraints",
                    ),
                    ResearchTask(
                        query="local evidence persistence", objective="Inspect evidence storage"
                    ),
                ]
            )
        if schema is Findings:
            findings = []
            for source in data["sources"][:3]:
                excerpt = source["content"].split("\n\n", 1)[-1][:300].strip()
                last_sentence = excerpt.rfind(".")
                if last_sentence >= 11:
                    excerpt = excerpt[: last_sentence + 1]
                if len(excerpt) >= 12:
                    findings.append(
                        Finding(
                            source_id=source["id"],
                            quote=excerpt,
                            claim="The supplied source states: " + excerpt,
                        )
                    )
            return Findings(findings=findings)
        if schema is Report:
            unique = {}
            for item in data["evidence"]:
                unique.setdefault((item["source_id"], item["quote"]), item)
            return Report(
                title="Test research report",
                claims=[
                    Claim(text=e["claim"], evidence_ids=[e["id"]])
                    for e in list(unique.values())[:4]
                ],
                limitations=["Offline fixture run: no live search or LLM was used."],
            )
        if schema is Critique:
            return Critique(
                verdict="pass",
                issues=[],
                suggestions=[],
                claim_reviews=[
                    ClaimReview(
                        claim_id=key, supported=True, reason="Supported by the supplied quote"
                    )
                    for key in data.get("claims", {})
                ],
            )
        raise ValueError(operation)
