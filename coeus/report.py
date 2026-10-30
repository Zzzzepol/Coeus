import re
from urllib.parse import quote, urlsplit

from coeus.models import Evidence, Report, Source


def plain(text: str) -> str:
    """Keep model/source text from adding Markdown links, HTML, or fake citations."""
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", " ".join(text.split()))


def validate_report(report: Report, evidence: list[Evidence]):
    known = {e.id for e in evidence}
    for claim in report.claims:
        if any(ref not in known for ref in claim.evidence_ids):
            raise ValueError("Report references unknown evidence identifiers.")


def render_report(
    question: str,
    report: Report,
    evidence: list[Evidence],
    sources: list[Source],
    mode: str,
    review: str,
    warnings: list[str],
) -> str:
    validate_report(report, evidence)
    by_id = {e.id: e for e in evidence}
    source_map = {s.id: s for s in sources}
    used = list(dict.fromkeys(ref for c in report.claims for ref in c.evidence_ids))
    lines = [
        f"# {plain(report.title)}",
        "",
        f"**Question:** {plain(question)}",
        "",
        f"**Mode:** {mode} · **Critic:** {review}",
        "",
    ]
    if mode == "demo":
        lines += ["> DEMO — synthetic fixtures; this is not a live research result.", ""]
    lines += ["## Findings", ""]
    for claim in report.claims:
        refs = " ".join(f"[^{ref}]" for ref in dict.fromkeys(claim.evidence_ids))
        lines += [f"- {plain(claim.text)} {refs}", ""]
    lines += ["## Limitations", ""]
    for item in report.limitations + warnings:
        lines.append(f"- {plain(item)}")
    lines += ["", "## Evidence and sources", ""]
    for ref in used:
        e = by_id[ref]
        s = source_map[e.source_id]
        uri = s.uri
        if s.kind == "web" and urlsplit(uri).scheme in {"http", "https"}:
            label = f"[{plain(s.title)}]({quote(uri, safe=':/?=&%+#@')})"
        else:
            label = f"{plain(s.title)} (uploaded document)"
        lines += [f"[^{ref}]: {label} — “{plain(e.quote)}”", ""]
    return "\n".join(lines)
