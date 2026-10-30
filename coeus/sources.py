"""Transparent source hints and diversity selection, not authority guarantees."""

from collections import defaultdict, deque
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonical_url(url):
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Invalid source URL")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in {"gclid", "fbclid"}
    ]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", urlencode(query), ""))


def authority_hint(url, preferred=()):
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    preferred = [domain.lower().removeprefix("www.") for domain in preferred]
    if any(host == domain or host.endswith("." + domain) for domain in preferred):
        return "preferred domain"
    if host.startswith(("docs.", "developer.", "developers.")) or host.endswith((".gov", ".edu")):
        return "documentation or institution (heuristic)"
    return "unverified"


def ranked_results(results, limit, preferred=()):
    unique = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            uri = canonical_url(result.get("url", ""))
        except (ValueError, TypeError, AttributeError):
            continue
        unique.setdefault(uri, {**result, "url": uri})
    ranked = sorted(
        unique.values(), key=lambda r: authority_hint(r["url"], preferred) == "unverified"
    )
    groups = defaultdict(deque)
    for result in ranked:
        groups[urlsplit(result["url"]).hostname].append(result)
    selected = []
    while groups and len(selected) < limit:
        for host in list(groups):
            selected.append(groups[host].popleft())
            if not groups[host]:
                del groups[host]
            if len(selected) == limit:
                break
    return selected


def diverse_evidence(evidence, sources):
    groups = defaultdict(deque)
    ordered = sorted(evidence, key=lambda e: sources[e.source_id].authority_hint == "unverified")
    for item in ordered:
        source = sources[item.source_id]
        groups[source.domain or source.id].append(item)
    while groups:
        for domain in list(groups):
            yield groups[domain].popleft()
            if not groups[domain]:
                del groups[domain]
