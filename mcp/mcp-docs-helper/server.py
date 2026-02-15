import os
from pathlib import Path
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

APP_NAME = "docs-helper"
DOCS_DIR = Path(__file__).parent / "docs"

mcp = FastMCP(APP_NAME)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


@mcp.tool()
def search_docs(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search local docs/ folder for query and return best matches.
    """
    query_l = query.lower().strip()
    if not query_l:
        return {"query": query, "results": []}

    results: List[Dict[str, Any]] = []

    for p in DOCS_DIR.rglob("*"):
        if p.is_dir() or p.suffix.lower() not in {".md", ".txt"}:
            continue

        text = _read_text(p)
        text_l = text.lower()

        # super simple scoring: count occurrences
        score = text_l.count(query_l)
        if score == 0:
            continue

        # return a short snippet around first match
        idx = text_l.find(query_l)
        start = max(0, idx - 60)
        end = min(len(text), idx + 140)
        snippet = text[start:end].replace("\n", " ").strip()

        results.append(
            {
                "file": str(p.relative_to(DOCS_DIR)),
                "score": score,
                "snippet": snippet,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"query": query, "results": results[: max_results]}


@mcp.tool()
def summarize(text: str, bullets: int = 4) -> Dict[str, Any]:
    """
    Tiny summarizer (no LLM): picks first N sentences-ish as bullets.
    Replace with a real LLM later.
    """
    cleaned = " ".join(text.replace("\n", " ").split())
    # naive split
    parts = [p.strip() for p in cleaned.split(".") if p.strip()]
    summary = parts[:bullets]
    return {"bullets": summary, "chars_in": len(text)}


if __name__ == "__main__":
    # MCP standard transport for local tools: stdio
    mcp.run()
