import argparse
import asyncio
import json
from typing import Any, Dict, List

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def merge_snippets(results: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    chunks: List[str] = []
    total = 0

    for r in results:
        file = r.get("file", "unknown")
        score = r.get("score", 0)
        snippet = (r.get("snippet") or "").strip()

        block = f"[{file} | score={score}]\n{snippet}\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                chunks.append(block[:remaining])
            break

        chunks.append(block)
        total += len(block)

    return "\n".join(chunks).strip()


def _extract_json(tool_resp: Any) -> Dict[str, Any]:
    """
    MCP tool responses often return content items (TextContent).
    FastMCP commonly returns JSON inside .text.
    """
    if hasattr(tool_resp, "content") and tool_resp.content:
        item = tool_resp.content[0]
        text = getattr(item, "text", None)
        if isinstance(text, str):
            return json.loads(text)
        data = getattr(item, "data", None)
        if isinstance(data, dict):
            return data

    data = getattr(tool_resp, "data", None)
    if isinstance(data, dict):
        return data

    if isinstance(tool_resp, dict):
        return tool_resp

    raise TypeError(f"Unexpected tool response shape: {type(tool_resp)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="MCP agent loop: search_docs -> merge -> summarize")
    parser.add_argument("--query", "-q", required=True, help="Search query for docs")
    parser.add_argument("--server", default="server.py", help="Path to MCP server script (default: server.py)")
    parser.add_argument("--max-results", type=int, default=5, help="Max search results (default: 5)")
    parser.add_argument("--bullets", type=int, default=4, help="Bullets in summary (default: 4)")
    args = parser.parse_args()

    server_params = StdioServerParameters(
        command="python",
        args=[args.server],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1) search
            search_resp = await session.call_tool(
                "search_docs",
                {"query": args.query, "max_results": args.max_results},
            )
            search_payload = _extract_json(search_resp)
            results = search_payload.get("results", [])
            if not results:
                print(f"No matches found for query: {args.query!r}")
                return

            # 2) merge
            merged = merge_snippets(results)

            # 3) summarize
            sum_resp = await session.call_tool(
                "summarize",
                {"text": merged, "bullets": args.bullets},
            )
            sum_payload = _extract_json(sum_resp)
            bullets = sum_payload.get("bullets", [])

            # 4) print final
            print("\n=== Top matches ===")
            for r in results[: args.max_results]:
                print(f"- {r.get('file')} (score={r.get('score')}): {r.get('snippet')}")

            print("\n=== Final answer (summary) ===")
            for i, b in enumerate(bullets, 1):
                print(f"{i}. {b}")


if __name__ == "__main__":
    asyncio.run(main())
