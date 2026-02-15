import argparse
import asyncio
from typing import Any, Dict, List

# MCP python client APIs (same package you installed: `mcp`)
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client


def merge_snippets(results: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    """
    Merge top snippets into one context block (with lightweight formatting).
    """
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


# uv run python agent.py --query "vpn"
async def main() -> None:
    parser = argparse.ArgumentParser(description="Simple MCP agent loop: search_docs -> merge -> summarize")
    parser.add_argument("--query", "-q", required=True, help="Search query for docs")
    parser.add_argument(
        "--server",
        default="server.py",
        help="Path to your MCP server script (default: server.py)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Max search results to merge (default: 5)",
    )
    parser.add_argument(
        "--bullets",
        type=int,
        default=4,
        help="How many bullets in summary (default: 4)",
    )
    args = parser.parse_args()

    # Start MCP server as a subprocess over stdio
    # Equivalent to running: python server.py
    async with stdio_client(["python", args.server]) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize handshake
            await session.initialize()

            # 1) call search_docs
            search_resp = await session.call_tool(
                "search_docs",
                {
                    "query": args.query,
                    "max_results": args.max_results,
                },
            )

            # MCP tool responses typically return a structured "content" array.
            # FastMCP will generally put JSON in a text blob. We'll handle both common shapes.
            payload = None
            if hasattr(search_resp, "content") and search_resp.content:
                # Usually first item is a TextContent with .text
                item = search_resp.content[0]
                payload = getattr(item, "text", None) or getattr(item, "data", None)
            else:
                payload = getattr(search_resp, "data", None) or search_resp

            # If payload is a string (JSON text), try to parse it
            if isinstance(payload, str):
                import json
                payload = json.loads(payload)

            results = (payload or {}).get("results", [])
            if not results:
                print(f"No matches found for query: {args.query!r}")
                return

            # 2) merge snippets
            merged = merge_snippets(results)

            # 3) call summarize
            sum_resp = await session.call_tool(
                "summarize",
                {
                    "text": merged,
                    "bullets": args.bullets,
                },
            )

            sum_payload = None
            if hasattr(sum_resp, "content") and sum_resp.content:
                item = sum_resp.content[0]
                sum_payload = getattr(item, "text", None) or getattr(item, "data", None)
            else:
                sum_payload = getattr(sum_resp, "data", None) or sum_resp

            if isinstance(sum_payload, str):
                import json
                sum_payload = json.loads(sum_payload)

            bullets = (sum_payload or {}).get("bullets", [])

            # 4) print final answer
            print("\n=== Top matches ===")
            for r in results[: args.max_results]:
                print(f"- {r.get('file')} (score={r.get('score')}): {r.get('snippet')}")

            print("\n=== Merged context ===")
            print(merged)

            print("\n=== Final answer (summary) ===")
            for i, b in enumerate(bullets, 1):
                print(f"{i}. {b}")


if __name__ == "__main__":
    asyncio.run(main())
