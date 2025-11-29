# src/run_mcp_answers.py

import asyncio
import json
import csv
from pathlib import Path

from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SERVER = {
    "command": "python",
    "args": ["mcp_server/server.py"],   # ← Run LOCAL server directly
    "cwd": str(Path(__file__).resolve().parent.parent),  # cleantech_mcp_agent
}

INPUT_CSV = "queries.csv"
OUTPUT_CSV = "answers.csv"

# ------------------------------------------------------------
# TOOL CALL LAYER
# ------------------------------------------------------------
async def ask_mcp(question: str) -> str:
    """
    Calls the 'answer_question' tool on the local MCP server.
    """

    async with stdio_client(SERVER) as (read_stream, write_stream):

        # Build the correct MCP tool request (MCP v1.0 / FastMCP-compliant)
        req = CallToolRequest(
            method="tools/call",
            params={
                "name": "answer_question",
                "arguments": {"question": question}
            }
        )

        # SEND REQUEST
        await write_stream.send(req.model_dump())

        # WAIT FOR RESPONSE
        async for msg in read_stream:
            if msg.get("result"):
                try:
                    answer = msg["result"]["content"][0]["text"]
                except:
                    answer = json.dumps(msg["result"], indent=2)
                return answer

        return "ERROR_NO_RESPONSE"  # failsafe


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
async def main():
    # Load input CSV
    rows = []
    with open(INPUT_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for r in reader:
            rows.append(r)

    results = [["question", "answer"]]

    # Process queries sequentially
    for i, (question, _) in enumerate(rows, 1):
        print(f"\n→ Asking MCP ({i}/{len(rows)}): {question[:60]}...")

        try:
            answer = await ask_mcp(question)
        except Exception as e:
            answer = f"ERROR: {e}"

        results.append([question, answer])

    # Write output CSV
    with open(OUTPUT_CSV, "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print("\nDONE. Answers saved to:", OUTPUT_CSV)


# ------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
