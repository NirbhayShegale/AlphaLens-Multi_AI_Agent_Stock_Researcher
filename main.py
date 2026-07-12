from AlphaLens.graph.graph import workflow
from Query_Extraction.ticker_selection import ticker_resolver
import asyncio
import json
from langsmith import traceable

@traceable
async def main(userquery: str):
    candidate = ticker_resolver(userquery)
    orchestrator_state = await workflow.ainvoke({"ticker": candidate})
    memo = orchestrator_state["synthesis"]
    f_memo=json.dumps(memo, indent=2)
    print(f_memo)

if __name__ == "__main__":
    query=input("query :")
    asyncio.run(main(query))