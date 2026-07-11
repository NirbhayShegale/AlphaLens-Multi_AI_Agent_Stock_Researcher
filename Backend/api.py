from fastapi import FastAPI
from pydantic import BaseModel
from AlphaLens.graph.graph import workflow
from langsmith import traceable
from Query_Extraction.query_extraction import ticker_extraction_tool

app = FastAPI()

class QueryRequest(BaseModel):
    userquery: str

@app.get('/')
def read_root():
    return {"message": "Welcome to the Stock Research Agent API!"}


@app.post("/search_company")
@traceable(run_type="chain", name="API: Search Company") 
def company_stock(request: QueryRequest):
    ticker = ticker_extraction_tool(request.userquery)
    
    return ticker


@app.post("/research")
@traceable(run_type="chain", name="API: Research Stock")
async def research_stock(request: QueryRequest):

    orchestrator_state = await workflow.ainvoke({"ticker": request.userquery})
    memo = orchestrator_state["synthesis"]
    
    return memo