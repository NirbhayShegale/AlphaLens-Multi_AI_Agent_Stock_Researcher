from Query_Extraction.query_extraction_model import TickerExtraction
from AlphaLens.config.llms import get_report_writer_llm
from langchain_core.prompts import ChatPromptTemplate
import requests
import yfinance as yf
from AlphaLens.utils.yf_utils import yf_delay



def ticker_extraction_tool(query: str) -> dict:
    """Used to find the official company and ticker of the company."""
    structured_llm = get_report_writer_llm().with_structured_output(TickerExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
                you have to extract the full official name of the company
                Always use the Safe Search tool for web Search and use that for finding the official name of the company
                """),
        ("human", f"""
                    Analyze the following user query:
                    ---
                    Query: "{{user_prompt}}"
                    ---
                    """)
        ])

    chain = prompt | structured_llm
    llm_response = chain.invoke({"user_prompt": query})
    result = llm_response.model_dump()

    if result['company_name'] == 'UNKNOWN':
        return {"status": "not_found", "message": "Could not identify a company from your query."}

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 ..."})

    yf_delay()
    search = yf.Search(result['company_name'], max_results=5, session=session)  # 5, not 1
    #no match
    if not search.quotes:
        return {"status": "not_found", "message": f"No ticker found for '{result['company_name']}'"}

    # Only ONE match
    if len(search.quotes) == 1:
        yf_delay()
        ticker_obj = yf.Ticker(search.quotes[0]['symbol'])
        info = ticker_obj.info
        return {"status": "resolved", "ticker": search.quotes[0]['symbol'], "name": info.get('longName') or info.get('shortName') }

    # MULTIPLE matches 
    candidates = []
    for quote in search.quotes[:5]:
        yf_delay()
        ticker_obj = yf.Ticker(quote['symbol'])
        info = ticker_obj.info
        candidates.append({
            "ticker": quote['symbol'],
            "name": info.get('longName') or info.get('shortName', quote['symbol']),
            "exchange": info.get('exchange', 'Unknown'),
        })

    return {"status": "needs_confirmation", "candidates": candidates}

