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

    # Only ONE match — shortName is already in the search result, no extra API call needed
    if len(search.quotes) == 1:
        q = search.quotes[0]
        return {
            "status": "resolved",
            "ticker": q['symbol'],
            "name": q.get('longname') or q.get('shortname', q['symbol']),
        }

    # MULTIPLE matches — use fields already present in the search result (zero extra API calls)
    candidates = [
        {
            "ticker": q['symbol'],
            "name": q.get('longname') or q.get('shortname', q['symbol']),
            "exchange": q.get('exchange', 'Unknown'),
        }
        for q in search.quotes[:5]
    ]

    return {"status": "needs_confirmation", "candidates": candidates}

