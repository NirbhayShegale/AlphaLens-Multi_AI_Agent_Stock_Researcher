from Query_Extraction.query_extraction_model import TickerExtraction
from AlphaLens.config.llms import get_report_writer_llm
from langchain_core.prompts import ChatPromptTemplate
import requests
from dotenv import load_dotenv
import os
load_dotenv()

list_of_exc={
    'NASDAQ',
    'NYSE',
    'NSE',
    'BSE'
}

def ticker_extraction_tool(query: str) -> dict:
    """Used to find the official company and ticker of the company."""
    structured_llm = get_report_writer_llm().with_structured_output(TickerExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
                you have to extract the full official name of the company
                if in cas the name for company is incorrect correct it.
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

    fmp_api_key = os.getenv("FMP_API_KEY")
    CompanyName = result['company_name']
    url = f"https://financialmodelingprep.com/stable/search-name?query={CompanyName}&apikey={fmp_api_key}"
    response = requests.get(url)
    company_data = response.json()

    stk_list=[]
    for i in range(len(company_data)):
        exc=company_data[i]["exchange"]
        if exc in list_of_exc:
            stk_list.append(company_data[i])

    #no match
    if len(stk_list) == 0:
        return {"status": "not_found", "message": f"No ticker found for '{result['company_name']}'"}

    # Only ONE match
    if len(stk_list) == 1:
        q = stk_list[0]
        return {
            "status": "resolved",
            "ticker": q['symbol'],
            "name": q['name'],
            "exc": q['exchange']
        }

    # MULTIPLE matches — use fields already present in the search result (zero extra API calls)
    candidates = [
        {
            "ticker": q['symbol'],
            "name": q['name'],
            "exc": q['exchange']
        }
        for q in stk_list
    ]

    return {"status": "needs_confirmation", "candidates": candidates}

