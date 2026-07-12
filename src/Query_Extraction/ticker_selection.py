# only used when running in cli ====================================

from Query_Extraction.query_extraction import ticker_extraction_tool

def ticker_resolver(query:str):
    res=ticker_extraction_tool(query)

    if res["status"] == "not_found" :
        return res["message"]
    
    if res["status"] == "resolved" :
        return res["ticker"]
    
    if res["status"] == "needs_confirmation" :
        print(f"I have found multiple matches which one are you talking about:")
        for i,c in enumerate(res["candidates"],1):
            print(f"{i}. {c['name']} ({c['ticker']}) — {c['exchange']}")
        choice = input("Enter number (or 'none' to cancel): ").strip() 
        if not choice.isdigit() or choice.lower() == "none":
            userquery = input("\nSorry for the wrong Company Matches\nTry again with more Specific Company Name\nEnter the Company Name:")
            return ticker_resolver(userquery)
        
        if int(choice) > len(res["candidates"]):
            selected = res["candidates"][-1]
            return selected["ticker"]
        
        selected = res["candidates"][int(choice) - 1]
        return selected["ticker"]