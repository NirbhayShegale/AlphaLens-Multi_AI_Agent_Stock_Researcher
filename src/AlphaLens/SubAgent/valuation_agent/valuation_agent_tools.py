from langchain_community.tools import tool
import statistics
import requests
import pandas as pd
import yfinance as yf
from AlphaLens.utils.yf_utils import yf_delay


@tool
def get_company_info(ticker: str) -> dict:
    """Get current price, EPS, book value, and sector for valuation."""
    yf_delay()
    info = yf.Ticker(ticker).info
    return {
        "current_price": (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        ),
        "sector":     info.get("sector"),
        "eps":        info.get("trailingEps"),
        "book_value": info.get("bookValue"),
        "market_cap": info.get("marketCap"),
        "forward_pe": info.get("forwardPE"),
        "beta":       info.get("beta"),
    }
 
 
@tool
def get_dcf_inputs(ticker: str) -> dict:
    """Fetch FCF history, revenue growth for DCF model."""
    yf_delay()
    stock    = yf.Ticker(ticker)
    info     = stock.info
    cashflow = stock.cashflow

    fcf_history = {}
    cols = sorted(cashflow.columns, reverse=True)[:3] if len(cashflow.columns) else []
    for col in cols:
        op_cf = cashflow.loc["Operating Cash Flow", col] if "Operating Cash Flow" in cashflow.index else None
        capex = cashflow.loc["Capital Expenditure", col] if "Capital Expenditure" in cashflow.index else None
 
        if op_cf is None or capex is None:
            continue
        if pd.isna(op_cf) or pd.isna(capex):
            continue
 
        fcf_history[str(col.year)] = float(op_cf + capex)
 
    rev_growth = info.get("revenueGrowth")
    rev_growth = rev_growth if rev_growth is not None else 0.08
 
    return {
        "fcf_history":        fcf_history,
        "revenue_growth":     round(rev_growth, 4),
        "ebitda":             info.get("ebitda"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "total_debt":         info.get("totalDebt"),
        "cash":               info.get("totalCash"),
    }

@tool
def get_peer_multiples(peer_company: list) -> dict:
    """Fetch PE, PB, EV/EBITDA for a list of peer company names and compute sector medians."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
 
    peers = {}
    not_found = []
 
    for company in peer_company:
        yf_delay()
        search = yf.Search(company, max_results=1, session=session)
        if not search.quotes:
            not_found.append(company)
            continue
 
        detected_ticker = search.quotes[0]["symbol"]
        yf_delay()
        info   = yf.Ticker(detected_ticker).info
        ev     = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
 
        peers[company] = {
            "company_ticker": detected_ticker,
            "pe_ratio":  info.get("trailingPE"),
            "pb_ratio":  info.get("priceToBook"),
            "ev_ebitda": round(ev / ebitda, 2) if ev and ebitda else None,
        }
 
    pe_list = [p["pe_ratio"]  for p in peers.values() if p["pe_ratio"]]
    pb_list = [p["pb_ratio"]  for p in peers.values() if p["pb_ratio"]]
    ev_list = [p["ev_ebitda"] for p in peers.values() if p["ev_ebitda"]]
 
    median_pe        = statistics.median(pe_list) if pe_list else None
    median_pb        = statistics.median(pb_list) if pb_list else None
    median_ev_ebitda = statistics.median(ev_list) if ev_list else None
 
    return {
        "peers":            peers,
        "peers_not_found":  not_found,
        "median_pe":        median_pe,
        "median_pb":        median_pb,
        "median_ev_ebitda": median_ev_ebitda,
    }
