import yfinance as yf
from AlphaLens.utils.yf_utils import yf_delay
from langchain_core.tools import tool
import logging
import requests
import os
import pandas as pd

@tool
def news_yfinance(ticker:str)->list:
    """
    Fetches and formats the latest news articles for a given stock ticker using the yfinance library.

    This function retrieves the raw news data for the specified ticker and extracts 
    key information into a clean, standardized dictionary format, safely handling 
    missing fields to prevent KeyErrors.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA').

    Returns:
        list: A list of dictionaries, where each dictionary represents a news article 
              and contains the following keys:
              - 'title' (str): The headline of the article.
              - 'summary' (str): A brief summary of the article.
              - 'published_at' (str): The publication date and time.
              - 'source' (str): The publisher or provider of the news.
              - 'url' (str): The canonical URL to the full article.
    """
    yf_delay()
    stock_news = yf.Ticker(ticker).news
    response = [] 
    for item in stock_news:
        content = item.get('content', {}) 
        clean_article = {
        "title": content.get('title', ''),
        "summary": content.get('summary', ''),
        "published_at": content.get('pubDate', ''),
        "source": content.get('provider', {}).get('displayName', 'Unknown'),
        "url": content.get('canonicalUrl', {}).get('url', '')
        }
        response.append(clean_article)

    return response

@tool
def earning_call(ticker: str)-> dict:
    """
    Fetches the latest earnings call data for a given stock ticker using the ROIC.ai API.
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT', 'BRK').
    Returns:
        dict: A dictionary containing the parsed JSON response from the API, 
              which includes the latest earnings call details.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the network request.
        ValueError: If the response payload is not valid JSON.
    """
    try:
        try:
            api_key = os.environ["ROC_AI_API_KEY"]
        except KeyError:
            return {"error": "ROC_AI_API_KEY environment variable is not set."}
        if "." in ticker: 
            ticker=ticker.split(".")[0]
        response = requests.get(f"https://api.roic.ai/v2/company/earnings-calls/latest/{ticker}?apikey={api_key}")
        data = response.json()
        if data:
            return data
        return {"error": "Empty response from ROIC API"}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_earnings_history(ticker: str) -> dict:
    """Fetch earnings beat/miss history and EPS surprises. Free via yfinance.
    Consistent beats = bullish signal. Misses = bearish red flag."""
    try:
        yf_delay()
        stock    = yf.Ticker(ticker)
        yf_logger = logging.getLogger("yfinance")
        previous_level = yf_logger.level
        yf_logger.setLevel(logging.ERROR)
        try:
            earnings = stock.earnings_dates
        finally:
            yf_logger.setLevel(previous_level)
 
        if earnings is None or earnings.empty:
            return {
                "ticker": ticker,
                "earnings_history": [],
                "warning": "No earnings dates returned by yfinance.",
                "source": "yfinance",
            }
 
        history = []
        def clean_number(value):
            return None if pd.isna(value) else float(value)

        for date, row in earnings.head(8).iterrows():
            eps_est    = row.get("EPS Estimate")
            eps_actual = row.get("Reported EPS")
            surprise   = row.get("Surprise(%)")
 
            history.append({
                "date":        str(date.date()) if hasattr(date, "date") else str(date),
                "eps_estimate": clean_number(eps_est),
                "eps_actual":   clean_number(eps_actual),
                "surprise_pct": clean_number(surprise),
                "beat":         bool(surprise > 0) if not pd.isna(surprise) else None,
            })
 
        return {"ticker": ticker, "earnings_history": history, "source": "yfinance"}
 
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_insider_transactions(ticker: str) -> dict:
    """Fetch recent insider buying and selling activity. Free via yfinance.
    Heavy insider selling = bearish signal. Buying = bullish signal."""
    try:
        yf_delay()
        stock    = yf.Ticker(ticker)
        insider  = stock.insider_transactions
 
        if insider.empty:
            return {"error": "No insider data found"}
 
        transactions = []
        for _, row in insider.head(10).iterrows():
            transactions.append({
                "insider":    row.get("Insider"),
                "relation":   row.get("Relation"),   
                "date":       str(row.get("Start Date", "")),
                "transaction":row.get("Transaction"), 
                "shares":     row.get("Shares"),
                "value":      row.get("Value"),
            })
 
        return {"ticker": ticker, "insider_transactions": transactions, "source": "yfinance"}
 
    except Exception as e:
        return {"error": str(e)}
