import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from AlphaLens.utils.yf_utils import yf_delay


def _safe_get(df, row, col):
    if row not in df.index:
        return None

    value = df.loc[row, col]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@tool
def get_income_statement(ticker: str) -> dict:
    """Fetch revenue, EPS, net income, and margins for the last 3 years."""
    try:
        yf_delay()
        stock = yf.Ticker(ticker)
        income = stock.financials  # annual income statement
 
        if income.empty:
            return {"error": "No income statement data found"}
 
        result = {}
        for col in income.columns[:3]:  # last 3 years
            year = str(col.year)
            result[year] = {
                "revenue":     _safe_get(income, "Total Revenue", col),
                "gross_profit":_safe_get(income, "Gross Profit", col),
                "net_income":  _safe_get(income, "Net Income", col),
                "ebitda":      _safe_get(income, "EBITDA", col),
                "eps":         _safe_get(income, "Diluted EPS", col),
            }
 
        return {"income_statement": result, "source": "yfinance"}
 
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_balance_sheet(ticker: str) -> dict:
    """Fetch assets, liabilities, debt, and equity from the balance sheet."""
    try:
        yf_delay()
        stock = yf.Ticker(ticker)
        bs = stock.balance_sheet
 
        if bs.empty:
            return {"error": f"No balance sheet data found for {ticker}"}
 
        col = bs.columns[0]  # most recent year
        return {
            "ticker": ticker,
            "balance_sheet": {
                "total_assets":       _safe_get(bs, "Total Assets", col),
                "total_liabilities":  _safe_get(bs, "Total Liabilities Net Minority Interest", col),
                "total_equity":       _safe_get(bs, "Stockholders Equity", col),
                "total_debt":         _safe_get(bs, "Total Debt", col),
                "cash":               _safe_get(bs, "Cash And Cash Equivalents", col),
            },
            "source": "yfinance"
        }
 
    except Exception as e:
        return {"error": str(e)}


@tool
def get_cash_flow(ticker: str) -> dict:
    """Fetch operating cash flow, free cash flow, and capex."""
    try:
        yf_delay()
        stock = yf.Ticker(ticker)
        cf = stock.cashflow
 
        if cf.empty:
            return {"error": "No cash flow data found"}
 
        result = {}
        for col in cf.columns[:3]:  # last 3 years
            year = str(col.year)
            op_cf   = _safe_get(cf, "Operating Cash Flow", col)
            capex   = _safe_get(cf, "Capital Expenditure", col)
            fcf     = (op_cf + capex) if (op_cf is not None and capex is not None) else None  # capex is negative
 
            result[year] = {
                "operating_cash_flow": op_cf,
                "capex":               capex,
                "free_cash_flow":      fcf,
            }
 
        return {"ticker": ticker, "cash_flow": result, "source": "yfinance"}
 
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_key_ratios(ticker: str) -> dict:
    """Fetch valuation and efficiency ratios: PE, PB, PS, ROE (calculated manually), debt-to-equity (calculated manually)."""
    try:
        yf_delay()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        try:
            bs = stock.balance_sheet
            fin = stock.financials
            
            if not bs.empty and not fin.empty:
                recent_bs_col = bs.columns[0]
                recent_fin_col = fin.columns[0]
                
                net_income = _safe_get(fin, "Net Income", recent_fin_col)
                equity = _safe_get(bs, "Stockholders Equity", recent_bs_col)
                
                # Debt fallback logic (Total Debt sometimes isn't listed as a single line)
                total_debt = _safe_get(bs, "Total Debt", recent_bs_col)
                if total_debt is None:
                    lt_debt = _safe_get(bs, "Long Term Debt", recent_bs_col) or 0
                    st_debt = _safe_get(bs, "Current Debt", recent_bs_col) or 0
                    if (lt_debt + st_debt) > 0:
                        total_debt = lt_debt + st_debt

                # Calculate true values
                real_roe = (net_income / equity) if (net_income and equity) else "Not available"
                real_dte = (total_debt / equity) if (total_debt is not None and equity) else "Not available"
            else:
                real_roe, real_dte = "Not available", "Not available"
                
        except Exception:
            real_roe, real_dte = "Not available (Calc error)", "Not available (Calc error)"


        return {
            "ticker": ticker,
            "ratios": {
                "pe_ratio":         info.get("trailingPE", "Not available"),
                "forward_pe":       info.get("forwardPE", "Not available"),
                "price_to_book":    info.get("priceToBook", "Not available"),
                "price_to_sales":   info.get("priceToSalesTrailing12Months", "Not available"),
                "debt_to_equity":   real_dte,
                "roe":              real_roe,  
                "roa":              info.get("returnOnAssets", "Not available"),
                "gross_margin":     info.get("grossMargins", "Not available"),
                "operating_margin": info.get("operatingMargins", "Not available"),
                "profit_margin":    info.get("profitMargins", "Not available"),
                "current_ratio":    info.get("currentRatio", "Not available"),
                "revenue_growth":   info.get("revenueGrowth", "Not available"),
                "earnings_growth":  info.get("earningsGrowth", "Not available"),
            },
            "source": "yfinance (hybrid calculation)"
        }

    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_company_info(ticker: str) -> dict:
    """Fetch company name, sector, industry, market cap, and analyst targets."""
    try:
        yf_delay()
        stock = yf.Ticker(ticker)
        info  = stock.info
        description = info.get("longBusinessSummary") or "Not available"
 
        return {
            "ticker": ticker,
            "company_name" :        info.get('longName') or info.get('shortName') ,
            "Exchange":             info.get('exchange', 'Unknown'),
            "company": {
                "name":              info.get("longName", "Not available"),
                "sector":            info.get("sector", "Not available"),
                "industry":          info.get("industry", "Not available"),
                "market_cap":        info.get("marketCap", "Not available"),
                "employees":         info.get("fullTimeEmployees", "Not available"),
                "description":       description[:400],
            },
            "analyst": {
                "target_price":      info.get("targetMeanPrice", "Not available"),
                "recommendation":    info.get("recommendationKey", "Not available"),
                "num_analysts":      info.get("numberOfAnalystOpinions", "Not available"),
            },
            "source": "yfinance"
        }
 
    except Exception as e:
        return {"error": str(e)}
    
