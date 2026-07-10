import json
import yfinance as yf
from AlphaLens.utils.yf_utils import yf_delay
import numpy as np
from AlphaLens.SubAgent.risk_agent.risk_agent_prompt import RISK_REPORT_SYSTEM_PROMPT
from langchain_core.messages import HumanMessage,SystemMessage
from AlphaLens.SubAgent.risk_agent.risk_agent_model import RiskReport
from langsmith import traceable
from AlphaLens.config.llms import get_report_writer_llm
from AlphaLens.SubAgent.risk_agent.risk_agent_state import RiskState


def risk_agent_node(state: RiskState) -> RiskState:
    ticker = state.get("ticker")
    liquidity = liquidity_risk(ticker)
    business  = Business_risk(ticker)
    financial = Financial_risk(ticker)
    market    = market_risk(ticker)

    computed = {
        "Liquidity Risk": liquidity,
        "Business Risk":  business,
        "Financial Risk": financial,
        "Market Risk":    market,
    }

    return {"computed_risks": computed}



def risk_report_node(state: RiskState) -> RiskState:
    computed = state.get("computed_risks")

    messages = [
        SystemMessage(content=RISK_REPORT_SYSTEM_PROMPT),
        HumanMessage(content=f"Here are the computed risk metrics:\n{json.dumps(computed, indent=2)}"),
    ]

    report_llm = get_report_writer_llm().with_structured_output(RiskReport)

    response = report_llm.invoke(messages)

    return {"report":{'report':response.model_dump(),
                    'computed_risks': computed}}

#=====================================================================================================
@traceable
def liquidity_risk(ticker: str) -> dict:
    yf_delay()
    info= yf.Ticker(ticker).info

    avg_daily_volume=  info.get("averageVolume")
    float_shares=      info.get("floatShares")
    current_ratio=     info.get("currentRatio")

    return {"avg_daily_volume":  avg_daily_volume,
            "float_shares":      float_shares,
            "current_ratio":    current_ratio,}

@traceable
def Business_risk(ticker:str)-> dict:
    yf_delay()
    symbol= yf.Ticker(ticker)
    info = symbol.info 
    financials = symbol.financials
    
    sector=info.get("sector")
    
    try:
        rev_0 = financials.loc['Total Revenue'].iloc[0]
        rev_1 = financials.loc['Total Revenue'].iloc[1]
        
        growth="Growing" if rev_0 > rev_1 else "Shrinking"
        return {
            "sector":sector,
            "revenue_growth":growth,
        }
    
    except Exception as e:
        return {"sector": sector, "revenue_growth": "Unknown", "error": str(e)}
    
@traceable
def Financial_risk (ticker:str)-> dict:
    yf_delay()
    symbol= yf.Ticker(ticker)
    info = symbol.info 
    financials = symbol.financials
    cashflow = symbol.cashflow
    balance = symbol.balance_sheet
    
    debt_to_equity= info.get("debtToEquity")
    try:
        ebit=financials.loc["EBIT"].iloc[0]  
        interest_expense=financials.loc["Interest Expense"].iloc[0]
        try:
            ebit = float(ebit)
            interest_expense = float(interest_expense)
        except (TypeError, ValueError):
            ebit = None
            interest_expense = None

        if (ebit == None or interest_expense == None or interest_expense == 0):
            coverage_ratio = None
        else:
            coverage_ratio = ebit / abs(interest_expense)

        ocf   = cashflow.loc["Operating Cash Flow"].iloc[0]
        capex = cashflow.loc["Capital Expenditure"].iloc[0]

        fcf = ocf + capex
        
        # The Brutal Diagnosis
        if fcf < 0:
            if ocf > 0:
                diagnosis = "Negative FCF (Hyper-Growth Burn): Core ops make money, but heavy CapEx is draining cash."
            else:
                diagnosis = "Negative FCF (Lethal Burn): Core ops are bleeding cash. Danger zone."
        else:
            diagnosis = "Positive FCF: Generating surplus cash."

        current_assets = balance.loc[ 'Current Assets'].iloc[0]
        current_liabilities = balance.loc[ 'Current Liabilities'].iloc[0]
        cash= balance.loc[ 'Cash And Cash Equivalents'].iloc[0]
        current_debt=balance.loc[ 'Total Debt'].iloc[0]

        operating_nwc = (current_assets - cash) - (current_liabilities - current_debt)

        operating_nwc_status="Healthy" if operating_nwc > 0 else "Warning (Negative)"
        
        return{
            "debt_to_equity" : debt_to_equity,
            "interest_coverage": round(float(coverage_ratio),2) if coverage_ratio is not None else None,
            "fcf_negative":      diagnosis,
            "Operating_Net_Working_Capital":round(float(operating_nwc),2),
            "Operating_Net_Working_Capital_status":operating_nwc_status,
            "Total Debt":round(float(current_debt),2),
        }
    except Exception as e:
        return {"debt_to_equity": debt_to_equity, "error": str(e)}

@traceable
def market_risk(ticker: str) -> dict:
    yf_delay()
    symbol= yf.Ticker(ticker)
    info = symbol.info
    df_hist=symbol.history(period="1y")

    beta=info.get("beta")

    try:
        beta_value = float(beta)
    except (TypeError, ValueError):
        beta_value = None

    if beta_value is None:
        beta_signal = "Unknown"
    else:
        beta_signal = (
            "very high"  if beta_value > 1.5  else
            "high"       if beta_value > 1.0  else
            "moderate"   if beta_value > 0.5  else
            "low"
        )

    if df_hist.empty or "Close" not in df_hist:
        return{
            "beta":             beta,
            "beta_signal":      beta_signal,
            "var_95":           None,
            "var_99":           None,
            "max_drawdown":     None,
            "drawdown_signal":  "Unknown",
            "error":            "No historical close data available",
        }

    returns = df_hist["Close"].pct_change().dropna()

    if returns.empty:
        return{
            "beta":             beta,
            "beta_signal":      beta_signal,
            "var_95":           None,
            "var_99":           None,
            "max_drawdown":     None,
            "drawdown_signal":  "Unknown",
            "error":            "Not enough historical data to calculate returns",
        }

    # sort returns, take the worst (1 - confidence) percentile
    var_95 = np.percentile(returns, (1 - 0.95) * 100)
    var_99 = np.percentile(returns, (1 - 0.99) * 100)
    
    close  = df_hist["Close"]
    peak   = close.cummax()      
    trough = (close - peak) / peak 
    max_drawdown=trough.min()

    drawdown_signal = (
        "severe"   if max_drawdown < -0.30 else
        "moderate" if max_drawdown < -0.15 else
        "low"
    )

    return{
        "beta":             beta,
        "beta_signal":      beta_signal,
        "var_95":           round(float(var_95), 4),       
        "var_99":           round(float(var_99), 4),        
        "max_drawdown":     round(float(max_drawdown), 4),  
        "drawdown_signal":  drawdown_signal,  
    
    }
