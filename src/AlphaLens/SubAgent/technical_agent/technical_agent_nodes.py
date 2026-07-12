from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
import pandas_ta as ta
import pandas as pd
import numpy as np
import yfinance as yf
from AlphaLens.utils.yf_utils import yf_delay
from AlphaLens.SubAgent.technical_agent.technical_agent_prompt import REPORT_SYSTEM_PROMPT
from AlphaLens.SubAgent.technical_agent.technical_agent_states import TechnicalState
from AlphaLens.SubAgent.technical_agent.technical_agent_model import TechnicalReport
from AlphaLens.config.llms import get_report_writer_llm


def technical_report_node(state: TechnicalState)->TechnicalState:
    data=state["data"]

    clean_messages = [
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the collected technical data:\n\n{data}"),
    ]
    report_llm = get_report_writer_llm().with_structured_output(TechnicalReport)
    response = report_llm.invoke(clean_messages)
    
    return {"report": response.model_dump()}


def technical_Agent_node(state: TechnicalState)->TechnicalState:
    result = technical_agent_pipeline(state["ticker"])
    return {"data": result}


@traceable
def technical_agent_pipeline (ticker:str)->dict:
    period="1y"
    yf_delay()
    df=yf.Ticker(ticker).history(period=period)

    if len(df) < 200:
        raise ValueError(f"Not enough data — only {len(df)} rows")
    
    trend      = trend_node(df)        
    momentum   = momentum_node(df)
    volume     = volume_node(df)
    volatility = volatility_node(df)
    sub_worker={"Trend": trend, "Momentum": momentum, "Volume": volume, "Volatility": volatility}
    return sub_worker


#==========================================================================================

@traceable
def trend_node(df: pd.DataFrame) -> dict:
    if len(df) < 200:
        raise ValueError(f"Not enough data for SMA 200 — only {len(df)} rows")

    # SMA 50 and 200
    df['SMA_50']  = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)

    # EMA 12 and 26
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['EMA_26'] = ta.ema(df['Close'], length=26)

    # Cross signal logic
    df['above'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    df['CrossEventSignal'] = df['above'].diff()

    # Bollinger Bands 
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['BB_lower'] = bbands.iloc[:, 0]
    df['BB_mid']   = bbands.iloc[:, 1]
    df['BB_upper'] = bbands.iloc[:, 2]
    df['BB_pct']   = bbands.iloc[:, 4]

    # FIX: Look for crosses in the last 5 days to avoid missing macro events
    recent_crosses = df['CrossEventSignal'].iloc[-5:].values
    latest = df.iloc[-1]

    # Determine bb_signal
    if latest['BB_pct'] > 0.8:
        bb_signal = "overbought"
    elif latest['BB_pct'] < 0.2:
        bb_signal = "oversold"
    else:
        bb_signal = "neutral"

    # Determine cross signal
    if 1 in recent_crosses:
        cross_signal = "recent_golden_cross"
    elif -1 in recent_crosses:
        cross_signal = "recent_death_cross"
    elif latest['SMA_50'] > latest['SMA_200']:
        cross_signal = "bullish" 
    else:
        cross_signal = "bearish"

    return {
        "Name": "Trend",
        "sma_50":        round(latest['SMA_50'].item(), 2),
        "sma_200":       round(latest['SMA_200'].item(), 2),
        "ema_12":        round(latest['EMA_12'].item(), 2),
        "ema_26":        round(latest['EMA_26'].item(), 2),
        "cross_signal":  cross_signal,             
        "bb_lower":      round(latest['BB_lower'].item(), 2),
        "bb_upper":      round(latest['BB_upper'].item(), 2),          
        "bb_pct":        round(latest['BB_pct'].item(), 3),
        "bb_signal":     bb_signal,
        "current_price": round(latest['Close'].item(), 2),
    }

@traceable
def momentum_node(df: pd.DataFrame) -> dict:
    # RSI
    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']        = macd.iloc[:, 0]
    df['MACD_hist']   = macd.iloc[:, 1]
    df['MACD_signal'] = macd.iloc[:, 2]

    # Stoch
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    df['STOCH_k'] = stoch.iloc[:, 0]
    df['STOCH_d'] = stoch.iloc[:, 1]

    # ROC
    df['ROC_10'] = ta.roc(df['Close'], length=10)

    # CRITICAL FIX: Removed df.dropna(inplace=True). 
    # yfinance guarantees enough rows that the last row will be populated.
    latest = df.iloc[-1]

    # RSI interpretation
    rsi_val = round(float(latest['RSI_14']), 2)
    if rsi_val >= 70:
        rsi_signal = "overbought"
    elif rsi_val <= 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    # MACD interpretation
    macd_signal = "bullish" if latest['MACD_hist'] > 0 else "bearish"

    # Stoch interpretation
    stoch_k = round(float(latest['STOCH_k']), 2)
    if stoch_k >= 80:
        stoch_signal = "overbought"
    elif stoch_k <= 20:
        stoch_signal = "oversold"
    else:
        stoch_signal = "neutral"

    return {
        "Name": "Momentum",
        "rsi":          rsi_val,
        "rsi_signal":   rsi_signal,
        "macd":         round(float(latest['MACD']), 2),
        "macd_hist":    round(float(latest['MACD_hist']), 2),
        "macd_signal":  round(float(latest['MACD_signal']), 2),
        "macd_verdict": macd_signal,
        "stoch_k":      stoch_k,
        "stoch_d":      round(float(latest['STOCH_d']), 2),
        "stoch_signal": stoch_signal,
        "roc_10":       round(float(latest['ROC_10']), 2),
    }

@traceable
def volume_node(df: pd.DataFrame) -> dict:
    df['OBV'] = ta.obv(df['Close'], df['Volume'])

    # Volume vs Average Spike
    df['Vol_SMA_20'] = ta.sma(df['Volume'], length=20)
    df['Vol_Spike_Ratio'] = df['Volume'] / df['Vol_SMA_20']

    # CRITICAL FIX: Converted to a 20-day rolling VWAP instead of useless cumulative VWAP
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['typical_price'] * df['Volume']
    df['VWAP_20'] = df['VP'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # Accumulation / Distribution Line (ADL)
    df['ADL'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])

    latest = df.iloc[-1]

    # FIX: Adjusted volume spike tolerances to practical daily levels
    if latest['Vol_Spike_Ratio'] > 1.25:
        vol_signal = "high"
    elif latest['Vol_Spike_Ratio'] < 0.75:
        vol_signal = "low"
    else:
        vol_signal = "normal"
        
    obv_trend = "accumulation" if float(latest['OBV']) > float(df['OBV'].iloc[-20]) else "distribution"
    
    return {
        "Name": "Volume",
        "obv":             round(float(latest['OBV']), 2),
        "adl":             round(float(latest['ADL']), 2),
        "vol_spike_ratio": round(float(latest['Vol_Spike_Ratio']), 3),
        "vwap_20_lvl":     round(float(latest['VWAP_20']), 2),
        "vol_signal":      vol_signal,
        "obv_trend":       obv_trend
    }

@traceable
def volatility_node(df: pd.DataFrame) -> dict:
    # ATR - 14 Day
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Historical Volatility (30-day Annualized)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Hist_Vol_30'] = df['Log_Ret'].rolling(window=30).std() * np.sqrt(252)

    # 52-Week High / Low
    df['52W_High'] = df['Close'].rolling(window=252, min_periods=1).max()
    df['52W_Low'] = df['Close'].rolling(window=252, min_periods=1).min()

    latest = df.iloc[-1]

    vol_regime = "high" if latest['Hist_Vol_30'] > 0.3 else "low" if latest['Hist_Vol_30'] < 0.15 else "normal"
    pct_from_high = (latest['Close'] - latest['52W_High']) / latest['52W_High']
    pct_from_low  = (latest['Close'] - latest['52W_Low'])  / latest['52W_Low']

    return {
        "Name": "Volatility",
        "atr_14":            round(float(latest['ATR_14']), 2),
        "hist_vol":          round(float(latest['Hist_Vol_30']), 4),
        "high_52w":          round(float(latest['52W_High']), 2),
        "low_52w":           round(float(latest['52W_Low']), 2),
        "pct_from_52w_high": round(float(pct_from_high), 3), 
        "pct_from_52w_low":  round(float(pct_from_low), 3),
        "vol_regime":        vol_regime 
    }