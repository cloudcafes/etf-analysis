import os
import time
import io
import sys
import warnings
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import urllib3
from sqlalchemy import create_engine, inspect, text
from google import genai

warnings.simplefilter(action='ignore', category=FutureWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
# API KEYS & CONFIGURATION
# ==============================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_MODEL       = "gemini-2.5-flash-lite"
DB_NAME            = "sqlite:///etf_rotation.db"

ETF_LIST = [
    "GOLDBEES.NS","SILVERBEES.NS","NIFTYBEES.NS","BANKBEES.NS",
    "JUNIORBEES.NS","PSUBNKBEES.NS","ITBEES.NS","PHARMABEES.NS",
    "AUTOBEES.NS","CPSEETF.NS","ICICIB22.NS","HNGSNGBEES.NS",
    "MON100.NS","MAFANG.NS","MONQ50.NS","MIDCAPETF.NS",
    "SMALLCAP.NS","MIDSMALL.NS","MOM30IETF.NS",
    "METALIETF.NS","COMMOIETF.NS","FINIETF.NS",
    "ALPHAETF.NS","LOWVOLIETF.NS","QUAL30IETF.NS",
    "NV20IETF.NS","PVTBANIETF.NS","BANKIETF.NS",
    "OILIETF.NS","FMCGIETF.NS","INFRAIETF.NS",
    "CONSUMIETF.NS","HEALTHIETF.NS","EVIETF.NS",
    "ITETF.NS","PSUBNKIETF.NS","NEXT50IETF.NS",
    "NIF100BEES.NS","SENSEXIETF.NS"
]

THEME_MAP = {
    "MONQ50.NS":"US_TECH","MON100.NS":"US_TECH","MAFANG.NS":"US_TECH",
    "HNGSNGBEES.NS":"CHINA",
    "NIFTYBEES.NS":"LARGE_CAP","NIF100BEES.NS":"LARGE_CAP","SENSEXIETF.NS":"LARGE_CAP",
    "NEXT50IETF.NS":"LARGE_MID",
    "BANKBEES.NS":"BANK","BANKIETF.NS":"BANK","PVTBANIETF.NS":"BANK",
    "FINIETF.NS":"FINANCIALS","ICICIB22.NS":"PSU",
    "CPSEETF.NS":"PSU","PSUBNKBEES.NS":"PSU","PSUBNKIETF.NS":"PSU",
    "MIDCAPETF.NS":"MIDCAP","SMALLCAP.NS":"SMALLCAP","MIDSMALL.NS":"MID_SMALL",
    "MOM30IETF.NS":"MOMENTUM","ALPHAETF.NS":"ALPHA",
    "LOWVOLIETF.NS":"LOWVOL","QUAL30IETF.NS":"QUALITY","NV20IETF.NS":"VALUE",
    "AUTOBEES.NS":"AUTO","ITBEES.NS":"IT","ITETF.NS":"IT",
    "PHARMABEES.NS":"PHARMA","HEALTHIETF.NS":"HEALTH",
    "FMCGIETF.NS":"FMCG","CONSUMIETF.NS":"CONSUMPTION",
    "INFRAIETF.NS":"INFRA","OILIETF.NS":"ENERGY",
    "METALIETF.NS":"METAL","EVIETF.NS":"EV",
    "GOLDBEES.NS":"GOLD","SILVERBEES.NS":"SILVER",
    "COMMOIETF.NS":"COMMODITY",
    "JUNIORBEES.NS":"NEXT50"
}

# UPDATED: Added RSI and Momentum Score to schema
EXPECTED_COLUMNS = [
    'symbol', 'theme', 'price', '50DMA', 'cycle', 'rsi', 'mom_score',
    'ret_1m', 'ret_3m', 'ret_6m', 'volatility', 'avg_volume', 
    'pullback', 'stretch', 'rank', 'exhausted', 'action', 'date'
]

def get_db_engine():
    return create_engine(DB_NAME)

def check_credentials():
    if not GEMINI_API_KEY or "YOUR_" in GEMINI_API_KEY:
        print("❌ CRITICAL: Missing Gemini API Key.")
        sys.exit(1)
    if not TELEGRAM_BOT_TOKEN or "YOUR_" in TELEGRAM_BOT_TOKEN:
        print("❌ CRITICAL: Missing Telegram Bot Token.")
        sys.exit(1)
    if not TELEGRAM_CHAT_ID or "YOUR_" in TELEGRAM_CHAT_ID:
        print("❌ CRITICAL: Missing Telegram Chat ID.")
        sys.exit(1)

def send_telegram_chunk(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        resp = requests.post(url, json=payload, verify=False, timeout=15)
        return resp.status_code == 200
    except: 
        return False

def send_telegram_message(text: str) -> bool:
    clean = text.replace('**', '*').replace('##', '').replace('`', "'").replace('_', '-')
    max_len = 3900 
    
    if len(clean) <= max_len: 
        return send_telegram_chunk(clean)
    
    lines = clean.split('\n')
    current, part, success = "", 1, True
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            if current and not send_telegram_chunk(f"Part {part}:\n\n{current}"): success = False
            part += 1; current = line + "\n"
        else:
            current += line + "\n"
            
    if current and not send_telegram_chunk(f"Part {part}:\n\n{current.strip()}"): 
        success = False
    return success

def run_ai_analysis_and_notify(snapshot_data: str, max_retries=3):
    ai_prompt = (
        "You are a professional quantitative trading analyst.\n\n"
        "Analyze the ETF rotation report below and generate a READY-TO-SEND TELEGRAM MESSAGE.\n\n"
        "STRICT RULES:\n"
        "- Keep output short, sharp, and actionable\n"
        "- No explanations, no raw tables\n"
        "- Focus only on decisions and market insight\n"
        "- Use clean formatting with emojis\n"
        "- Avoid listing too many ETFs\n\n"
        "OUTPUT STRUCTURE:\n\n"
        "🔥 Market:\n"
        "- 1–2 lines describing overall market condition (breadth, trend, leadership)\n\n"
        "📈 BUY (Momentum Leaders):\n"
        "- List ONLY new buy signals. Mention their Momentum Score & RSI.\n"
        "- Format: ETF (Theme) – reason in 1 short line\n\n"
        "🟡 HOLD:\n"
        "- Mention only strongest leaders worth holding\n"
        "- Do NOT list everything\n\n"
        "❌ EXIT:\n"
        "- Summarize weakness (sectors/themes), NOT full list\n\n"
        "💰 Allocation:\n"
        "- Clear capital allocation guidance (%, cash if needed)\n\n"
        "⚠️ Risk:\n"
        "- One-line key risk\n\n"
        "TONE:\n"
        "- Professional, Confident, No hype\n\n"
        "[ETF REPORT DATA]\n"
        f"{snapshot_data.strip()}"
    )

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(max_retries):
        try:
            print(f"\n🤖 Analyzing market data with Gemini API (Attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(model=GEMINI_MODEL, contents=[ai_prompt])
            
            print("📲 Sending AI Analysis to Telegram...")
            if send_telegram_message(response.text):
                print("✅ Telegram Notification Sent Successfully!")
            else:
                print("❌ Failed to send Telegram message.")
            return  

        except Exception as e:
            error_msg = str(e)
            print(f"❌ AI API Error: {error_msg}")
            if "503" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5 
                    print(f"⏳ Server busy. Retrying in {wait_time} seconds...\n")
                    time.sleep(wait_time)
                else:
                    print("🚨 Max retries reached. Gemini API is too busy.")
            else:
                break

def fetch_data(symbol, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) >= 200:
                    df["symbol"] = symbol
                    return df
            return None 

        except Exception as e:
            error_msg = str(e)
            if "Rate limit" in error_msg or "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries - 1:
                    backoff = random.uniform(5, 10) * (attempt + 1)
                    print(f"\n  ⚠️ Rate limited on {symbol}. Backing off for {backoff:.1f}s...")
                    time.sleep(backoff)
                    continue
            print(f"\n  ❌ Failed to fetch {symbol}: {error_msg}")
            return None
    return None

def calculate_metrics(df):
    # Standard Moving Averages
    df["20DMA"] = df["Close"].rolling(20).mean()
    df["50DMA"] = df["Close"].rolling(50).mean()
    df["200DMA"] = df["Close"].rolling(200).mean()
    df["returns"] = df["Close"].pct_change()
    
    # Volatility and Volume
    df["vol_20"] = df["returns"].rolling(20).std()
    df["avg_vol_20"] = df["Volume"].rolling(20).mean()

    # Wilder's 14-Day RSI (Momentum Indicator)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    if pd.isna(latest["200DMA"]): return None

    close = float(latest["Close"])
    dma20 = float(latest["20DMA"])
    dma50 = float(latest["50DMA"])
    dma200 = float(latest["200DMA"])
    rsi = float(latest["RSI"])
    
    # Multi-Timeframe Returns
    ret_1m = (close / float(df["Close"].iloc[-22])) - 1
    ret_3m = (close / float(df["Close"].iloc[-66])) - 1
    ret_6m = (close / float(df["Close"].iloc[-130])) - 1

    # Composite Momentum Score (Weights: 1M=30%, 3M=50%, 6M=20%)
    mom_score = (ret_1m * 0.30) + (ret_3m * 0.50) + (ret_6m * 0.20)

    # Trend Cycle Classification
    if close > dma50 and dma50 > dma200: cycle = "UPTREND"
    elif close < dma50 and dma50 < dma200: cycle = "DOWNTREND"
    else: cycle = "SIDEWAYS"

    return {
        "symbol": str(latest["symbol"]),
        "theme": THEME_MAP.get(str(latest["symbol"]), "OTHER"),
        "price": round(close, 2),
        "50DMA": round(dma50, 2),
        "cycle": cycle,
        "rsi": round(rsi, 2),
        "mom_score": round(mom_score, 4),
        "ret_1m": round(ret_1m, 4),
        "ret_3m": round(ret_3m, 4),
        "ret_6m": round(ret_6m, 4),
        "volatility": round(float(latest["vol_20"]), 4),
        "avg_volume": int(latest["avg_vol_20"]),
        "pullback": round(abs(close - dma20) / dma20, 4),
        "stretch": round((close - dma50) / dma50, 4)
    }

def process_all():
    results = []
    total = len(ETF_LIST)
    print(f"🚀 Fetching data for {total} ETFs...")

    for idx, etf in enumerate(ETF_LIST, 1):
        sys.stdout.write(f"\r[{idx}/{total}] Fetching {etf}...".ljust(50))
        sys.stdout.flush()

        df = fetch_data(etf)
        if df is not None:
            m = calculate_metrics(df)
            if m: results.append(m)

        if idx < total:
            time.sleep(random.uniform(1.0, 2.0))

    print("\n✅ Data fetch complete.\n")

    if not results:
        print("⚠️ Warning: No data was retrieved. Yahoo Finance is blocking all requests.")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    
    # RANKING: Now based on True Composite Momentum, not just 3-month returns
    result_df["rank"] = result_df["mom_score"].rank(ascending=False, method="min").astype(int)
    return result_df

def detect_exhaustion(row):
    # Advanced exhaustion logic using RSI and Over-extension
    signals = 0
    if row["rsi"] > 75: signals += 2             # Severely Overbought
    if row["stretch"] > 0.08: signals += 1       # Price is extended >8% above 50DMA
    if row["volatility"] > 0.035: signals += 1   # Price action becoming erratic
    if row["ret_1m"] < 0: signals += 1           # Immediate momentum has stalled
    return signals >= 2

def pick_unique_themes(df):
    selected = []
    used = set()
    for _, row in df.sort_values("rank").iterrows():
        if row["theme"] not in used:
            selected.append(row)
            used.add(row["theme"])
        if len(selected) == 3: break
    return pd.DataFrame(selected)

def classify(df):
    df = df.copy()
    df["exhausted"] = df.apply(detect_exhaustion, axis=1)
    df["action"] = "AVOID"
    
    mask_sell = (df["price"] < df["50DMA"]) | ((df["cycle"] == "UPTREND") & df["exhausted"])
    df.loc[mask_sell, "action"] = "SELL"
    
    mask_hold = (df["cycle"] == "UPTREND") & (~df["exhausted"]) & (df["price"] >= df["50DMA"])
    df.loc[mask_hold, "action"] = "HOLD"
    
    # ADVANCED BUY LOGIC: Volume > 50k, Low Volatility, RSI Sweet Spot (55-70) indicating active strength
    mask_buy_candidates = (
        mask_hold & 
        (df["avg_volume"] > 50000) & 
        (df["volatility"] < 0.035) & 
        (df["rsi"] >= 55) & 
        (df["rsi"] <= 72) &
        (df["mom_score"] > 0) # Must have positive aggregate momentum
    )
    
    candidates = df[mask_buy_candidates].copy()
    invest = pick_unique_themes(candidates)
    
    if not invest.empty:
        # Require a modest 20DMA pullback entry (don't chase)
        invest = invest[invest["pullback"] < 0.04]
        df.loc[df["symbol"].isin(invest["symbol"]), "action"] = "BUY"
        
    return df

def validate_db(engine):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        print(f"❌ DB Integrity Check Failed: {e}")
        print("🚨 Database corrupted or inaccessible. Hard failing pipeline.")
        sys.exit(1)

def get_previous_states(engine):
    try:
        insp = inspect(engine)
        if not insp.has_table("etf_metrics"): 
            return {}, "Never"
            
        with engine.connect() as conn:
            max_date = conn.execute(text("SELECT MAX(date) FROM etf_metrics")).scalar()
            
        if not max_date: 
            return {}, "Never"
            
        query_data = f"SELECT DISTINCT symbol, action FROM etf_metrics WHERE date = '{max_date}'"
        prev_df = pd.read_sql(query_data, engine)
        return dict(zip(prev_df['symbol'], prev_df['action'])), max_date
    except Exception as e:
        print(f"⚠️ Error fetching previous states: {e}")
        return {}, "Error"

def validate_schema(df):
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ Schema error! Missing columns: {missing_cols}")
        sys.exit(1)
    return df[EXPECTED_COLUMNS]

def remove_existing_rows(df, today_date, engine):
    try:
        insp = inspect(engine)
        if not insp.has_table("etf_metrics"):
            return df
            
        query = f"SELECT symbol FROM etf_metrics WHERE date = '{today_date}'"
        existing_df = pd.read_sql(query, engine)
        existing_symbols = existing_df["symbol"].tolist()
        
        if existing_symbols:
            filtered_df = df[~df["symbol"].isin(existing_symbols)]
            print(f"🔄 Deduplication: Filtered out {len(df) - len(filtered_df)} rows already existing for today.")
            return filtered_df
    except Exception as e:
        print(f"⚠️ Error during deduplication check: {e}")
        
    return df

def save_to_db(df, engine):
    if df.empty:
        print("ℹ️ No new distinct records to insert for today.")
        return
        
    try:
        print("💾 Initializing secure database transaction...")
        with engine.begin() as conn: 
            df.to_sql("etf_metrics", conn, if_exists="append", index=False)
        print("✅ DB Transaction Committed Successfully.")
    except Exception as e:
        print(f"❌ DB Write Error: {e}")
        print("🚨 Transaction Rolled Back. Hard failing pipeline.")
        sys.exit(1)

def print_clean_table(df, columns, headers):
    if df.empty:
        print("  --> No ETFs currently match this criteria.")
        return
    display_df = df.copy()
    for col in ["mom_score", "ret_1m", "ret_3m", "ret_6m", "pullback", "stretch", "volatility"]:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
            
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(display_df[columns].rename(columns=dict(zip(columns, headers))).to_string(index=False))

def main():
    check_credentials()
    engine = get_db_engine()

    try:
        ist_tz = ZoneInfo('Asia/Kolkata')
        now_ist = datetime.now(ist_tz)
        today_str = now_ist.strftime("%B %d, %Y")
        today_date = now_ist.date()
        
        print("=" * 80)
        print(f" ETF QUANTITATIVE INVESTMENT GUIDE | {today_str}".center(80))
        print("=" * 80)

        validate_db(engine)

        df = process_all()
        if df.empty:
            print("❌ Critical: Market data frame is empty. Failing pipeline.")
            sys.exit(1)

        df = classify(df)
        
        prev_states, last_run_date = get_previous_states(engine)
        df["prev_action"] = df["symbol"].map(prev_states).fillna("NONE")
        
        save_df = df.drop(columns=["prev_action"])
        save_df["date"] = today_date
        
        save_df = validate_schema(save_df)
        save_df = remove_existing_rows(save_df, today_date, engine) 
        save_to_db(save_df, engine)

        print(f"\nComparing today's data against last run: {last_run_date}\n")

        new_buys = df[(df["action"] == "BUY") & (df["prev_action"] != "BUY")].sort_values("rank")
        maintained_buys = df[(df["action"] == "BUY") & (df["prev_action"] == "BUY")].sort_values("rank")
        sell_alerts = df[(df["prev_action"].isin(["BUY", "HOLD"])) & (df["action"].isin(["SELL", "AVOID"]))].sort_values("rank")
        hold_df = df[df["action"] == "HOLD"].sort_values("rank")

        output_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer 
        
        print(f"REPORT DATE: {today_str}\n")
        
        if not sell_alerts.empty:
            print("🚨 ACTION REQUIRED: DOWNGRADED TO SELL (Exit these positions)")
            print("-" * 80)
            print_clean_table(sell_alerts, ["symbol", "prev_action", "cycle", "rsi", "50DMA", "exhausted"], ["ETF", "Previous Status", "Current Cycle", "RSI", "50 DMA", "Is Exhausted?"])
            print()

        print("🟢 NEW BUYS (Triggered Today)")
        print("-" * 80)
        print_clean_table(new_buys, ["symbol", "theme", "rank", "mom_score", "rsi", "pullback"], ["ETF", "Theme", "Rank", "Mom Score", "RSI", "Pullback"])
        
        print("\n🔵 MAINTAINED BUYS (Already recommended on previous runs)")
        print("-" * 80)
        print_clean_table(maintained_buys, ["symbol", "theme", "rank", "mom_score", "rsi", "stretch"], ["ETF", "Theme", "Rank", "Mom Score", "RSI", "Stretch"])

        print("\n🟡 HOLD (Healthy Uptrends, keep if you already own)")
        print("-" * 80)
        print_clean_table(hold_df, ["symbol", "rank", "mom_score", "rsi", "50DMA"], ["ETF", "Rank", "Mom Score", "RSI", "50 DMA"])

        print("\n💰 RECOMMENDED PORTFOLIO ALLOCATION (NEW CAPITAL)")
        print("-" * 80)
        buy_df = df[df["action"] == "BUY"]
        if not buy_df.empty:
            weight = round(100 / len(buy_df), 2)
            for _, row in buy_df.iterrows():
                print(f"  • {row['symbol']:<15} : {weight}% ({row['theme']})")
        else:
            print("  --> NO NEW BUYS MEETING CRITERIA TODAY. HOLD CASH.")
            
        print("\n" + "=" * 80)
        print(" 📊 MASTER ETF UNIVERSE METRICS (ALL DATA)".center(80))
        print("=" * 80)
        
        master_df = df.sort_values("rank")
        print_clean_table(master_df, ["symbol", "rank", "action", "cycle", "rsi", "mom_score", "ret_3m", "volatility", "pullback", "stretch", "exhausted"], ["ETF", "Rank", "Status", "Cycle", "RSI", "Mom Score", "3M Ret", "Vol", "Pullback", "Stretch", "Exhausted"])
        print("\n" + "=" * 80 + "\n")

        sys.stdout = original_stdout 
        captured_report = output_buffer.getvalue()

        print(captured_report)
        run_ai_analysis_and_notify(captured_report)
        
    except Exception as e:
        print(f"\n❌ CRITICAL PIPELINE FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
