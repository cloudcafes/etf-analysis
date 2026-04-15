import os
import time
import io
import sys
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import urllib3
from sqlalchemy import create_engine, inspect
from google import genai

# Suppress warnings for clean console output
warnings.simplefilter(action='ignore', category=FutureWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
# API KEYS & CONFIGURATION
# ==============================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"

DB_NAME = "sqlite:///etf_rotation.db"

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

engine = create_engine(DB_NAME)

# ==============================
# TELEGRAM & AI LOGIC
# ==============================

def send_telegram_chunk(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        resp = requests.post(url, json=payload, verify=False, timeout=15)
        return resp.status_code == 200
    except: 
        return False

def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or "YOUR_" in TELEGRAM_BOT_TOKEN: 
        return False
        
    clean = text.replace('**', '*').replace('##', '').replace('`', "'").replace('_', '-')
    max_len = 4000
    
    if len(clean) <= max_len: 
        return send_telegram_chunk(clean)
    
    lines = clean.split('\n')
    current, part, success = "", 1, True
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            if current and not send_telegram_chunk(f"Part {part}:\n\n{current}"): success = False
            part += 1; current = line
        else:
            current = (current + "\n" + line) if current else line
            
    if current and not send_telegram_chunk(f"Part {part}:\n\n{current}"): 
        success = False
    return success

def run_ai_analysis_and_notify(snapshot_data: str, max_retries=3):
    """Sends the data to Gemini with automatic retries, then forwards to Telegram."""
    if not GEMINI_API_KEY or "YOUR_" in GEMINI_API_KEY: 
        print("⚠️ Missing Gemini API Key. Skipping AI Analysis.")
        return

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
        "📈 BUY:\n"
        "- List ONLY new buy signals\n"
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
        "- Professional\n"
        "- Confident\n"
        "- No hype, no generic advice\n\n"
        "[ETF REPORT DATA]\n"
        f"{snapshot_data.strip()}"
    )

    for attempt in range(max_retries):
        try:
            print(f"🤖 Analyzing market data with Gemini API (Attempt {attempt + 1}/{max_retries})...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=[ai_prompt])
            ai_response_text = response.text
            
            print("📲 Sending AI Analysis to Telegram...")
            if send_telegram_message(ai_response_text):
                print("✅ Telegram Notification Sent Successfully!")
            else:
                print("❌ Failed to send Telegram message.")
                
            return  # Exit function successfully if we make it here

        except Exception as e:
            error_msg = str(e)
            print(f"❌ AI API Error: {error_msg}")
            
            # If it's a server overload (503) or rate limit (429), try again
            if "503" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5 
                    print(f"⏳ Server busy. Retrying in {wait_time} seconds...\n")
                    time.sleep(wait_time)
                else:
                    print("🚨 Max retries reached. Gemini API is too busy to answer right now.")
            else:
                break

# ==============================
# BATCH FETCH & METRICS
# ==============================

def calculate_metrics(df):
    df["20DMA"] = df["Close"].rolling(20).mean()
    df["50DMA"] = df["Close"].rolling(50).mean()
    df["200DMA"] = df["Close"].rolling(200).mean()
    df["returns"] = df["Close"].pct_change()

    if pd.isna(df["200DMA"].iloc[-1]): return None

    latest = df.iloc[-1]
    close = float(latest["Close"])
    dma20 = float(latest["20DMA"])
    dma50 = float(latest["50DMA"])
    dma200 = float(latest["200DMA"])

    ret_1m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-22])) - 1
    ret_3m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-66])) - 1
    volatility = float(df["returns"].rolling(20).std().iloc[-1])
    avg_volume = float(df["Volume"].rolling(20).mean().iloc[-1])

    if close > dma50 and dma50 > dma200: cycle = "UPTREND"
    elif close < dma50 and dma50 < dma200: cycle = "DOWNTREND"
    else: cycle = "SIDEWAYS"

    return {
        "symbol": str(latest["symbol"]),
        "theme": THEME_MAP.get(str(latest["symbol"]), "OTHER"),
        "price": round(close, 2),
        "50DMA": round(dma50, 2),
        "cycle": cycle,
        "ret_1m": round(ret_1m, 4),
        "ret_3m": round(ret_3m, 4),
        "volatility": round(volatility, 4),
        "avg_volume": int(avg_volume),
        "pullback": round(abs(close - dma20) / dma20, 4),
        "stretch": round((close - dma50) / dma50, 4)
    }

def process_all():
    results = []
    print(f"🚀 Batch fetching data for {len(ETF_LIST)} ETFs in a single API call...")

    try:
        raw_data = yf.download(ETF_LIST, period="1y", interval="1d", group_by="ticker", progress=False)
    except Exception as e:
        print(f"⚠️ Batch download failed: {e}")
        return pd.DataFrame()

    if raw_data.empty:
        print("⚠️ Warning: No data was retrieved from Yahoo Finance.")
        return pd.DataFrame()

    print("✅ Download successful. Processing metrics...")

    for etf in ETF_LIST:
        try:
            df = raw_data[etf].copy()
            df = df.dropna(how='all') 

            if df.empty or len(df) < 200: 
                continue

            df["symbol"] = etf
            m = calculate_metrics(df)
            if m: results.append(m)

        except KeyError:
            print(f"⚠️ Could not process {etf} (Not found in Yahoo's batch return).")
            continue

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df["rank"] = result_df["ret_3m"].rank(ascending=False, method="min").astype(int)
    return result_df

# ==============================
# LOGIC & CLASSIFICATION
# ==============================

def detect_exhaustion(row):
    signals = 0
    if row["ret_1m"] < row["ret_3m"] / 3: signals += 1
    if row["stretch"] > 0.1: signals += 1
    if row["volatility"] > 0.035: signals += 1
    if row["rank"] > 10: signals += 1
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
    
    candidates = df[mask_hold & (df["avg_volume"] > 50000) & (df["volatility"] < 0.04)].copy()
    invest = pick_unique_themes(candidates)
    if not invest.empty:
        invest = invest[invest["pullback"] < 0.05]
        df.loc[df["symbol"].isin(invest["symbol"]), "action"] = "BUY"
        
    return df

def get_previous_states():
    try:
        insp = inspect(engine)
        if not insp.has_table("etf_metrics"): return {}, "Never"
            
        query_date = "SELECT MAX(date) FROM etf_metrics"
        max_date = pd.read_sql(query_date, engine).iloc[0, 0]
        
        if not max_date: return {}, "Never"
            
        query_data = f"SELECT symbol, action FROM etf_metrics WHERE date = '{max_date}'"
        prev_df = pd.read_sql(query_data, engine)
        return dict(zip(prev_df['symbol'], prev_df['action'])), max_date
    except:
        return {}, "Error"

# ==============================
# DISPLAY HELPERS
# ==============================

def print_clean_table(df, columns, headers):
    if df.empty:
        print("  --> No ETFs currently match this criteria.")
        return
    display_df = df.copy()
    for col in ["ret_1m", "ret_3m", "pullback", "stretch", "volatility"]:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
            
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(display_df[columns].rename(columns=dict(zip(columns, headers))).to_string(index=False))

# ==============================
# MAIN EXECUTOR
# ==============================

def main():
    ist_tz = ZoneInfo('Asia/Kolkata')
    today_str = datetime.now(ist_tz).strftime("%B %d, %Y")
    
    print("=" * 80)
    print(f" ETF QUANTITATIVE INVESTMENT GUIDE | {today_str}".center(80))
    print("=" * 80)

    df = process_all()
    
    if df.empty:
        print("Error: Could not retrieve market data.")
        return

    # Process Logic & States
    df = classify(df)
    prev_states, last_run_date = get_previous_states()
    df["prev_action"] = df["symbol"].map(prev_states).fillna("NONE")
    
    # Save to DB 
    save_df = df.drop(columns=["prev_action"])
    save_df["date"] = datetime.now(ist_tz).date()
    try:
        save_df.to_sql("etf_metrics", engine, if_exists="append", index=False)
    except:
        save_df.to_sql("etf_metrics", engine, if_exists="replace", index=False)

    print(f"Comparing today's data against last run: {last_run_date}\n")

    # Generate Delta DataFrames
    new_buys = df[(df["action"] == "BUY") & (df["prev_action"] != "BUY")].sort_values("rank")
    maintained_buys = df[(df["action"] == "BUY") & (df["prev_action"] == "BUY")].sort_values("rank")
    sell_alerts = df[(df["prev_action"].isin(["BUY", "HOLD"])) & (df["action"].isin(["SELL", "AVOID"]))].sort_values("rank")
    hold_df = df[df["action"] == "HOLD"].sort_values("rank")

    # ==============================
    # 📝 IN-MEMORY OUTPUT CAPTURE
    # ==============================
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer 
    
    print(f"REPORT DATE: {today_str}\n")
    
    if not sell_alerts.empty:
        print("🚨 ACTION REQUIRED: DOWNGRADED TO SELL (Exit these positions)")
        print("-" * 80)
        print_clean_table(sell_alerts, ["symbol", "prev_action", "cycle", "price", "50DMA", "exhausted"], ["ETF", "Previous Status", "Current Cycle", "Price", "50 DMA", "Is Exhausted?"])
        print()

    print("🟢 NEW BUYS (Triggered Today)")
    print("-" * 80)
    print_clean_table(new_buys, ["symbol", "theme", "rank", "ret_3m", "pullback"], ["ETF", "Theme", "Rank", "3M Return", "Pullback"])
    
    print("\n🔵 MAINTAINED BUYS (Already recommended on previous runs)")
    print("-" * 80)
    print_clean_table(maintained_buys, ["symbol", "theme", "rank", "ret_3m", "stretch"], ["ETF", "Theme", "Rank", "3M Return", "Stretch"])

    print("\n🟡 HOLD (Healthy Uptrends, keep if you already own)")
    print("-" * 80)
    print_clean_table(hold_df, ["symbol", "rank", "price", "50DMA", "stretch"], ["ETF", "Rank", "Price", "50 DMA", "Stretch"])

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
    print_clean_table(master_df, ["symbol", "rank", "action", "cycle", "price", "ret_1m", "ret_3m", "volatility", "pullback", "stretch", "exhausted"], ["ETF", "Rank", "Status", "Cycle", "Price", "1M Ret", "3M Ret", "Vol", "Pullback", "Stretch", "Exhausted"])
    print("\n" + "=" * 80 + "\n")

    # ==============================
    # 🔚 RESTORE CONSOLE & TRIGGER AI
    # ==============================
    sys.stdout = original_stdout 
    captured_report = output_buffer.getvalue()

    # 1. Print the report to the console so you can see it locally
    print(captured_report)

    # 2. Send the exact string in memory directly to AI & Telegram
    run_ai_analysis_and_notify(captured_report)

if __name__ == "__main__":
    main()
