# ==============================================================================
# 👑 F&O ULTRA PRO MAX v21.0 : BALANCED QUANT RADAR (TELEGRAM CLOUD)
# ==============================================================================
# Improvements:
# - EMA20/EMA50 trend + VWAP + RSI + ATR + breakout confirmation
# - Candle-based duplicate protection (not direction-only LAST_SIGNAL)
# - ATR-based SPOT stop-loss / target suggestions
# - OI/PCR is optional confirmation, never the only decision-maker
# - NIFTYBEES volume is explicitly treated as a proxy, NOT NIFTY true volume
# - Daily signal cap, EOD shutdown and Telegram via GitHub Secrets
#
# IMPORTANT:
# This is an alerting scanner, not an auto-trading system.
# No strategy can guarantee profitable signals. Backtest and paper-trade first.
# ==============================================================================

import datetime
import time
import json
import os
import warnings
import requests
import pytz
import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# MASTER SETTINGS
# ==============================================================================
IST = pytz.timezone("Asia/Kolkata")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MAX_TRADES_PER_DAY = 5          # This code counts ALERTS, not executed trades
SCAN_INTERVAL_MINUTES = 5
START_TIME = datetime.time(9, 30)
NO_ENTRY_AFTER = datetime.time(15, 20)
SHUTDOWN_TIME = datetime.time(15, 41)

MIN_VIX = 11.0
RSI_CE_MIN = 55
RSI_PE_MAX = 45
MIN_ATR_POINTS = 8.0
MAX_EMA_DISTANCE_ATR = 2.5
BREAKOUT_LOOKBACK = 6
COOLDOWN_CANDLES = 3

# ATR risk levels are NIFTY SPOT levels, not option-premium levels.
STOP_LOSS_ATR_MULTIPLIER = 1.2
TARGET_ATR_MULTIPLIER = 2.0

# PCR extremes are used only as a veto. Set False to ignore PCR completely.
USE_PCR_VETO = True
PCR_BEARISH_CE_BLOCK = 0.65
PCR_BULLISH_PE_BLOCK = 1.50

TRADES_TAKEN_TODAY = 0
DAILY_SIGNALS_COUNT = 0
TODAYS_DATE = None
LAST_SIGNAL_DIRECTION = None
LAST_SIGNAL_BAR = None
LAST_SIGNAL_BAR_POSITION = -999999


# ==============================================================================
# TELEGRAM
# ==============================================================================
def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(" [🔴 CONFIG ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            print(" [🟢 SUCCESS] Telegram alert sent.")
            return True
        print(f" [🔴 TELEGRAM ERROR] {response.status_code}: {response.text[:500]}")
    except Exception as exc:
        print(f" [🔴 TELEGRAM CONNECTION ERROR] {exc}")
    return False


def fire_startup_test():
    now_str = datetime.datetime.now(IST).strftime("%d-%b-%Y %I:%M:%S %p IST")
    msg = (
        "*🟢 F&O ULTRA PRO MAX v21.0 ACTIVE!*\n\n"
        "⚡ Balanced Quant Radar started.\n"
        f"📅 Time: {now_str}\n"
        f"🛡️ Daily alert cap: {MAX_TRADES_PER_DAY}\n"
        "📊 Indicators: EMA20/50 + VWAP + RSI + ATR + breakout\n"
        "⚠️ Alerts are not guaranteed winning trades."
    )
    send_telegram_msg(msg)


# ==============================================================================
# DATA HELPERS
# ==============================================================================
def normalize_yf_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # yfinance may return either (Price, Ticker) or (Ticker, Price) columns.
    if isinstance(df.columns, pd.MultiIndex):
        price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        level0 = set(df.columns.get_level_values(0))
        level1 = set(df.columns.get_level_values(1))

        if price_names.intersection(level0):
            df.columns = df.columns.get_level_values(0)
        elif price_names.intersection(level1):
            df.columns = df.columns.get_level_values(1)

    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    # Yahoo intraday timestamps are normally exchange-local and timezone-aware.
    # Normalize them to IST so candle filtering and volume alignment are consistent.
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize(IST)
        else:
            df.index = df.index.tz_convert(IST)
    except Exception:
        return pd.DataFrame()

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=required)


def download_intraday(ticker, period="5d"):
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval="5m",
            progress=False,
            threads=False,
            auto_adjust=False
        )
        return normalize_yf_frame(raw)
    except Exception as exc:
        print(f" [⚠️ DATA ERROR] {ticker}: {exc}")
        return pd.DataFrame()


def scalar(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def closed_candles(df, now_ist):
    if df.empty:
        return df
    current_block_start = now_ist.replace(
        minute=(now_ist.minute // 5) * 5,
        second=0,
        microsecond=0
    )
    return df[df.index < current_block_start].copy()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_indicators(df):
    df = df.copy()

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI_14"] = calculate_rsi(df["Close"], 14)

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)
    df["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # Volume-weighted average price using NIFTYBEES volume as a PROXY.
    # This is not true NIFTY index volume or an exchange-provided NIFTY VWAP.
    df["Date"] = df.index.date
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["PV"] = typical * df["Volume"]
    cum_pv = df.groupby("Date")["PV"].cumsum()
    cum_vol = df.groupby("Date")["Volume"].cumsum()
    df["VWAP_PROXY"] = (cum_pv / cum_vol.replace(0, pd.NA)).fillna(typical)

    # Breakout levels use only candles BEFORE the current closed candle.
    df["PREV_RANGE_HIGH"] = (
        df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    )
    df["PREV_RANGE_LOW"] = (
        df["Low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    )
    return df


# ==============================================================================
# NSE OPTION-CHAIN / PCR (OPTIONAL)
# ==============================================================================
def get_nse_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain"
    }

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        response = session.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f" [⚠️ OI WARNING] NSE option-chain unavailable: {exc}")
        return None


def analyze_oi_data(spot_price):
    data = get_nse_option_chain()
    if not data:
        return None, "OI unavailable"

    try:
        records = data["records"]["data"]
        expiries = data["records"]["expiryDates"]
        if not expiries:
            return None, "Expiry unavailable"

        nearest_expiry = expiries[0]
        atm = round(spot_price / 50) * 50
        ce_oi = 0
        pe_oi = 0

        for item in records:
            if item.get("expiryDate") != nearest_expiry:
                continue
            strike = scalar(item.get("strikePrice"))
            if abs(strike - atm) <= 250:
                ce_oi += scalar(item.get("CE", {}).get("openInterest", 0))
                pe_oi += scalar(item.get("PE", {}).get("openInterest", 0))

        if ce_oi <= 0:
            return None, "OI data insufficient"

        pcr = round(pe_oi / ce_oi, 2)
        return pcr, f"PCR {pcr} (nearest expiry, ±250 points)"
    except Exception as exc:
        print(f" [⚠️ OI PARSE WARNING] {exc}")
        return None, "OI parsing unavailable"


# ==============================================================================
# DAILY STATE
# ==============================================================================
def check_daily_reset():
    global TRADES_TAKEN_TODAY, DAILY_SIGNALS_COUNT, TODAYS_DATE
    global LAST_SIGNAL_DIRECTION, LAST_SIGNAL_BAR, LAST_SIGNAL_BAR_POSITION

    today = datetime.datetime.now(IST).date()
    if TODAYS_DATE != today:
        TODAYS_DATE = today
        TRADES_TAKEN_TODAY = 0
        DAILY_SIGNALS_COUNT = 0
        LAST_SIGNAL_DIRECTION = None
        LAST_SIGNAL_BAR = None
        LAST_SIGNAL_BAR_POSITION = -999999
        print(f"\n [🔄 NEW DAY] Daily counters reset for {today}.")


def send_daily_brief():
    msg = (
        "📊 *TODAY'S DAILY BRIEF*\n\n"
        f"🔹 *Total Alerts Generated:* {DAILY_SIGNALS_COUNT}\n"
        f"🔹 *Daily Alert Cap:* {MAX_TRADES_PER_DAY}\n"
        "🛑 Market closed. Scanner shutting down.\n\n"
        "⚠️ Alerts are not the same as executed trades or profitable trades."
    )
    send_telegram_msg(msg)


# ==============================================================================
# SCANNER
# ==============================================================================
def run_yfinance_scan():
    global LAST_SIGNAL_DIRECTION, LAST_SIGNAL_BAR, LAST_SIGNAL_BAR_POSITION
    global TRADES_TAKEN_TODAY, DAILY_SIGNALS_COUNT

    now_ist = datetime.datetime.now(IST)
    print(
        f"\n 🔓 SCANNING v21.0... "
        f"[{now_ist.strftime('%I:%M:%S %p IST')}]"
    )

    if TRADES_TAKEN_TODAY >= MAX_TRADES_PER_DAY:
        print(" [🛑 ALERT CAP] Daily alert limit reached.")
        return

    if now_ist.time() >= NO_ENTRY_AFTER:
        print(" [🛑 EOD SHIELD] No new entries after 03:20 PM IST.")
        return

    # 1. NIFTY spot candles
    nifty = closed_candles(download_intraday("^NSEI"), now_ist)
    if len(nifty) < 65:
        print(" [⚠️] Not enough closed NIFTY candles for EMA50/indicators.")
        return

    # 2. NIFTYBEES volume proxy
    bees = closed_candles(download_intraday("NIFTYBEES.NS"), now_ist)
    if not bees.empty and "Volume" in bees.columns:
        proxy_volume = bees["Volume"].reindex(nifty.index).fillna(0)
        nifty["Volume"] = proxy_volume
        print(" [📊] NIFTYBEES volume proxy aligned. Not true NIFTY volume.")
    else:
        nifty["Volume"] = 0.0
        print(" [⚠️] NIFTYBEES volume unavailable; volume confirmation disabled.")

    # 3. VIX: use only completed candles
    vix_df = closed_candles(download_intraday("^INDIAVIX"), now_ist)
    if vix_df.empty:
        print(" [⚠️] VIX unavailable. Skipping this scan for safety.")
        return

    live_vix = scalar(vix_df["Close"].iloc[-1])
    if live_vix < MIN_VIX:
        print(f" [🔕] VIX too low ({live_vix:.2f}); option buying filtered.")
        return

    # 4. Previous completed daily candle
    try:
        daily_raw = yf.download(
            "^NSEI", period="10d", interval="1d",
            progress=False, threads=False, auto_adjust=False
        )
        daily = normalize_yf_frame(daily_raw)
    except Exception:
        daily = pd.DataFrame()

    if len(daily) >= 2:
        prev_day_high = scalar(daily["High"].iloc[-2])
        prev_day_low = scalar(daily["Low"].iloc[-2])
    else:
        prev_day_high = None
        prev_day_low = None

    # 5. Indicators
    nifty = add_indicators(nifty).dropna(
        subset=["EMA_50", "RSI_14", "ATR_14", "PREV_RANGE_HIGH", "PREV_RANGE_LOW"]
    )
    if len(nifty) < 3:
        print(" [⚠️] Indicator history insufficient.")
        return

    current = nifty.iloc[-1]
    previous = nifty.iloc[-2]
    bar_time = nifty.index[-1]
    bar_position = len(nifty) - 1

    close = scalar(current["Close"])
    open_price = scalar(current["Open"])
    high = scalar(current["High"])
    low = scalar(current["Low"])
    ema20 = scalar(current["EMA_20"])
    ema50 = scalar(current["EMA_50"])
    vwap = scalar(current["VWAP_PROXY"])
    rsi = scalar(current["RSI_14"])
    atr = scalar(current["ATR_14"])
    prev_range_high = scalar(current["PREV_RANGE_HIGH"])
    prev_range_low = scalar(current["PREV_RANGE_LOW"])

    if atr < MIN_ATR_POINTS:
        print(f" [🛡️] ATR too low ({atr:.1f}); insufficient movement.")
        return

    ema_distance_atr = abs(close - ema20) / atr
    if ema_distance_atr > MAX_EMA_DISTANCE_ATR:
        print(
            f" [🛡️] Price overextended from EMA20 "
            f"({ema_distance_atr:.2f} ATR)."
        )
        return

    # Optional volume confirmation. Never treat missing proxy as strong volume.
    volume_ok = True
    volume_note = "Volume proxy unavailable"
    if "Volume" in nifty.columns:
        recent_vol = nifty["Volume"].iloc[-1]
        avg_vol = nifty["Volume"].rolling(10).mean().shift(1).iloc[-1]
        if scalar(avg_vol) > 0:
            volume_ok = scalar(recent_vol) >= scalar(avg_vol) * 1.05
            volume_note = (
                f"Proxy Vol {int(scalar(recent_vol))} / "
                f"Avg {int(scalar(avg_vol))}"
            )

    # Candle quality: body must be meaningful and close near breakout end.
    candle_range = max(high - low, 0.01)
    body = abs(close - open_price)
    bullish_candle = (
        close > open_price and body / candle_range >= 0.45
        and (high - close) / candle_range <= 0.30
    )
    bearish_candle = (
        close < open_price and body / candle_range >= 0.45
        and (close - low) / candle_range <= 0.30
    )

    # True breakout event: prior candle was at/below the level; latest closed
    # candle closes beyond the previous six-candle range.
    bullish_breakout = (
        close > prev_range_high
        and scalar(previous["Close"]) <= prev_range_high
    )
    bearish_breakout = (
        close < prev_range_low
        and scalar(previous["Close"]) >= prev_range_low
    )

    # OI is optional. Failed OI fetch does not automatically block a signal.
    pcr, oi_note = analyze_oi_data(close)

    print(
        f" 🎯 Candle: {bar_time.strftime('%d-%b %I:%M %p')} | "
        f"NIFTY: {close:.2f} | VWAP*: {vwap:.2f}"
    )
    print(
        f" 📈 EMA20: {ema20:.2f} | EMA50: {ema50:.2f} | "
        f"RSI: {rsi:.1f} | ATR: {atr:.1f} | VIX: {live_vix:.2f}"
    )
    print(f" 📊 {volume_note} | {oi_note}")

    signal = None
    logic = ""

    # CE setup
    if (
        close > vwap
        and ema20 > ema50
        and close > ema20
        and rsi >= RSI_CE_MIN
        and bullish_breakout
        and bullish_candle
        and volume_ok
    ):
        if prev_day_high is not None and 0 < (prev_day_high - close) < atr * 0.35:
            print(" [🛡️] CE blocked: too close to previous-day high.")
        elif USE_PCR_VETO and pcr is not None and pcr < PCR_BEARISH_CE_BLOCK:
            print(f" [🛡️] CE blocked by bearish PCR extreme ({pcr}).")
        else:
            signal = "CE"
            logic = "EMA20>EMA50 + above VWAP + RSI + bullish range breakout"

    # PE setup
    elif (
        close < vwap
        and ema20 < ema50
        and close < ema20
        and rsi <= RSI_PE_MAX
        and bearish_breakout
        and bearish_candle
        and volume_ok
    ):
        if prev_day_low is not None and 0 < (close - prev_day_low) < atr * 0.35:
            print(" [🛡️] PE blocked: too close to previous-day low.")
        elif USE_PCR_VETO and pcr is not None and pcr > PCR_BULLISH_PE_BLOCK:
            print(f" [🛡️] PE blocked by bullish PCR extreme ({pcr}).")
        else:
            signal = "PE"
            logic = "EMA20<EMA50 + below VWAP + RSI + bearish range breakdown"

    if not signal:
        print(" [⚪] No confirmed setup on this closed candle.")
        return

    # Duplicate and cooldown protection based on candle position.
    if LAST_SIGNAL_BAR == bar_time:
        print(" [🛡️] Duplicate alert blocked: same candle.")
        return

    candles_since_last = bar_position - LAST_SIGNAL_BAR_POSITION
    if (
        LAST_SIGNAL_DIRECTION == signal
        and candles_since_last < COOLDOWN_CANDLES
    ):
        print(
            f" [🛡️] Same-direction cooldown active "
            f"({candles_since_last}/{COOLDOWN_CANDLES} candles)."
        )
        return

    # Strike suggestion: nearest ATM. Confirm actual available strikes/expiry
    # and liquidity in the broker before placing any order.
    atm_strike = round(close / 50) * 50
    stop_loss = close - (atr * STOP_LOSS_ATR_MULTIPLIER) if signal == "CE" else close + (atr * STOP_LOSS_ATR_MULTIPLIER)
    target = close + (atr * TARGET_ATR_MULTIPLIER) if signal == "CE" else close - (atr * TARGET_ATR_MULTIPLIER)

    trade_info = {
        "Type": signal,
        "Strike": atm_strike,
        "Spot": close,
        "SL": stop_loss,
        "Target": target,
        "ATR": atr,
        "RSI": rsi,
        "VIX": live_vix,
        "PCR": pcr,
        "Logic": logic,
        "Volume": volume_note,
        "BarTime": bar_time.strftime("%d-%b-%Y %I:%M %p IST")
    }

    msg = (
        "*👑 F&O ULTRA PRO MAX v21.0*\n\n"
        f"⚡ Action: *BUY NIFTY {signal}*\n"
        f"📌 Suggested Strike: *{atm_strike} {signal} (ATM)*\n"
        f"📉 NIFTY Spot: ₹{close:.2f}\n"
        f"🕒 Closed Candle: {trade_info['BarTime']}\n\n"
        f"🧠 Logic: {logic}\n"
        f"📊 RSI: {rsi:.1f} | ATR: {atr:.1f} | VIX: {live_vix:.2f}\n"
        f"🧲 OI: {oi_note}\n"
        f"📦 {volume_note}\n\n"
        f"🛑 Suggested SPOT SL: ₹{stop_loss:.2f}\n"
        f"🎯 Suggested SPOT Target: ₹{target:.2f}\n"
        f"⚖️ Risk/Reward: 1 : "
        f"{TARGET_ATR_MULTIPLIER / STOP_LOSS_ATR_MULTIPLIER:.2f}\n\n"
        f"📊 Daily Alerts: {TRADES_TAKEN_TODAY + 1}/{MAX_TRADES_PER_DAY}\n\n"
        "⚠️ SL/Target are NIFTY SPOT reference levels, not option premium levels. "
        "Check option liquidity, spread and premium chart before entry. "
        "No guaranteed-profit signal."
    )

    print(f" [🟢] CONFIRMED {signal} SETUP DETECTED.")
    sent = send_telegram_msg(msg)

    # Count only alerts successfully accepted by Telegram.
    if sent:
        TRADES_TAKEN_TODAY += 1
        DAILY_SIGNALS_COUNT += 1
        LAST_SIGNAL_DIRECTION = signal
        LAST_SIGNAL_BAR = bar_time
        LAST_SIGNAL_BAR_POSITION = bar_position


# ==============================================================================
# CONTINUOUS CLOUD LOOP
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 75)
    print(" 👑 F&O ULTRA PRO MAX v21.0 - BALANCED QUANT RADAR")
    print("=" * 75)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. "
            "Add both in GitHub Actions Secrets."
        )

    fire_startup_test()

    daily_brief_sent_for = None

    while True:
        now_ist = datetime.datetime.now(IST)
        today = now_ist.date()

        # Weekend shutdown
        if now_ist.weekday() >= 5:
            print(" [🛑] Weekend detected. Scanner stopped.")
            break

        # Reset daily counters before checking the session.
        check_daily_reset()

        # Daily close brief
        if now_ist.time() >= SHUTDOWN_TIME:
            if daily_brief_sent_for != today:
                send_daily_brief()
                daily_brief_sent_for = today
            print(" [🛑] 03:41 PM IST reached. Graceful shutdown.")
            break

        # Wait until market scan start time
        if now_ist.time() < START_TIME:
            target = now_ist.replace(
                hour=START_TIME.hour,
                minute=START_TIME.minute,
                second=0,
                microsecond=0
            )
            wait_seconds = max(10, (target - now_ist).total_seconds())
            print(f" [⏳] Waiting until 09:30 AM IST ({int(wait_seconds)}s).")
            time.sleep(wait_seconds)
            continue

        if now_ist.time() < NO_ENTRY_AFTER:
            try:
                run_yfinance_scan()
            except Exception as exc:
                print(f" [🔴 SCANNER ERROR] {exc}")

        # Align next scan to the next 5-minute boundary + 20 seconds.
        now_ist = datetime.datetime.now(IST)
        next_minute = ((now_ist.minute // SCAN_INTERVAL_MINUTES) + 1) * SCAN_INTERVAL_MINUTES

        if next_minute >= 60:
            next_scan = now_ist.replace(
                hour=(now_ist.hour + 1) % 24,
                minute=0,
                second=20,
                microsecond=0
            )
        else:
            next_scan = now_ist.replace(
                minute=next_minute,
                second=20,
                microsecond=0
            )

        if next_scan <= now_ist:
            next_scan += datetime.timedelta(minutes=SCAN_INTERVAL_MINUTES)

        sleep_seconds = max(5, (next_scan - now_ist).total_seconds())
        print(
            f" [⏳] Next scan at "
            f"{next_scan.strftime('%I:%M:%S %p IST')} "
            f"(sleep {int(sleep_seconds)}s)."
        )
        time.sleep(sleep_seconds)
