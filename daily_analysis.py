import requests
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# 업비트 캔들 데이터 가져오기
def fetch_upbit_daily_candles(market="KRW-BTC", count=30):
    url = "https://api.upbit.com/v1/candles/days"
    params = {
        "market": market,
        "count": count
    }
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df = df.rename(columns={
        "candle_date_time_kst": "date",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    })
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").reset_index(drop=True)

# RSI 계산
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

# MACD 계산
def calculate_macd(prices, short=12, long=26, signal=9):
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema = prices.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return round(macd.iloc[-1], 5), round(signal_line.iloc[-1], 5)

# 패턴 감지 (고급)
def detect_pattern_enhanced(highs, lows):
    recent_highs = highs[-20:].to_numpy()
    recent_lows = lows[-20:].to_numpy()

    def is_ascending_triangle():
        highs_diff = np.diff(recent_highs[-10:])
        lows_trend = all(x < y for x, y in zip(recent_lows[-10:], recent_lows[-9:]))
        highs_flat = np.std(recent_highs[-10:]) < 0.01 * np.mean(recent_highs[-10:])
        return lows_trend and highs_flat

    def is_descending_triangle():
        highs_trend = all(x > y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        lows_flat = np.std(recent_lows[-10:]) < 0.01 * np.mean(recent_lows[-10:])
        return highs_trend and lows_flat

    def is_rectangle_range():
        high_range = max(recent_highs[-10:]) - min(recent_highs[-10:])
        low_range = max(recent_lows[-10:]) - min(recent_lows[-10:])
        return high_range / np.mean(recent_highs[-10:]) < 0.02 and low_range / np.mean(recent_lows[-10:]) < 0.02

    def is_cup_and_handle():
        if len(recent_lows) < 20:
            return False
        cup = recent_lows[-20:-10]
        handle = recent_lows[-10:]
        cup_bottom = min(cup)
        cup_top = max(cup[0], cup[-1])
        handle_max = max(handle)
        return (cup[0] > cup_bottom and cup[-1] > cup_bottom and handle_max < cup_top)

    def is_falling_wedge():
        high_trend = all(x > y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        low_trend = all(x > y for x, y in zip(recent_lows[-10:], recent_lows[-9:]))
        width_start = recent_highs[-10] - recent_lows[-10]
        width_end = recent_highs[-1] - recent_lows[-1]
        return high_trend and low_trend and width_end < width_start * 0.7

    def is_rising_wedge():
        high_trend = all(x < y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        low_trend = all(x < y for x, y in zip(recent_lows[-10:], recent_lows[-9:]))
        width_start = recent_highs[-10] - recent_lows[-10]
        width_end = recent_highs[-1] - recent_lows[-1]
        return high_trend and low_trend and width_end < width_start * 0.7

    def is_symmetrical_triangle():
        high_trend = all(x > y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        low_trend = all(x < y for x, y in zip(recent_lows[-10:], recent_lows[-9:]))
        return high_trend and low_trend

    def is_double_top():
        peaks = sorted(recent_highs[-5:])
        return abs(peaks[-1] - peaks[-2]) / peaks[-1] < 0.01

    def is_double_bottom():
        troughs = sorted(recent_lows[-5:])
        return abs(troughs[0] - troughs[1]) / troughs[1] < 0.01

    def is_head_and_shoulders():
        if len(recent_highs) < 7:
            return False
        try:
            left = recent_highs[-7]
            head = recent_highs[-5]
            right = recent_highs[-3]
            return head > left and head > right and abs(left - right) / head < 0.1
        except IndexError:
            return False

    def is_triple_top():
        tops = recent_highs[-6::2]
        return len(tops) == 3 and max(tops) - min(tops) < 0.01 * max(tops)

    def is_triple_bottom():
        bottoms = recent_lows[-6::2]
        return len(bottoms) == 3 and max(bottoms) - min(bottoms) < 0.01 * max(bottoms)

    def is_flag_pattern():
        if len(recent_highs) < 15:
            return False
        flag = recent_highs[-6:]
        return np.std(flag) / np.mean(flag) < 0.01

    def is_pennant_pattern():
        if len(recent_highs) < 15:
            return False
        return is_symmetrical_triangle()

    # 패턴 순서대로 검사(여러 패턴 동시 발견 시 첫 번째만 반환)
    if is_falling_wedge():
        return "하락 쐐기형"
    elif is_rising_wedge():
        return "상승 쐐기형"
    elif is_symmetrical_triangle():
        return "대칭 삼각형"
    elif is_double_top():
        return "이중 천장"
    elif is_double_bottom():
        return "이중 바닥"
    elif is_head_and_shoulders():
        return "헤드 앤 숄더"
    elif is_triple_top():
        return "삼중 천장"
    elif is_triple_bottom():
        return "삼중 바닥"
    elif is_flag_pattern():
        return "플래그 패턴"
    elif is_pennant_pattern():
        return "페넌트 패턴"
    elif is_ascending_triangle():
        return "상승 삼각형"
    elif is_descending_triangle():
        return "하락 삼각형"
    elif is_rectangle_range():
        return "박스권"
    elif is_cup_and_handle():
        return "컵 앤 핸들"
    else:
        return "none"

# Firestore 초기화
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def analyze_volume(df):
    recent_volumes = df["volume"].iloc[-15:-1]  # 직전 14일 평균
    today_volume = df["volume"].iloc[-1]
    avg_volume = recent_volumes.mean()
    if today_volume > avg_volume * 2:
        return f"거래량이 평균({round(avg_volume, 2)}) 대비 2배 이상({round(today_volume, 2)}) 급증하여 시장 참여가 활발합니다."
    elif today_volume > avg_volume * 1.2:
        return f"거래량이 평균보다 높아 매수/매도 심리가 강화되고 있습니다."
    else:
        return f"거래량({round(today_volume, 2)})은 최근 평균({round(avg_volume, 2)})과 유사하여 뚜렷한 변화는 없습니다."

# 분석 및 저장 실행
df = fetch_upbit_daily_candles("KRW-BTC", 30)
rsi = calculate_rsi(df["close"])
macd, macd_signal = calculate_macd(df["close"])
pattern = detect_pattern_enhanced(df["high"], df["low"])
volume_comment = analyze_volume(df)

# RSI 해석 문장
if rsi < 30:
    rsi_comment = f"RSI가 {rsi}로 과매도 구간에 진입해 반등 가능성이 있습니다."
elif rsi > 70:
    rsi_comment = f"RSI가 {rsi}로 과매수 구간에 위치해 조정 가능성이 있습니다."
else:
    rsi_comment = f"RSI는 {rsi}로 중립 구간입니다."

# MACD 해석 문장
if macd > macd_signal:
    macd_comment = f"MACD({macd})가 시그널선({macd_signal}) 위로 돌파하여 상승세로의 전환 신호입니다."
else:
    macd_comment = f"MACD({macd})가 시그널선({macd_signal}) 아래로 유지되며 하락세 지속 가능성을 시사합니다."

# 패턴별 해석 사전
pattern_explanations = {
    "하락 쐐기형": "통상 하락세에서 나타나며 이후 반등 가능성을 시사합니다.",
    "상승 쐐기형": "상승세에서 나타나며 이후 가격 조정 가능성을 시사합니다.",
    "대칭 삼각형": "변동성 축소 후 방향성 돌파가 예상됩니다.",
    "이중 천장": "고점 부근에서의 하락 반전 가능성을 나타냅니다.",
    "이중 바닥": "저점 부근에서의 상승 반전 가능성을 나타냅니다.",
    "헤드 앤 숄더": "상승세 이후 하락 반전의 대표적인 신호입니다.",
    "삼중 천장": "고점에서의 강한 하락 반전 가능성을 나타냅니다.",
    "삼중 바닥": "저점에서의 강한 반등 신호로 해석될 수 있습니다.",
    "플래그 패턴": "단기 조정 이후 기존 추세 지속 가능성이 높습니다.",
    "페넌트 패턴": "가격 조정 이후 추세 지속 가능성을 나타냅니다.",
    "상승 삼각형": "상승 돌파 가능성이 높은 패턴입니다.",
    "하락 삼각형": "하락 돌파 가능성이 높은 패턴입니다.",
    "박스권": "횡보장이 지속될 가능성을 나타냅니다.",
    "컵 앤 핸들": "상승 돌파 가능성을 가지는 중장기 강세 패턴입니다.",
}

# 패턴 해석 문장
if pattern != "none":
    explanation = pattern_explanations.get(pattern, "기술적 해석이 제공되지 않았습니다.")
    pattern_comment = f"현재 '{pattern}' 패턴이 감지되었습니다. {explanation}"
else:
    pattern_comment = "현재 명확한 차트 패턴은 감지되지 않았습니다."

# 통합 요약
summary = f"{rsi_comment} {macd_comment} {pattern_comment} {volume_comment}"

doc_id = f"BTC_{datetime.now().strftime('%Y-%m-%d')}"
data = {
    "symbol": "BTC",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "rsi": rsi,
    "macd": macd,
    "macdSignal": macd_signal,
    "pattern": pattern,
    "summary": summary,
    "premiumOnly": False,
    "isDailyPick": True,
    "chartData": df.to_dict(orient="records")
}

db.collection("daily_analysis").document(doc_id).set(data)
print(f"✅ Firestore 저장 완료: {doc_id}")