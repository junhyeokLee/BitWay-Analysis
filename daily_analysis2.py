"""
requirements.txt 에 들어갈 패키지
---------------------------------
requests
pandas
numpy
firebase-admin
python-dotenv
"""

import requests
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os  # OpenAI 키 환경변수 사용
from dotenv import load_dotenv
load_dotenv()
import time
import random

# ────────────────────────────────────────────────
# 0) Firestore 초기화 ─ 서비스 계정 JSON 경로 수정
# ────────────────────────────────────────────────
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

UPBIT_HEADERS = {"Accept": "application/json"}

# ────────────────────────────────────────────────
# 1) 업비트 일봉 캔들 데이터 가져오기
# ────────────────────────────────────────────────
def fetch_upbit_daily_candles(market="KRW-BTC", count=30):
    url = "https://api.upbit.com/v1/candles/days"
    params = {"market": market, "count": count}
    response = requests.get(url, params=params, headers=UPBIT_HEADERS, timeout=5)
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

# ────────────────────────────────────────────────
# 2) 기술 지표 계산
# ────────────────────────────────────────────────
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_macd(prices, short=12, long=26, signal=9):
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema  = prices.ewm(span=long,  adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return round(macd.iloc[-1], 5), round(signal_line.iloc[-1], 5)

# ────────────────────────────────────────────────
# 3) 고급 차트 패턴 감지
# ────────────────────────────────────────────────
def detect_pattern_enhanced(highs, lows):
    recent_highs = highs[-20:].to_numpy()
    recent_lows  = lows[-20:].to_numpy()

    def is_ascending_triangle():
        highs_diff = np.diff(recent_highs[-10:])
        lows_trend = all(x < y for x, y in zip(recent_lows[-10:], recent_lows[-9:]))
        highs_flat = np.std(recent_highs[-10:]) < 0.01 * np.mean(recent_highs[-10:])
        return lows_trend and highs_flat

    def is_descending_triangle():
        highs_trend = all(x > y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        lows_flat   = np.std(recent_lows[-10:]) < 0.01 * np.mean(recent_lows[-10:])
        return highs_trend and lows_flat

    def is_rectangle_range():
        high_range = max(recent_highs[-10:]) - min(recent_highs[-10:])
        low_range  = max(recent_lows[-10:])  - min(recent_lows[-10:])
        return (high_range / np.mean(recent_highs[-10:]) < 0.02 and
                low_range  / np.mean(recent_lows[-10:])  < 0.02)

    def is_cup_and_handle():
        if len(recent_lows) < 20:
            return False
        cup     = recent_lows[-20:-10]
        handle  = recent_lows[-10:]
        cup_bottom = min(cup)
        cup_top    = max(cup[0], cup[-1])
        handle_max = max(handle)
        return (cup[0] > cup_bottom and cup[-1] > cup_bottom and handle_max < cup_top)

    def is_falling_wedge():
        high_trend = all(x > y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        low_trend  = all(x > y for x, y in zip(recent_lows[-10:],  recent_lows[-9:]))
        width_start = recent_highs[-10] - recent_lows[-10]
        width_end   = recent_highs[-1]  - recent_lows[-1]
        return high_trend and low_trend and width_end < width_start * 0.7

    def is_rising_wedge():
        high_trend = all(x < y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        low_trend  = all(x < y for x, y in zip(recent_lows[-10:],  recent_lows[-9:]))
        width_start = recent_highs[-10] - recent_lows[-10]
        width_end   = recent_highs[-1]  - recent_lows[-1]
        return high_trend and low_trend and width_end < width_start * 0.7

    def is_symmetrical_triangle():
        high_trend = all(x > y for x, y in zip(recent_highs[-10:], recent_highs[-9:]))
        low_trend  = all(x < y for x, y in zip(recent_lows[-10:],  recent_lows[-9:]))
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
        left  = recent_highs[-7]
        head  = recent_highs[-5]
        right = recent_highs[-3]
        return (head > left and head > right and
                abs(left - right) / head < 0.1)

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

    # 패턴 판별 순서
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

# ────────────────────────────────────────────────
# 3.5) AI 요약 유틸 (OpenAI) - 환경변수 OPENAI_API_KEY 필요
# ────────────────────────────────────────────────
def validate_inputs(df, rsi, macd, macd_signal, pattern, volume_comment):
    # 한글 주석: generate_ai_summary 호출 전 필수 값/컬럼 검증
    if df is None or len(df) < 2:
        raise ValueError("df가 비었거나 데이터가 2행 미만입니다.")
    required_cols = {"date", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {missing}")
    for name, val in [("rsi", rsi), ("macd", macd), ("macd_signal", macd_signal), ("pattern", pattern), ("volume_comment", volume_comment)]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            raise ValueError(f"입력 값이 비어 있습니다: {name}")

def compute_change_metrics(df):
    """
    한글 주석: 변화 지표 계산 (전일 대비 수익률, 7일 평균 대비 괴리, ATR(14), 거래량 Z-Score)
    df에는 'date','open','high','low','close','volume'가 존재한다고 가정.
    """
    s_close = df["close"].astype(float)
    s_high  = df["high"].astype(float)
    s_low   = df["low"].astype(float)
    s_vol   = df["volume"].astype(float)

    # 전일 대비 수익률(%)
    if len(s_close) >= 2:
        d1_ret = (s_close.iloc[-1] / s_close.iloc[-2] - 1) * 100.0
    else:
        d1_ret = 0.0

    # 7일 평균 대비 괴리(%)
    w7 = s_close.tail(7)
    w7_mean = w7.mean() if len(w7) > 0 else s_close.mean()
    dev7 = (s_close.iloc[-1] / w7_mean - 1) * 100.0 if w7_mean else 0.0

    # ATR(14)
    tr_list = []
    for i in range(1, len(df)):
        tr = max(
            s_high.iloc[i] - s_low.iloc[i],
            abs(s_high.iloc[i] - s_close.iloc[i-1]),
            abs(s_low.iloc[i] - s_close.iloc[i-1])
        )
        tr_list.append(tr)
    atr14 = float(pd.Series(tr_list).rolling(14).mean().iloc[-1]) if len(tr_list) >= 14 else float(np.mean(tr_list) if tr_list else 0.0)

    # 거래량 Z-Score (최근 20일)
    vol_tail = s_vol.tail(20) if len(s_vol) >= 20 else s_vol
    mu = float(vol_tail.mean()) if len(vol_tail) else 0.0
    sd = float(vol_tail.std()) if len(vol_tail) else 0.0
    vol_z = (s_vol.iloc[-1] - mu) / (sd + 1e-8) if sd > 0 else 0.0

    return {
        "d1_return_pct": round(float(d1_ret), 2),
        "dev7_pct": round(float(dev7), 2),
        "atr14": round(float(atr14), 4),
        "vol_z": round(float(vol_z), 2),
        "last_close": float(s_close.iloc[-1])
    }

def build_ai_prompt(df, rsi, macd, macd_signal, pattern, volume_comment):
    # 한글 주석: 프롬프트 구성 (사실/해석 분리, 보수적 톤)
    change = compute_change_metrics(df)
    recent = df[["date", "open", "high", "low", "close", "volume"]].tail(7).to_dict("records")
    pattern_text = pattern if pattern and pattern != "none" else "감지됨 없음"
    prompt = f"""
너는 보수적인 크립토 애널리스트야. 한국어로 작성해.
다음 데이터를 바탕으로 오늘의 '변화'를 중심으로 구조화된 일간 리포트를 만들어.
- 과장/투자 권유/수익 보장 금지
- '사실'과 '해석'을 분리

[핵심 지표]
- 종가: {change["last_close"]}
- 전일 대비 수익률(%): {change["d1_return_pct"]}
- 7일 평균 대비 괴리(%): {change["dev7_pct"]}
- ATR(14): {change["atr14"]}
- 거래량 Z-Score(20일): {change["vol_z"]}
- RSI: {rsi}
- MACD: {macd} / Signal: {macd_signal}
- 패턴: {pattern_text}
- 거래량 코멘트: {volume_comment}

[최근 7일 요약 데이터] (최신이 마지막)
{recent}

[출력 형식]
1) 한 줄 요약(오늘 무엇이 달랐나)
2) 사실(Facts): 숫자/패턴/변화 포인트 3~5개
3) 해석(Interpretation): 보수적 관점에서 3~4문장
4) 리스크 2개 (구체적 지표/이벤트 기반)
5) 시나리오(상방/하방)와 각각의 트리거 1개씩
6) 확신도(0~100)와 근거 1문장
"""
    return prompt

# 간단한 지수 백오프 재시도 유틸
def _with_retry(fn, max_try=2, base_delay=0.8):
    last = None
    for i in range(max_try):
        last = fn()
        if last:
            return last
        time.sleep(base_delay * (2 ** i))
    return None

def _call_openai_chat(prompt):
    """
    한글 주석: OpenAI Chat Completions REST 호출 (requests만 사용).
    환경변수 OPENAI_API_KEY 필요, OPENAI_MODEL 미지정 시 gpt-4o-mini 사용.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "temperature": 0.3,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": "You are a cautious Korean crypto analyst. Never give financial advice."},
                {"role": "user", "content": prompt}
            ]
        }
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def generate_ai_summary(df, rsi, macd, macd_signal, pattern, volume_comment):
    """
    한글 주석: OpenAI 실패 시 규칙 기반 폴백 문구로 대체.
    키는 반드시 환경변수(또는 .env)로 주입.
    """
    validate_inputs(df, rsi, macd, macd_signal, pattern, volume_comment)
    prompt = build_ai_prompt(df, rsi, macd, macd_signal, pattern, volume_comment)

    # 1) OpenAI 호출 (최대 2회 재시도)
    text = _with_retry(lambda: _call_openai_chat(prompt), max_try=2, base_delay=0.8)
    provider = 'openai' if text else 'fallback'

    # 2) 실패 시 규칙 기반 폴백
    if not text:
        pattern_comment = "명확한 패턴 없음" if (not pattern or pattern == "none") else f"'{pattern}' 패턴 감지"
        text = (
            f"시장 요약: RSI {rsi}, MACD {macd} / 시그널 {macd_signal}. {pattern_comment}.\n"
            f"해석: 과장 없이 보수적으로 접근이 필요합니다. {volume_comment}\n"
            "- 체크포인트: 변동성, 거래량 흐름\n- 체크포인트: 주요 지지/저항 확인"
        )

    try:
        print(f"AI provider used: {provider}")
    except Exception:
        pass
    disclaimer = "\n\n※ 본 내용은 정보 제공 목적이며, 투자 조언이 아닙니다."
    return (text + disclaimer).strip()

# ────────────────────────────────────────────────
# 4) 거래량 해석
# ────────────────────────────────────────────────
def analyze_volume(df):
    recent_volumes = df["volume"].iloc[-15:-1]  # 직전 14일
    today_volume   = df["volume"].iloc[-1]
    avg_volume     = recent_volumes.mean()
    if today_volume > avg_volume * 2:
        return f"거래량이 평균({round(avg_volume,2)}) 대비 2배 이상({round(today_volume,2)}) 급증하여 시장 참여가 활발합니다."
    elif today_volume > avg_volume * 1.2:
        return f"거래량이 평균보다 높아 매수/매도 심리가 강화되고 있습니다."
    else:
        return f"거래량({round(today_volume,2)})은 최근 평균({round(avg_volume,2)})과 유사하여 뚜렷한 변화는 없습니다."

# ────────────────────────────────────────────────
# 5) HOT 코인 후보 추출 & 무작위 1개 선택
# ────────────────────────────────────────────────
def get_krw_markets():
    resp = requests.get("https://api.upbit.com/v1/market/all",
                        params={"isDetails": "false"},
                        headers=UPBIT_HEADERS, timeout=5)
    resp.raise_for_status()
    return [m["market"] for m in resp.json() if m["market"].startswith("KRW-")]

def get_top_movers(top_n=10, min_change=0.05):
    markets = get_krw_markets()
    rows = []
    for i in range(0, len(markets), 100):
        chunk = ",".join(markets[i:i+100])
        r = requests.get("https://api.upbit.com/v1/ticker",
                         params={"markets": chunk},
                         headers=UPBIT_HEADERS, timeout=5)
        r.raise_for_status()
        rows.extend(r.json())
        time.sleep(0.05)  # 속도 제한

    df = pd.DataFrame(rows)
    df["change_rate"]  = df["signed_change_rate"].astype(float)
    df["trade_amount"] = df["acc_trade_price_24h"].astype(float)

    cand = (df[df["change_rate"] >= min_change]
            .sort_values(["change_rate", "trade_amount"], ascending=[False, False])
            .head(top_n)["market"].tolist())

    if len(cand) < top_n:
        extra = df.sort_values("trade_amount", ascending=False).head(top_n)["market"].tolist()
        cand = list(dict.fromkeys(cand + extra))[:top_n]
    return cand

def pick_today_hot_coin():
    candidates = get_top_movers()
    print("🔥 HOT 후보:", candidates)
    chosen = random.choice(candidates)
    print("🎯 오늘의 HOT 코인:", chosen)
    return chosen

# ────────────────────────────────────────────────
# 6) 메인 실행: 핫 코인 1개 분석 후 Firestore 저장
# ────────────────────────────────────────────────
def main():
    # ① 오늘의 핫 코인 선정
    hot_market = pick_today_hot_coin()          # 예: 'KRW-ETH'

    # ② 캔들 데이터 불러오기
    df = fetch_upbit_daily_candles(hot_market, 30)

    # ③ 지표·패턴·거래량 분석
    rsi            = calculate_rsi(df["close"])
    macd, macd_sig = calculate_macd(df["close"])
    pattern        = detect_pattern_enhanced(df["high"], df["low"])
    volume_comment = analyze_volume(df)

    # ④ RSI 코멘트
    if rsi < 30:
        rsi_comment = f"RSI가 {rsi}로 과매도 구간에 진입해 반등 가능성이 있습니다."
    elif rsi > 70:
        rsi_comment = f"RSI가 {rsi}로 과매수 구간에 위치해 조정 가능성이 있습니다."
    else:
        rsi_comment = f"RSI는 {rsi}로 중립 구간입니다."

    # ⑤ MACD 코멘트
    if macd > macd_sig:
        macd_comment = f"MACD({macd})가 시그널선({macd_sig}) 위로 돌파하여 상승세로의 전환 신호입니다."
    else:
        macd_comment = f"MACD({macd})가 시그널선({macd_sig}) 아래로 유지되며 하락세 지속 가능성을 시사합니다."

    # ⑥ 패턴 해석
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
    if pattern != "none":
        explanation = pattern_explanations.get(pattern, "기술적 해석이 제공되지 않았습니다.")
        pattern_comment = f"현재 '{pattern}' 패턴이 감지되었습니다. {explanation}"
    else:
        pattern_comment = "현재 명확한 차트 패턴은 감지되지 않았습니다."

    # ⑦ 통합 요약
    summary = f"{rsi_comment} {macd_comment} {pattern_comment} {volume_comment}"

    # ⑦-1) AI 요약 (OpenAI → 실패 시 폴백)
    ai_summary = generate_ai_summary(df, rsi, macd, macd_sig, pattern, volume_comment)

    # ⑧ Firestore 저장
    symbol = hot_market.replace("KRW-", "")
    today  = datetime.now().strftime("%Y-%m-%d")
    doc_id = f"HOT_{symbol}_{today}"

    data = {
        "symbol": symbol,
        "date": today,
        "rsi": rsi,
        "macd": macd,
        "macdSignal": macd_sig,
        "pattern": pattern,
        "summary": summary,
        "aiSummary": ai_summary,
        "premiumOnly": False,
        "isDailyPick": True,
        "chartData": df.to_dict(orient="records")
    }

    db.collection("daily_analysis").document(doc_id).set(data)
    print(f"✅ Firestore 저장 완료: {doc_id}")

# ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
