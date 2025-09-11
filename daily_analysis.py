# === 규칙기반 요약/JSON 백업 생성기 ===
def _build_rule_based_ai_json(*, symbol: str, rsi: float, macd: float, macd_signal: float,
                              pattern: str, volume_comment: str, features: dict,
                              scores: dict, triggers: dict) -> (dict, str):
    """
    AI 요약 JSON이 없을 때(또는 파싱 실패 시) 최소 필수 구조를 채워주는 백업.
    반환: (ai_json_dict, headline_str)
    """
    regime = (features or {}).get("vol_regime") or "중립"
    support = (features or {}).get("support_20d")
    resistance = (features or {}).get("resistance_20d")
    bull_trig = (triggers or {}).get("bull_trigger")
    bear_trig = (triggers or {}).get("bear_trigger")
    macd_cross = "상향크로스" if macd > macd_signal else "하향크로스"
    headline = f"RSI {round(rsi,1)} / MACD {macd_cross}, 패턴 {'없음' if pattern=='none' else pattern} (레짐: {regime})"

    facts = []
    if rsi is not None:
        facts.append(f"RSI: {round(rsi,1)}")
    facts.append(f"MACD: {round(macd,3)} / Signal: {round(macd_signal,3)}")
    if pattern and pattern != "none":
        facts.append(f"패턴: {pattern}")
    if support is not None and resistance is not None:
        facts.append(f"지지/저항(20D): {support} / {resistance}")
    if bull_trig is not None and bear_trig is not None:
        facts.append(f"자동 트리거: 상방 {bull_trig} / 하방 {bear_trig}")

    interpretation = []
    if rsi < 30:
        interpretation.append("과매도 구간으로 반등 가능성.")
    elif rsi > 70:
        interpretation.append("과매수 구간으로 조정 가능성.")
    else:
        interpretation.append("RSI 중립 구간.")
    if macd > macd_signal:
        interpretation.append("MACD가 시그널 상방 → 상승 전환 신호.")
    else:
        interpretation.append("MACD가 시그널 하방 → 하락 지속 신호.")
    if regime:
        interpretation.append(f"레짐: {regime}.")
    if volume_comment:
        interpretation.append(volume_comment.strip())

    # === 초보 친화 요약/뱃지 생성 ===
    # 한 줄 요약을 쉬운 문장으로 재구성하고, 현재 상태를 나타내는 키워드 뱃지를 만든다.
    try:
        rsi_zone = ("과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립"))
    except Exception:
        rsi_zone = "중립"
    macd_dir = "상승" if macd > macd_signal else "하락"
    # volume_comment를 간단 요약으로 변환(길면 첫 문장만 사용)
    vol_simple = (volume_comment or "").split(".")[0].strip()
    if not vol_simple:
        vol_simple = "거래량 변화 정보 부족"
    easy_summary = (
        f"현재 {symbol}은 {regime} 시장에 있으며, RSI {round(rsi,1)}로 {rsi_zone} 구간입니다. "
        f"MACD는 {macd_dir} 신호를 보이고 있고, {vol_simple}"
    )

    badges = []
    if regime: badges.append(regime)
    # RSI 구간 배지
    if rsi > 70:
        badges.append("과매수")
    elif rsi < 30:
        badges.append("과매도")
    else:
        badges.append("RSI 중립")
    # MACD 방향 배지
    badges.append("상승 신호" if macd > macd_signal else "하락 신호")
    # 거래량 배지
    if any(k in (volume_comment or "") for k in ["급증", "높아", "강화"]):
        badges.append("거래량↑")

    ai_json = {
        "symbol": symbol,
        "headline": headline,
        "easy_summary": easy_summary,
        "badges": badges,
        "regime": regime,
        "key_levels": {"support": support, "resistance": resistance},
        "facts": facts,
        "interpretation": interpretation,
        "scenarios": {
            "bull": {"trigger": f">= {bull_trig}" if bull_trig is not None else "N/A",
                      "invalidate": "종가가 20D 지지 하회"},
            "bear": {"trigger": f"<= {bear_trig}" if bear_trig is not None else "N/A",
                      "invalidate": "종가가 20D 저항 상회"}
        },
        "risks": [
            "고변동 구간에서는 슬리피지 확대 가능",
            "거래량 급증 후 되돌림 리스크"
        ],
        "confidence": int((scores or {}).get("ruleConfidence", 50)),
        "indicators": {
            "rsi": None if rsi is None else float(round(rsi, 2)),
            "macd": float(round(macd, 3)),
            "macdSignal": float(round(macd_signal, 3)),
            "ema_cross": (features or {}).get("ema_cross"),
            "bb_pos_pct": (features or {}).get("bb_pos_pct"),
        },
    }
    return ai_json, headline

import requests
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import os  # 환경변수로 API 키 관리
import json  # JSON 파싱/검증용
# from dotenv import load_dotenv
from datetime import timezone
# load_dotenv()

# 본 스크립트는 KST 기준 하루 1회, 업비트 거래대금/규칙점수 기반으로 추천 코인을 선택하고 저장합니다.

# === KST(UTC+9) 기준 'YYYY-MM-DD' ===
def kst_today_str():
    return (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")


# --------- 파라미터화(튜닝 가능) ---------
K_TRIGGER = float(os.getenv("K_TRIGGER", "0.30"))           # 트리거 계수 k (0.2~0.5 권장)
BT_HORIZON_DAYS = int(os.getenv("BT_HORIZON_DAYS", "3"))    # 백테스트: 목표 달성 확인 기간(일)
BT_TARGET_ATR = float(os.getenv("BT_TARGET_ATR", "1.0"))    # 백테스트: 목표 폭(ATR 배수)
# ---------------------------------------

# === 오늘의 추천 심볼 설정/자동선정 ===
HOT_PREFIX = os.getenv("HOT_PREFIX", "HOT").strip()
HOT_SYMBOL_ENV = os.getenv("HOT_SYMBOL", "ADA").upper().strip()

# === 업비트 기반 일일 추천 선정 파라미터 ===
DAILY_TOPK = int(os.getenv("DAILY_TOPK", "30"))               # 거래대금 상위 K개만 심층 평가
MIN_TURNOVER_KRW = float(os.getenv("MIN_TURNOVER_KRW", "500000000"))  # 24h 거래대금 하한 (기본 5억 KRW)
EXCLUDE_SYMBOLS = {s.strip().upper() for s in os.getenv("EXCLUDE_SYMBOLS", "USDT,USDC,DAI,KRW").split(",") if s.strip()}

# === 업비트 KRW 마켓 리스트/티커 헬퍼 ===
def _upbit_get_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        r = requests.get(url, params={"isDetails": "true"}, timeout=10)
        r.raise_for_status()
        j = r.json()
        markets = [x["market"] for x in j if x.get("market", "").startswith("KRW-") and not x.get("market_warning")]
        syms = [m.split("-")[1] for m in markets]
        return syms
    except Exception:
        return ["BTC", "ETH", "ADA", "SOL", "XRP"]

def _upbit_get_tickers_24h(symbols):
    markets = ",".join([f"KRW-{s}" for s in symbols])
    if not markets:
        return {}
    url = "https://api.upbit.com/v1/ticker"
    try:
        r = requests.get(url, params={"markets": markets}, timeout=10)
        r.raise_for_status()
        j = r.json()
        out = {}
        for x in j:
            m = x.get("market", "KRW-XXX")
            sym = m.split("-")[1]
            out[sym] = {
                "acc_trade_price_24h": float(x.get("acc_trade_price_24h", 0.0)),
                "acc_trade_volume_24h": float(x.get("acc_trade_volume_24h", 0.0)),
                "trade_price": float(x.get("trade_price", 0.0)),
            }
        return out
    except Exception:
        return {}

def select_daily_symbol_kst(date_str: str):
    """
    KST 날짜(YYYY-MM-DD)별로 1개의 추천 심볼을 결정.
    규칙:
      1) KRW 마켓 전량 → EXCLUDE_SYMBOLS 제외
      2) 24h 거래대금(acc_trade_price_24h) 내림차순 상위 DAILY_TOPK 선별
      3) MIN_TURNOVER_KRW 미만 제거
      4) 상위 후보에 대해 규칙점수(ruleConfidence) 재계산 → 최고점 선택
         (동점 시) 거래대금 큰 순 → 심볼 오름차순
    Firestore의 `daily_picks/{date_str}` 문서로 동결(idempotent)
    """
    # 1) 이미 오늘자 픽이 존재하면 그것을 사용
    picks_doc = db.collection("daily_picks").document(date_str).get()
    if picks_doc.exists:
        saved = picks_doc.to_dict() or {}
        sym = (saved.get("symbol") or saved.get("sel_symbol") or HOT_SYMBOL_ENV).upper()
        return sym

    # 2) 후보 수집 (Upbit)
    syms = _upbit_get_krw_markets()
    syms = [s for s in syms if s.upper() not in EXCLUDE_SYMBOLS]
    tick = _upbit_get_tickers_24h(syms)
    ranked = [
        (s, t.get("acc_trade_price_24h", 0.0)) for s, t in tick.items() if t.get("acc_trade_price_24h", 0.0) >= MIN_TURNOVER_KRW
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_syms = [s for s, _ in ranked[:DAILY_TOPK]]
    if not top_syms:
        top_syms = [HOT_SYMBOL_ENV]

    # 3) 상위 후보 재평가 (규칙 점수)
    evaluated = []
    for s in top_syms:
        try:
            df_tmp = fetch_upbit_daily_candles(f"KRW-{s}", 200)
            rsi_tmp = calculate_rsi(df_tmp["close"]) if len(df_tmp) else 50
            feats_tmp = compute_advanced_features_v2(df_tmp) if len(df_tmp) else {}
            scores_tmp = compute_rule_based_scores(feats_tmp, rsi_value=rsi_tmp)
            score_tmp = int(scores_tmp.get("ruleConfidence", 0))
            tv_tmp = float(tick.get(s, {}).get("acc_trade_price_24h", 0.0))
            evaluated.append((s, score_tmp, tv_tmp))
        except Exception:
            continue
    if not evaluated:
        chosen = HOT_SYMBOL_ENV
    else:
        # 정렬: score DESC → turnover DESC → symbol ASC
        evaluated.sort(key=lambda x: x[0])               # tie-break 2: symbol ASC
        evaluated.sort(key=lambda x: x[2], reverse=True) # tie-break 1: turnover DESC
        evaluated.sort(key=lambda x: x[1], reverse=True) # primary: score DESC
        chosen = evaluated[0][0]

    # 4) Firestore에 오늘자 픽을 기록(동결)
    try:
        db.collection("daily_picks").document(date_str).set({
            "symbol": chosen,
            "decidedAtKst": date_str,
            "topK": DAILY_TOPK,
            "minTurnoverKrw": MIN_TURNOVER_KRW,
        })
    except Exception:
        pass
    return chosen

# 업비트 캔들 데이터 가져오기
def fetch_upbit_daily_candles(market="KRW-BTC", count=200):
    url = "https://api.upbit.com/v1/candles/days"
    params = {"market": market, "count": count}
    headers = {"Accept": "application/json"}
    # 간단 재시도 + 타임아웃
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.7 * (2 ** attempt))
                continue
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
            # Firestore 호환을 위해 문자열 유지
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df.sort_values("date").reset_index(drop=True)
        except requests.RequestException:
            time.sleep(0.7 * (2 ** attempt))
    raise RuntimeError("Upbit API 호출 실패")

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
    # 한국어 주석: 캔들 수가 부족하면(예: 신규 상장/데이터 적음) 패턴 감지를 건너뛰고 'none' 반환
    try:
        if highs is None or lows is None:
            return "none"
        if len(highs) < 20 or len(lows) < 20:
            return "none"
    except Exception:
        return "none"
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

 # 0) Firestore 초기화 ─ 서비스 계정 JSON 경로 수정
 # ────────────────────────────────────────────────
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def analyze_volume(df):
    # 한글 주석: 거래량 분석 (데이터 부족/NaN 대비)
    if df is None or len(df) < 2 or "volume" not in df:
        return "거래량 데이터가 부족하여 해석을 제공할 수 없습니다."
    recent_volumes = df["volume"].iloc[-15:-1]  # 직전 14일 평균
    today_volume = float(df["volume"].iloc[-1])
    avg_volume = float(recent_volumes.mean()) if len(recent_volumes) > 0 else float("nan")
    if np.isnan(avg_volume) or avg_volume == 0:
        return "거래량 데이터가 부족하여 해석을 제공할 수 없습니다."
    if today_volume > avg_volume * 2:
        return f"거래량이 평균({round(avg_volume, 2)}) 대비 2배 이상({round(today_volume, 2)}) 급증하여 시장 참여가 활발합니다."
    elif today_volume > avg_volume * 1.2:
        return "거래량이 평균보다 높아 매수/매도 심리가 강화되고 있습니다."
    else:
        return f"거래량({round(today_volume, 2)})은 최근 평균({round(avg_volume, 2)})과 유사하여 뚜렷한 변화는 없습니다."

def validate_inputs(df, rsi, macd, macd_signal, pattern, volume_comment):
    # 한글 주석: generate_ai_summary 호출 전 필수 값/컬럼 검증
    if df is None or len(df) < 2:
        raise ValueError("df가 비었거나 데이터가 2행 미만입니다.")
    required_cols = {"date", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {missing}")
    for name, val in [("rsi", rsi), ("macd", macd), ("macd_signal", macd_signal), ("pattern", pattern), ("volume_comment", volume_comment)]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
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

# ======================= 시각화용 타임시리즈 유틸 =======================

def _macd_series(prices, short=12, long=26, signal=9):
    """MACD 전체 시리즈(라인/시그널)"""
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema  = prices.ewm(span=long,  adjust=False).mean()
    macd = short_ema - long_ema
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def build_viz_timeseries(df, limit=120):
    """
    프론트 시각화를 위한 지표 타임시리즈 묶음 생성
    반환: {dates, rsi, macd, macdSignal, volume, atr14, pctb, obv, mfi14}
    """
    if df is None or len(df) < 2:
        return {}
    d = df.tail(limit).reset_index(drop=True).copy()
    s_close = d["close"].astype(float)
    s_high  = d["high"].astype(float)
    s_low   = d["low"].astype(float)
    s_vol   = d["volume"].astype(float)

    # RSI(14)
    # rsi_series = calculate_rsi_series(s_close, period=14).fillna(method="bfill").fillna(50)
    rsi_series = calculate_rsi_series(s_close, period=14).bfill().fillna(50)

    # MACD
    macd_line, macd_sig = _macd_series(s_close)

    # ATR(14)
    trs = [np.nan]
    for i in range(1, len(d)):
        trs.append(max(s_high.iloc[i]-s_low.iloc[i], abs(s_high.iloc[i]-s_close.iloc[i-1]), abs(s_low.iloc[i]-s_close.iloc[i-1])))
    atr14 = pd.Series(trs).rolling(14).mean()

    # Bollinger %b
    win = 20 if len(s_close) >= 20 else len(s_close)
    ma20 = s_close.rolling(win).mean()
    sd20 = s_close.rolling(win).std()
    upper = ma20 + 2*sd20
    lower = ma20 - 2*sd20
    rng = (upper - lower).replace(0, np.nan)
    pctb = ((s_close - lower) / (rng + 1e-8)).clip(0, 1)

    # OBV / MFI(14)
    obv = (np.sign(s_close.diff().fillna(0)) * s_vol).cumsum()
    tp = (s_high + s_low + s_close) / 3.0
    raw_money = tp * s_vol
    pos_flow = np.where(tp.diff() > 0, raw_money, 0.0)
    neg_flow = np.where(tp.diff() < 0, raw_money, 0.0)
    mfi = 100 - (100 / (1 + (pd.Series(pos_flow).rolling(14).sum() / (pd.Series(neg_flow).rolling(14).sum() + 1e-8))))

    return {
        "dates": d["date"].astype(str).tolist(),
        "rsi": rsi_series.round(2).tolist(),
        "macd": macd_line.round(5).fillna(0).tolist(),
        "macdSignal": macd_sig.round(5).fillna(0).tolist(),
        "volume": s_vol.round(4).tolist(),
        "atr14": atr14.round(6).fillna(0).tolist(),
        "pctb": pctb.round(4).fillna(0.5).tolist(),
        "obv": pd.Series(obv).round(4).fillna(0).tolist(),
        "mfi14": pd.Series(mfi).round(2).fillna(50).tolist(),
    }

# ======================= /시각화용 타임시리즈 유틸 끝 =======================

# === 시각화 확장: 지지/저항/트리거/타깃/EMA/BB/히트 ===
def build_viz_plus(df, k=0.3, target_atr=1.0, lookback=120):
    """
    한글 주석:
    - 과거 각 시점의 support/resistance(20D), ATR14로 bull/bear 트리거/타깃 시계열 생성
    - EMA12/26, Bollinger 상/하단도 함께 제공
    - backtest 윈도우 기준 hitBull/hitBear 플래그(0/1) 시각화용 생성
    """
    if df is None or len(df) < 30:
        return {}
    d = df.copy().reset_index(drop=True)
    s_o, s_h, s_l, s_c, s_v = [d[x].astype(float) for x in ["open","high","low","close","volume"]]

    # EMA12/26
    ema12 = s_c.ewm(span=12, adjust=False).mean()
    ema26 = s_c.ewm(span=26, adjust=False).mean()

    # BB(20)
    win = 20 if len(s_c) >= 20 else len(s_c)
    ma20 = s_c.rolling(win).mean()
    sd20 = s_c.rolling(win).std()
    bb_upper = ma20 + 2*sd20
    bb_lower = ma20 - 2*sd20

    # 롤링 S/R(20) + ATR14
    lows = s_l.rolling(20).min().shift(1)
    highs = s_h.rolling(20).max().shift(1)
    trs = [np.nan]
    for i in range(1, len(d)):
        trs.append(max(s_h.iloc[i]-s_l.iloc[i], abs(s_h.iloc[i]-s_c.iloc[i-1]), abs(s_l.iloc[i]-s_c.iloc[i-1])))
    atr14 = pd.Series(trs).rolling(14).mean()

    bull_trg = highs + k*atr14
    bear_trg = lows  - k*atr14
    bull_tgt = bull_trg + target_atr*atr14
    bear_tgt = bear_trg - target_atr*atr14

    # 백테스트 윈도우(환경변수) hit 플래그 (미래 봉 참조)
    horizon = BT_HORIZON_DAYS
    hit_bull = [0]*len(d)
    hit_bear = [0]*len(d)
    last_i = len(d) - horizon - 1
    for i in range(25, last_i):
        if any(np.isnan([bull_trg.iloc[i], bear_trg.iloc[i], atr14.iloc[i]])):
            continue
        hi = float(s_h.iloc[i+1:i+1+horizon].max())
        lo = float(s_l.iloc[i+1:i+1+horizon].min())
        if hi >= float(bull_tgt.iloc[i]): hit_bull[i] = 1
        if lo <= float(bear_tgt.iloc[i]): hit_bear[i] = 1

    # tail(lookback)만 반환 (프론트 렌더 최적화)
    t = -lookback if len(d) > lookback else 0
    def _lst(x):
        return (
            x.iloc[t:]
            .round(6)
            .bfill()
            .ffill()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .tolist()
        )

    return {
        "dates": d["date"].astype(str).iloc[t:].tolist(),
        "ema12": _lst(ema12), "ema26": _lst(ema26),
        "bbUpper": _lst(bb_upper), "bbLower": _lst(bb_lower),
        "support20": _lst(lows), "resistance20": _lst(highs),
        "bullTrigger": _lst(bull_trg), "bearTrigger": _lst(bear_trg),
        "bullTarget": _lst(bull_tgt), "bearTarget": _lst(bear_tgt),
        "hitBull": hit_bull[t:] if t != 0 else hit_bull,
        "hitBear": hit_bear[t:] if t != 0 else hit_bear,
    }

# ======================= 고급 지표/다이버전스/점수 유틸 =======================
def _true_range(h, l, c_prev):
    return max(h - l, abs(h - c_prev), abs(l - c_prev))

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calculate_rsi_series(prices, period=14):
    """한글 주석: RSI 전체 시리즈(마지막 값만이 아니라) 반환"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_rsi_divergence(prices, rsi_series, lookback=20):
    """
    한글 주석: 약식 RSI 다이버전스 탐지
    - Bullish: 가격 저점 하락, RSI 저점 상승
    - Bearish: 가격 고점 상승, RSI 고점 하락
    """
    p = prices.tail(lookback).reset_index(drop=True)
    r = rsi_series.tail(lookback).reset_index(drop=True)
    if len(p) < 3 or len(r) < 3:
        return "none"
    p_low1 = float(p.min()); p_low2 = float(p.iloc[-1])
    r_low1 = float(r.min()); r_low2 = float(r.iloc[-1] if not np.isnan(r.iloc[-1]) else 50.0)
    p_high1 = float(p.max()); p_high2 = float(p.iloc[-1])
    r_high1 = float(r.max()); r_high2 = float(r.iloc[-1] if not np.isnan(r.iloc[-1]) else 50.0)
    bullish = (p_low2 < p_low1) and (r_low2 > r_low1)
    bearish = (p_high2 > p_high1) and (r_high2 < r_high1)
    if bullish and not bearish: return "bullish"
    if bearish and not bullish: return "bearish"
    return "none"

def compute_advanced_features_v2(df):
    """
    한글 주석:
    - 기존 features + ADX/DI, MFI, OBV, Stoch, 볼린저 squeeze, RSI 다이버전스
    - NaN/데이터 부족 방어
    """
    s_close = df["close"].astype(float)
    s_high  = df["high"].astype(float)
    s_low   = df["low"].astype(float)
    s_vol   = df["volume"].astype(float)

    # EMA 12/26
    ema12 = s_close.ewm(span=12, adjust=False).mean()
    ema26 = s_close.ewm(span=26, adjust=False).mean()
    ema_spread = (ema12 - ema26).iloc[-1]
    ema_cross = "golden" if ema12.iloc[-1] > ema26.iloc[-1] else "death"

    # 볼린저(20)
    win = 20 if len(s_close) >= 20 else len(s_close)
    ma20 = s_close.rolling(win).mean()
    sd20 = s_close.rolling(win).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    rng = (upper.iloc[-1] - lower.iloc[-1]) if win >= 2 else np.nan
    if rng and not np.isnan(rng) and rng != 0:
        bb_pos = round(float((s_close.iloc[-1] - lower.iloc[-1]) / rng * 100), 2)
    else:
        bb_pos = 50.0

    # ATR(14) / 변동성 레짐
    trs = []
    for i in range(1, len(df)):
        trs.append(_true_range(s_high.iloc[i], s_low.iloc[i], s_close.iloc[i-1]))
    atr14 = float(pd.Series(trs).rolling(14).mean().iloc[-1]) if len(trs) >= 14 else float(np.mean(trs) if trs else 0.0)
    price = float(s_close.iloc[-1]) if len(s_close) else 0.0
    vol_regime_ratio = (atr14 / price) if price else 0.0
    vol_regime = "고변동" if vol_regime_ratio >= 0.08 else ("중간" if vol_regime_ratio >= 0.04 else "저변동")

    # 모멘텀
    def _ret(n):
        return round(float((s_close.iloc[-1] / s_close.iloc[-n-1] - 1) * 100), 2) if len(s_close) > n and s_close.iloc[-n-1] != 0 else 0.0
    mom7, mom14, mom28 = _ret(7), _ret(14), _ret(28)

    # 거래량 퍼센타일(30D)
    vol_tail = s_vol.tail(30) if len(s_vol) >= 30 else s_vol
    if len(vol_tail) >= 2:
        sorted_vals = np.sort(vol_tail.values)
        vol_percentile_30d = round(float(np.searchsorted(sorted_vals, s_vol.iloc[-1]) / max(1, len(sorted_vals) - 1) * 100), 2)
    else:
        vol_percentile_30d = 50.0

    # 지지/저항 (20D)
    lookback = df.tail(20) if len(df) >= 20 else df
    support_20d = float(lookback["low"].min()) if len(lookback) else price
    resistance_20d = float(lookback["high"].max()) if len(lookback) else price

    # ADX / +DI / -DI (Wilder 간소화)
    up_move = s_high.diff()
    down_move = -s_low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.Series([_true_range(s_high.iloc[i], s_low.iloc[i], s_close.iloc[i-1]) for i in range(1, len(df))])
    tr = pd.concat([pd.Series([np.nan]), tr], ignore_index=True)
    atr_14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / (atr_14 + 1e-8)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / (atr_14 + 1e-8)
    dx = (100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8))).rolling(14).mean()
    adx14 = float(dx.iloc[-1]) if not np.isnan(dx.iloc[-1]) else 0.0
    plus_di_last = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0
    minus_di_last = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0

    # MFI(14)
    tp = (s_high + s_low + s_close) / 3.0
    raw_money = tp * s_vol
    pos_flow = np.where(tp.diff() > 0, raw_money, 0.0)
    neg_flow = np.where(tp.diff() < 0, raw_money, 0.0)
    mfi = 100 - (100 / (1 + (pd.Series(pos_flow).rolling(14).sum() / (pd.Series(neg_flow).rolling(14).sum() + 1e-8))))
    mfi14 = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50.0

    # OBV
    obv = (np.sign(s_close.diff().fillna(0)) * s_vol).cumsum()
    obv_slope = float(obv.tail(5).diff().mean()) if len(obv) >= 5 else 0.0

    # Stochastic (14,3)
    lowest14 = s_low.rolling(14).min()
    highest14 = s_high.rolling(14).max()
    stoch_k = 100 * (s_close - lowest14) / (highest14 - lowest14 + 1e-8)
    stoch_d = stoch_k.rolling(3).mean()
    stoch_k_last = float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else 50.0
    stoch_d_last = float(stoch_d.iloc[-1]) if not np.isnan(stoch_d.iloc[-1]) else 50.0

    # Squeeze (Bollinger Bandwidth vs Keltner)
    ema20 = s_close.ewm(span=20, adjust=False).mean()
    bandwidth = float(((upper - lower) / (ma20 + 1e-8)).iloc[-1]) if win >= 2 else np.nan
    kelt_upper = ema20 + 1.5 * atr14
    kelt_lower = ema20 - 1.5 * atr14
    squeeze_on = bool((upper.iloc[-1] < kelt_upper.iloc[-1]) and (lower.iloc[-1] > kelt_lower.iloc[-1])) if win >= 2 else False

    # RSI 다이버전스
    rsi_series = calculate_rsi_series(s_close, period=14)
    rsi_div = detect_rsi_divergence(s_close, rsi_series, lookback=20)

    # 추가 통계/멀티타임프레임 계산
    bw_stats = compute_bandwidth_stats(df, lookback=120)
    trend_reg = compute_trend_regression(df, window=20)
    pivots = compute_pivot_levels(latest_high=s_high.iloc[-1], latest_low=s_low.iloc[-1], latest_close=s_close.iloc[-1])
    wtrend = compute_weekly_trend(df)

    # squeeze 강도(정규화): Bandwidth / (ATR/가격)
    squeeze_strength = None
    try:
        ratio = (upper.iloc[-1] - lower.iloc[-1]) / (ma20.iloc[-1] + 1e-8)
        vol_norm = (atr14 / (price + 1e-8)) if price else None
        squeeze_strength = round(float(ratio / (vol_norm + 1e-8)), 4) if vol_norm else None
    except Exception:
        squeeze_strength = None

    # 변동성 조정 모멘텀 및 타임프레임 충돌
    vam20 = compute_vol_adj_momentum(df, ma_win=20)
    tf_conflict = compute_timeframe_conflict({"ema_cross": ema_cross, "weekly_trend": wtrend.get("weekly_trend")})
    return {
        "ema12": float(ema12.iloc[-1]), "ema26": float(ema26.iloc[-1]),
        "ema_spread": float(ema_spread), "ema_cross": ema_cross,
        "bb_pos_pct": float(bb_pos), "atr14": round(float(atr14), 4),
        "vol_regime": vol_regime, "vol_regime_ratio": round(float(vol_regime_ratio), 4),
        "mom7_pct": mom7, "mom14_pct": mom14, "mom28_pct": mom28,
        "vol_percentile_30d": vol_percentile_30d,
        "support_20d": round(float(support_20d), 4), "resistance_20d": round(float(resistance_20d), 4),
        "adx14": round(adx14, 2), "plus_di": round(plus_di_last, 2), "minus_di": round(minus_di_last, 2),
        "mfi14": round(mfi14, 2),
        "obv_slope5": round(obv_slope, 4),
        "stoch_k": round(stoch_k_last, 2), "stoch_d": round(stoch_d_last, 2),
        "bb_bandwidth": round(float(bandwidth), 6) if bandwidth == bandwidth else None,
        "squeeze_on": squeeze_on,
        "rsi_divergence": rsi_div,
        # --- 추가 출력 ---
        "bb_bandwidth_pctile_120d": bw_stats.get("bb_bandwidth_pctile_120d"),
        "bb_pctb": bw_stats.get("bb_pctb"),
        "squeeze_strength": squeeze_strength,
        "trend_slope_pct_per_day": trend_reg.get("trend_slope_pct_per_day"),
        "trend_r2": trend_reg.get("trend_r2"),
        "pivot_P": pivots.get("pivot_P"),
        "pivot_R1": pivots.get("pivot_R1"),
        "pivot_S1": pivots.get("pivot_S1"),
        "pivot_R2": pivots.get("pivot_R2"),
        "pivot_S2": pivots.get("pivot_S2"),
        "weekly_trend": wtrend.get("weekly_trend"),
        "vam20": round(float(vam20), 4),
        "timeframe_conflict": bool(tf_conflict),
    }

# ======================= 신뢰/난이도/유동성 유틸 =======================

def compute_data_fresh_minutes(df):
    """한글 주석: 최신 캔들의 날짜(KST 자정 기준)와 현재(KST) 시각 차이를 분 단위로 계산."""
    try:
        latest_str = str(df["date"].iloc[-1])  # 'YYYY-MM-DD'
        latest_dt = datetime.strptime(latest_str, "%Y-%m-%d").replace(tzinfo=timezone(timedelta(hours=9)))
        now_kst = (datetime.utcnow() + timedelta(hours=9)).replace(tzinfo=timezone(timedelta(hours=9)))
        return int((now_kst - latest_dt).total_seconds() // 60)
    except Exception:
        return None

def compute_trigger_difficulty(df, auto_triggers):
    """한글 주석: 현재 종가 기준 트리거까지 거리(%), 최근 5일 근접도(%)"""
    try:
        close = float(df["close"].astype(float).iloc[-1])
        bull = (auto_triggers or {}).get("bull_trigger")
        bear = (auto_triggers or {}).get("bear_trigger")
        dist_bull = None if bull is None or close == 0 else round((bull - close) / close * 100.0, 2)
        dist_bear = None if bear is None or close == 0 else round((close - bear) / close * 100.0, 2)
        tail = df.tail(6)
        highs = tail["high"].astype(float).iloc[1:] if len(tail) >= 2 else pd.Series([])
        lows  = tail["low"].astype(float).iloc[1:] if len(tail) >= 2 else pd.Series([])
        def _near_pct(vals, target):
            if target is None or len(vals) == 0:
                return None
            diffs = [abs(v - target) / abs(target) * 100.0 for v in vals if target != 0]
            return None if not diffs else round(float(min(diffs)), 2)
        near_bull = _near_pct(highs, bull)
        near_bear = _near_pct(lows, bear)
        return {
            "distToBullPct": dist_bull,
            "distToBearPct": dist_bear,
            "nearBullPctMin": near_bull,
            "nearBearPctMin": near_bear,
        }
    except Exception:
        return {"distToBullPct": None, "distToBearPct": None}

def compute_liquidity_flag(ticker24h: dict, symbol: str, min_turnover_krw: float) -> dict:
    """한글 주석: 24h 거래대금 하한 충족 여부 및 실제 거래대금 표시"""
    tv = float((ticker24h or {}).get(symbol, {}).get("acc_trade_price_24h", 0.0))
    return {
        "liquidityOk": bool(tv >= float(min_turnover_krw)),
        "turnover24hKrw": tv,
        "minTurnoverKrw": float(min_turnover_krw),
    }

def build_provenance(df, features, scores, backtest, auto_triggers, latency_ms, ticker24h, symbol: str) -> dict:
    """한글 주석: 데이터 출처/샘플/파라미터/신뢰 지표 메타."""
    try:
        fresh_min = compute_data_fresh_minutes(df)
    except Exception:
        fresh_min = None
    return {
        "schemaVersion": "1.2.0",
        "dataSource": "Upbit REST v1",
        "market": f"KRW-{symbol}",
        "candleCount": int(len(df) if df is not None else 0),
        "windows": {
            "rsi": 14, "macd": [12,26,9], "stoch": [14,3], "bb": 20, "atr": 14,
            "sr_lookback_days": 20, "trend_regression": 20, "bandwidth_pctile": 120
        },
        "backtestParams": {
            "k": float((auto_triggers or {}).get("k", 0.3)),
            "horizonDays": int(os.getenv("BT_HORIZON_DAYS", "3")),
            "targetAtr": float(os.getenv("BT_TARGET_ATR", "1.0"))
        },
        "backtestSamples": int((backtest or {}).get("samples", 0)),
        "latencyMs": int(latency_ms) if latency_ms is not None else None,
        "dataFreshMinutes": fresh_min,
        "liquidity": compute_liquidity_flag(ticker24h, symbol, MIN_TURNOVER_KRW),
        "validation": {
            "errors": [],
            "featuresPresent": bool(features),
        }
    }
# ======================= /신뢰/난이도/유동성 유틸 끝 =======================

# ======================= 실전 보강: 변동성 조정 모멘텀/롤링 지지저항/백테스트 =======================
def compute_vol_adj_momentum(df, ma_win=20):
    """
    한글 주석: 변동성 조정 모멘텀(VAM)
    - 정의: (Close/MA(ma_win) - 1) / (ATR(14)/가격)
    - 해석: 같은 수익률이라도 변동성이 낮을수록 점수↑
    """
    s_close = df["close"].astype(float)
    s_high  = df["high"].astype(float)
    s_low   = df["low"].astype(float)
    ma = s_close.rolling(ma_win).mean()
    trs = []
    for i in range(1, len(df)):
        trs.append(max(s_high.iloc[i]-s_low.iloc[i], abs(s_high.iloc[i]-s_close.iloc[i-1]), abs(s_low.iloc[i]-s_close.iloc[i-1])))
    atr14 = pd.Series(trs).rolling(14).mean()
    atr14 = pd.concat([pd.Series([np.nan]), atr14], ignore_index=True)
    price = s_close
    denom = (atr14 / (price + 1e-8))
    raw = (s_close / (ma + 1e-8) - 1.0) / (denom + 1e-8)
    return float(raw.iloc[-1]) if len(raw) and not np.isnan(raw.iloc[-1]) else 0.0

def _rolling_support_resistance(df, window=20):
    """
    한글 주석: 롤링 지지/저항 시계열 (각 시점에서 직전 window 구간 기준)
    """
    lows = df["low"].astype(float).rolling(window).min()
    highs = df["high"].astype(float).rolling(window).max()
    # 현재 봉 포함을 피하기 위해 한 칸 시프트(보수적)
    support_sr = lows.shift(1)
    resistance_sr = highs.shift(1)
    return support_sr, resistance_sr

def backtest_auto_triggers(df, k=0.3, horizon_days=3, target_atr=1.0):
    """
    한글 주석: 자동 트리거(상방=저항+k*ATR, 하방=지지-k*ATR)의 과거 적중률 간단 검증
    - 각 일자 i에서 직전 20일로 support/resistance, ATR14 계산
    - 향후 horizon_days 내 high가 (bull_trigger + target_atr*ATR) 이상 도달 → bull_hit
    - 향후 horizon_days 내 low가 (bear_trigger - target_atr*ATR) 이하 도달 → bear_hit
    반환: dict(hitRate_bull, hitRate_bear, hitRate_any, samples)
    """
    if len(df) < 25:
        return {"hitRate_bull": None, "hitRate_bear": None, "hitRate_any": None, "samples": 0}
    s_high  = df["high"].astype(float).reset_index(drop=True)
    s_low   = df["low"].astype(float).reset_index(drop=True)
    sup_sr, res_sr = _rolling_support_resistance(df, window=20)
    sup_sr = sup_sr.reset_index(drop=True)
    res_sr = res_sr.reset_index(drop=True)
    # ATR14 시계열
    trs = [np.nan]
    for i in range(1, len(df)):
        trs.append(max(df["high"].iloc[i]-df["low"].iloc[i], abs(df["high"].iloc[i]-df["close"].iloc[i-1]), abs(df["low"].iloc[i]-df["close"].iloc[i-1])))
    atr14_sr = pd.Series(trs).rolling(14).mean().reset_index(drop=True)
    bulls = 0; bears = 0; either = 0; n = 0
    last_i = len(df) - horizon_days - 1
    for i in range(25, last_i):
        sup = sup_sr.iloc[i]; res = res_sr.iloc[i]; atr = atr14_sr.iloc[i]
        if np.isnan(sup) or np.isnan(res) or np.isnan(atr) or atr <= 0:
            continue
        bull_trigger = res + k * atr
        bear_trigger = sup - k * atr
        bull_target = bull_trigger + target_atr * atr
        bear_target = bear_trigger - target_atr * atr
        hi = float(s_high.iloc[i+1:i+1+horizon_days].max()) if i+1 < len(df) else np.nan
        lo = float(s_low.iloc[i+1:i+1+horizon_days].min()) if i+1 < len(df) else np.nan
        hit_bull = (not np.isnan(hi)) and (hi >= bull_target)
        hit_bear = (not np.isnan(lo)) and (lo <= bear_target)
        if hit_bull or hit_bear:
            either += 1
        if hit_bull:
            bulls += 1
        if hit_bear:
            bears += 1
        n += 1
    if n == 0:
        return {"hitRate_bull": None, "hitRate_bear": None, "hitRate_any": None, "samples": 0}
    return {
        "hitRate_bull": round(bulls / n * 100.0, 2),
        "hitRate_bear": round(bears / n * 100.0, 2),
        "hitRate_any": round(either / n * 100.0, 2),
        "samples": n
    }

def _wilson_interval(p, n, z=1.96):
    if n == 0:
        return (None, None)
    phat = max(0.0, min(1.0, p))
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    margin = z*np.sqrt((phat*(1-phat)+z*z/(4*n))/n)
    lower = (centre - margin)/denom
    upper = (centre + margin)/denom
    return (round(lower*100, 2), round(upper*100, 2))

def compute_backtest_detailed(df, k=0.3, horizon_days=3, target_atr=1.0):
    """
    레짐(저/중/고)별 적중률과 Wilson 95% CI, 시각화용 시리즈를 함께 반환
    반환:
      overall: {hitRate_bull, hitRate_bear, hitRate_any, samples, ci_any: [lo,hi]}
      byRegime: {저변동|중간|고변동: {...}}
      series: {index:[], bullHit:[], bearHit:[], bullTrigger:[], bearTrigger:[]}
    """
    if len(df) < 25:
        return {"overall": {"hitRate_bull": None, "hitRate_bear": None, "hitRate_any": None, "samples": 0, "ci_any": [None, None]},
                "byRegime": {}, "series": {}}
    s_close = df["close"].astype(float).reset_index(drop=True)
    s_high  = df["high"].astype(float).reset_index(drop=True)
    s_low   = df["low"].astype(float).reset_index(drop=True)

    sup_sr, res_sr = _rolling_support_resistance(df, window=20)
    trs = [np.nan]
    for i in range(1, len(df)):
        trs.append(max(df["high"].iloc[i]-df["low"].iloc[i], abs(df["high"].iloc[i]-df["close"].iloc[i-1]), abs(df["low"].iloc[i]-df["close"].iloc[i-1])))
    atr14_sr = pd.Series(trs).rolling(14).mean().reset_index(drop=True)

    price_sr = s_close
    vr = (atr14_sr / (price_sr + 1e-8)).fillna(0)
    def _regime(v):
        return "고변동" if v >= 0.08 else ("중간" if v >= 0.04 else "저변동")

    bulls=bears=either=n=0
    by = {"저변동": {"bull":0,"bear":0,"either":0,"n":0},
          "중간":   {"bull":0,"bear":0,"either":0,"n":0},
          "고변동": {"bull":0,"bear":0,"either":0,"n":0}}
    ser_idx=[]; ser_bull=[]; ser_bear=[]; ser_bt=[]; ser_brt=[]

    last_i = len(df) - horizon_days - 1
    for i in range(25, last_i):
        sup = sup_sr.iloc[i]; res = res_sr.iloc[i]; atr = atr14_sr.iloc[i]
        if np.isnan(sup) or np.isnan(res) or np.isnan(atr) or atr <= 0:
            continue
        bull_trigger = res + k * atr
        bear_trigger = sup - k * atr
        bull_target = bull_trigger + target_atr * atr
        bear_target = bear_trigger - target_atr * atr
        hi = float(s_high.iloc[i+1:i+1+horizon_days].max()) if i+1 < len(df) else np.nan
        lo = float(s_low.iloc[i+1:i+1+horizon_days].min()) if i+1 < len(df) else np.nan
        hit_bull = (not np.isnan(hi)) and (hi >= bull_target)
        hit_bear = (not np.isnan(lo)) and (lo <= bear_target)

        reg = _regime(vr.iloc[i] if i < len(vr) else 0)
        by[reg]["n"] += 1
        by[reg]["either"] += 1 if (hit_bull or hit_bear) else 0
        by[reg]["bull"] += 1 if hit_bull else 0
        by[reg]["bear"] += 1 if hit_bear else 0

        n += 1; either += 1 if (hit_bull or hit_bear) else 0
        bulls += 1 if hit_bull else 0
        bears += 1 if hit_bear else 0

        ser_idx.append(i)
        ser_bull.append(1 if hit_bull else 0)
        ser_bear.append(1 if hit_bear else 0)
        ser_bt.append(round(float(bull_trigger), 6))
        ser_brt.append(round(float(bear_trigger), 6))

    overall = {"hitRate_bull": round(bulls/max(1,n)*100.0,2),
               "hitRate_bear": round(bears/max(1,n)*100.0,2),
               "hitRate_any": round(either/max(1,n)*100.0,2),
               "samples": n}
    lo, hi = _wilson_interval(either/max(1,n), n) if n>0 else (None,None)
    overall["ci_any"] = [lo, hi]

    by_out = {}
    for kreg, v in by.items():
        if v["n"] == 0: continue
        p_any = v["either"]/v["n"]
        lo2, hi2 = _wilson_interval(p_any, v["n"]) if v["n"]>0 else (None, None)
        by_out[kreg] = {"hitRate_bull": round(v["bull"]/v["n"]*100.0,2),
                        "hitRate_bear": round(v["bear"]/v["n"]*100.0,2),
                        "hitRate_any": round(p_any*100.0,2),
                        "samples": v["n"],
                        "ci_any": [lo2, hi2]}

    return {"overall": overall,
            "byRegime": by_out,
            "series": {"index": ser_idx, "bullHit": ser_bull, "bearHit": ser_bear,
                       "bullTrigger": ser_bt, "bearTrigger": ser_brt}}

def backtest_trades(df, k=0.3, horizon_days=3, target_atr=1.0, fee_bps=10):
    """
    한글 주석: 각 시점에서 상/하방 트리거를 가정한 간단 트레이드 리스트 생성
    - 목표/손절/타임아웃 규칙으로 청산하여 retPct/MAE/MFE/보유기간 기록
    - 수수료 왕복 fee_bps(기본 10bp=0.10%) 반영
    반환: {trades:[], equityCurve:{dates, equity}, stats:{winRate, avgWinPct, avgLossPct, expPct, profitFactor, maxDDPct}}
    """
    if df is None or len(df) < 30:
        return {"trades": [], "equityCurve": {"dates": [], "equity": []}, "stats": {}}

    d = df.reset_index(drop=True).copy()
    s_h = d["high"].astype(float); s_l = d["low"].astype(float); s_c = d["close"].astype(float)

    lows = d["low"].astype(float).rolling(20).min().shift(1)
    highs = d["high"].astype(float).rolling(20).max().shift(1)
    trs = [np.nan]
    for i in range(1, len(d)):
        trs.append(max(d["high"].iloc[i]-d["low"].iloc[i], abs(d["high"].iloc[i]-d["close"].iloc[i-1]), abs(d["low"].iloc[i]-d["close"].iloc[i-1])))
    atr14 = pd.Series(trs).rolling(14).mean()

    trades = []
    eq = [1.0]
    eq_dates = [d["date"].iloc[0]]
    fee = fee_bps/10000.0

    last_i = len(d) - horizon_days - 1
    for i in range(25, last_i):
        sup = float(lows.iloc[i]) if not np.isnan(lows.iloc[i]) else None
        res = float(highs.iloc[i]) if not np.isnan(highs.iloc[i]) else None
        atr = float(atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) else None
        if sup is None or res is None or atr is None or atr <= 0:
            continue
        entry = float(s_c.iloc[i])
        bull_trigger = res + k*atr
        bear_trigger = sup - k*atr
        bull_target = bull_trigger + target_atr*atr
        bear_target = bear_trigger - target_atr*atr
        hi = float(s_h.iloc[i+1:i+1+horizon_days].max())
        lo = float(s_l.iloc[i+1:i+1+horizon_days].min())

        def _simulate(direction):
            if direction == "bull":
                hit_target = hi >= bull_target
                hit_stop   = lo <= bear_trigger
                reason = "timeout"; exitp = float(s_c.iloc[i+horizon_days])
                if hit_target: reason = "target"; exitp = bull_target
                elif hit_stop: reason = "stop";   exitp = bear_trigger
                ret = (exitp/entry - 1) - 2*fee
                mfe = (max(hi, entry)/entry - 1) if entry>0 else 0.0
                mae = (min(lo, entry)/entry - 1) if entry>0 else 0.0
                return ret*100.0, reason, mfe*100.0, mae*100.0
            else:
                hit_target = lo <= bear_target
                hit_stop   = hi >= bull_trigger
                reason = "timeout"; exitp = float(s_c.iloc[i+horizon_days])
                if hit_target: reason = "target"; exitp = bear_target
                elif hit_stop: reason = "stop";   exitp = bull_trigger
                ret = (entry/exitp - 1) - 2*fee
                mfe = (entry/min(lo, entry) - 1) if entry>0 else 0.0
                mae = (entry/max(hi, entry) - 1) if entry>0 else 0.0
                return ret*100.0, reason, mfe*100.0, mae*100.0

        for direction in ("bull","bear"):
            retPct, reason, mfePct, maePct = _simulate(direction)
            trades.append({
                "date": d["date"].iloc[i],
                "direction": direction,
                "entry": round(entry, 6),
                "retPct": round(retPct, 2),
                "holdDays": horizon_days,
                "reason": reason,
                "mfePct": round(mfePct, 2),
                "maePct": round(maePct, 2),
            })
            eq.append(eq[-1]*(1 + retPct/100.0))
            eq_dates.append(d["date"].iloc[i+horizon_days])

    wins = [t for t in trades if t["retPct"] > 0]
    losses = [t for t in trades if t["retPct"] <= 0]
    winRate = round(len(wins)/max(1,len(trades))*100.0, 2) if trades else None
    avgWin = round(np.mean([t["retPct"] for t in wins]), 2) if wins else None
    avgLoss = round(np.mean([t["retPct"] for t in losses]), 2) if losses else None
    expPct = round(np.mean([t["retPct"] for t in trades]), 2) if trades else None
    pf = round(abs((np.sum([t["retPct"] for t in wins]) / (np.sum([abs(t['retPct']) for t in losses]) + 1e-8))), 2) if losses else None
    cur_max = -1e9; dd = []
    for val in eq:
        cur_max = max(cur_max, val)
        dd.append((val/cur_max - 1))
    maxDDPct = round(min(dd)*100.0, 2) if dd else None

    stats = {"winRate": winRate, "avgWinPct": avgWin, "avgLossPct": avgLoss,
             "expPct": expPct, "profitFactor": pf, "maxDDPct": maxDDPct}
    return {"trades": trades,
            "equityCurve": {"dates": eq_dates, "equity": [round(x,6) for x in eq]},
            "stats": stats}

def compute_timeframe_conflict(features):
    """
    한글 주석: 데일리(ema_cross)와 주간(weekly_trend)의 방향이 충돌하면 True
    """
    daily = features.get("ema_cross")
    weekly = features.get("weekly_trend")
    if not daily or not weekly:
        return False
    return (daily == "golden" and weekly == "death") or (daily == "death" and weekly == "golden")
# ======================= /실전 보강 유틸 끝 =======================

def compute_rule_based_scores(features, rsi_value):
    """
    한글 주석: 규칙 기반 점수(0~100) 산출
    - TrendScore: ADX>20, ema_cross=golden, +DI>-DI
    - MomentumScore: mom7>0, mom14>0, RSI 45~65
    - VolumeScore: MFI>50, OBV_slope>0, vol_percentile_30d>60
    - Risk: squeeze_on이면 변동성 급팽창 리스크
    """
    trend = 0
    trend += 1 if features.get("adx14", 0) > 20 else 0
    trend += 1 if features.get("ema_cross") == "golden" else 0
    trend += 1 if features.get("plus_di", 0) > features.get("minus_di", 0) else 0

    momentum = 0
    momentum += 1 if features.get("mom7_pct", 0) > 0 else 0
    momentum += 1 if features.get("mom14_pct", 0) > 0 else 0
    momentum += 1 if 45 <= float(rsi_value) <= 65 else 0

    volume = 0
    volume += 1 if features.get("mfi14", 50) > 50 else 0
    volume += 1 if features.get("obv_slope5", 0) > 0 else 0
    volume += 1 if features.get("vol_percentile_30d", 50) > 60 else 0

    risk = 1 if features.get("squeeze_on") else 0

    # 가중합 (Trend:4, Momentum:3, Volume:3 → 총 10점, risk는 감점 0~1)
    raw = (trend*4 + momentum*3 + volume*3)  # 0..(3*4 + 3*3 + 3*3) = 0..30
    score = int(round(raw / 30 * 100))
    score = max(0, min(100, score - (risk*5)))
    return {
        "trendScore": trend*33,
        "momentumScore": momentum*33,
        "volumeScore": volume*33,
        "riskFlags": {"squeeze_on": bool(features.get("squeeze_on"))},
        "ruleConfidence": score
    }

def compute_auto_triggers(features, k=0.3):
    """
    한글 주석: 기계적 트리거 레벨 산출 (상방/하방)
    - 상방: resistance_20d + k * ATR(14)
    - 하방: support_20d - k * ATR(14)
    """
    atr = float(features.get("atr14") or 0.0)
    sup = float(features.get("support_20d") or 0.0)
    res = float(features.get("resistance_20d") or 0.0)
    return {
        "bull_trigger": round(res + k * atr, 4),
        "bear_trigger": round(sup - k * atr, 4),
        "k": k
    }
# ======================= /고급 지표/다이버전스/점수 유틸 끝 =======================

# ======================= 추가 통계/멀티타임프레임 유틸 =======================
def _safe_percentile_rank(series, value):
    """한글 주석: value가 series 내에서 차지하는 백분위(0~100). 빈/단일값 방어."""
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return 50.0
    arr = np.sort(s.values)
    idx = np.searchsorted(arr, value, side="right") - 1
    pct = (idx / (len(arr) - 1)) * 100
    return round(float(max(0.0, min(100.0, pct))), 2)

def compute_bandwidth_stats(df, lookback=120):
    """
    한글 주석: 볼린저 밴드 폭(Bandwidth), %b, 그리고 과거 lookback 대비 백분위 계산
    - Bandwidth = (Upper-Lower)/MA
    - %b = (Close-Lower)/(Upper-Lower)
    """
    s_close = df["close"].astype(float)
    win = 20 if len(s_close) >= 20 else len(s_close)
    if win < 2:
        return {"bb_bandwidth_pctile": None, "bb_pctb": None}
    ma20 = s_close.rolling(win).mean()
    sd20 = s_close.rolling(win).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    bandwidth_series = (upper - lower) / (ma20 + 1e-8)
    bandwidth_series = bandwidth_series.tail(lookback)
    cur_bandwidth = float(bandwidth_series.iloc[-1]) if len(bandwidth_series) else None

    rng = (upper.iloc[-1] - lower.iloc[-1])
    pctb = None
    if rng and not np.isnan(rng) and rng != 0:
        pctb = float((s_close.iloc[-1] - lower.iloc[-1]) / rng)
        pctb = round(float(max(0.0, min(1.0, pctb))), 4)

    pctile = _safe_percentile_rank(bandwidth_series.dropna(), cur_bandwidth) if cur_bandwidth is not None else None
    return {
        "bb_bandwidth_pctile_120d": pctile,  # 0~100
        "bb_pctb": pctb                      # 0~1
    }

def compute_trend_regression(df, window=20):
    """
    한글 주석: 최근 window일 종가에 대해 선형회귀로 추세 기울기/설명력 산출
    - slope_pct_per_day: 하루당 % 변화량(기울기/마지막 종가)
    - r2: 선형 적합 결정계수
    """
    s_close = df["close"].astype(float).tail(window)
    if len(s_close) < 3:
        return {"trend_slope_pct_per_day": 0.0, "trend_r2": 0.0}
    x = np.arange(len(s_close))
    y = s_close.values
    A = np.vstack([x, np.ones(len(x))]).T
    # 최소제곱 직선 적합
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    # R^2 계산
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot
    # slope_pct_per_day = (m / (y[-1] + 1e-8)) * 100.0
    base = y[-1] if (len(y) and not np.isnan(y[-1]) and y[-1] != 0) else 1.0
    slope_pct_per_day = (m / (base + 1e-8)) * 100.0
    return {"trend_slope_pct_per_day": round(float(slope_pct_per_day), 4),
            "trend_r2": round(float(max(0.0, min(1.0, r2))), 4)}

def compute_pivot_levels(latest_high, latest_low, latest_close):
    """
    한글 주석: 클래식 피벗 포인트(P, R1/S1, R2/S2)
    - 참고: 단기 intraday보단 데일리에서 보조 레벨로 활용
    """
    H = float(latest_high); L = float(latest_low); C = float(latest_close)
    P = (H + L + C) / 3.0
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {
        "pivot_P": round(P, 4),
        "pivot_R1": round(R1, 4), "pivot_S1": round(S1, 4),
        "pivot_R2": round(R2, 4), "pivot_S2": round(S2, 4),
    }

def compute_weekly_trend(df):
    """
    한글 주석: 데일리 데이터로 주간 종가를 집계하여 EMA 12/26의 골든/데드 여부 산출
    """
    if "date" not in df:
        return {"weekly_trend": None}
    tmp = df.copy()
    # 한글 주석: 날짜를 KST로 로컬라이즈하여 주간 경계를 한국 시간 기준으로 맞춘다
    tmp["date_dt"] = pd.to_datetime(tmp["date"])
    if tmp["date_dt"].dt.tz is None:
        tmp["date_dt"] = tmp["date_dt"].dt.tz_localize("Asia/Seoul")
    else:
        tmp["date_dt"] = tmp["date_dt"].dt.tz_convert("Asia/Seoul")
    tmp = tmp.set_index("date_dt").sort_index()
    wk = tmp["close"].astype(float).resample("W-SUN").last()  # KST 기준 주간(일요일 종료)
    if len(wk) < 28:
        return {"weekly_trend": None}
    ema12w = wk.ewm(span=12, adjust=False).mean()
    ema26w = wk.ewm(span=26, adjust=False).mean()
    weekly_trend = "golden" if ema12w.iloc[-1] > ema26w.iloc[-1] else "death"
    return {"weekly_trend": weekly_trend}
# ======================= /추가 통계/멀티타임프레임 유틸 끝 =======================

# 하위 호환: 이전 코드에서 compute_advanced_features를 호출하는 경우 대비
def compute_advanced_features(df):
    return compute_advanced_features_v2(df)

# ======================= AI JSON 검증/수정 유틸 =======================
def _required(d, key, typ):
    return (key in d) and isinstance(d[key], typ)

def _is_number(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))

def validate_ai_json_against_schema(ai_json):
    """
    한글 주석: 모델이 반환한 JSON을 경량 스키마로 검증한다.
    반환: (is_valid: bool, errors: list[str])
    """
    errs = []
    if not isinstance(ai_json, dict):
        return False, ["root is not an object"]

    # 필수 키 체크
    req_keys = ["headline", "regime", "key_levels", "facts", "interpretation", "scenarios", "risks", "confidence"]
    for k in req_keys:
        if k not in ai_json:
            errs.append(f"missing key: {k}")

    # 타입/값 범위 체크
    if "headline" in ai_json and not isinstance(ai_json.get("headline"), str):
        errs.append("headline must be string")
    if "regime" in ai_json and not isinstance(ai_json.get("regime"), str):
        errs.append("regime must be string")
    if "key_levels" in ai_json:
        kl = ai_json.get("key_levels", {})
        if not isinstance(kl, dict):
            errs.append("key_levels must be object")
        else:
            sup = kl.get("support")
            res = kl.get("resistance")
            if not _is_number(sup): errs.append("key_levels.support must be number")
            if not _is_number(res): errs.append("key_levels.resistance must be number")
            if _is_number(sup) and _is_number(res) and sup >= res:
                errs.append("support should be less than resistance")
    if "facts" in ai_json and not isinstance(ai_json.get("facts"), list):
        errs.append("facts must be list")
    if "interpretation" in ai_json and not isinstance(ai_json.get("interpretation"), list):
        errs.append("interpretation must be list")
    if "risks" in ai_json and not isinstance(ai_json.get("risks"), list):
        errs.append("risks must be list")
    if "scenarios" in ai_json:
        sc = ai_json.get("scenarios", {})
        if not isinstance(sc, dict):
            errs.append("scenarios must be object")
        else:
            for side in ["bull", "bear"]:
                s = sc.get(side, {})
                if s:
                    if not isinstance(s, dict):
                        errs.append(f"scenarios.{side} must be object")
                    else:
                        if "trigger" in s and not isinstance(s.get("trigger"), str):
                            errs.append(f"scenarios.{side}.trigger must be string")
                        if "invalidate" in s and not isinstance(s.get("invalidate"), str):
                            errs.append(f"scenarios.{side}.invalidate must be string")
    if "confidence" in ai_json:
        c = ai_json.get("confidence")
        if not isinstance(c, int): errs.append("confidence must be integer")
        else:
            if c < 0 or c > 100: errs.append("confidence must be 0..100")

    # 선택(옵션) 필드 타입 점검 (있으면 타입만 확인)
    if "confidenceDrivers" in ai_json and not isinstance(ai_json.get("confidenceDrivers"), list):
        errs.append("confidenceDrivers must be list")
    if "unknowns_or_gaps" in ai_json and not isinstance(ai_json.get("unknowns_or_gaps"), list):
        errs.append("unknowns_or_gaps must be list")
    if "regime_breakdown" in ai_json and not isinstance(ai_json.get("regime_breakdown"), dict):
        errs.append("regime_breakdown must be object")
    if "liquidity" in ai_json and not isinstance(ai_json.get("liquidity"), dict):
        errs.append("liquidity must be object")
    if "risk_model" in ai_json and not isinstance(ai_json.get("risk_model"), dict):
        errs.append("risk_model must be object")
    if "scenarios" in ai_json and isinstance(ai_json.get("scenarios"), dict):
        det = ai_json["scenarios"].get("detail")
        if det is not None and not isinstance(det, list):
            errs.append("scenarios.detail must be list")

    return (len(errs) == 0), errs

def _repair_ai_json_with_defaults(ai_json, features):
    """
    한글 주석: 경미한 오류는 코드단에서 보정한다(보수적 기본값).
    - 누락된 key_levels은 20D 지지/저항으로 채움
    - confidence 범위 보정
    """
    if not isinstance(ai_json, dict):
        return ai_json
    kl = ai_json.get("key_levels")
    if not isinstance(kl, dict):
        ai_json["key_levels"] = {"support": features.get("support_20d"), "resistance": features.get("resistance_20d")}
    else:
        if not _is_number(kl.get("support")):
            ai_json["key_levels"]["support"] = features.get("support_20d")
        if not _is_number(kl.get("resistance")):
            ai_json["key_levels"]["resistance"] = features.get("resistance_20d")
    c = ai_json.get("confidence", 50)
    if isinstance(c, int):
        ai_json["confidence"] = max(0, min(100, c))
    else:
        ai_json["confidence"] = 50
    return ai_json

def _repair_ai_json_via_model(prompt, ai_json, errors):
    """
    한글 주석: 모델에게 '수정 지시'를 내려 JSON을 재생성하도록 요청.
    - prompt: 최초 [system,user] 메시지 리스트
    - ai_json: 최초 결과
    - errors: validate_ai_json_against_schema가 반환한 에러 목록
    """
    try:
        fix_instruction = {
            "role": "user",
            "content": {
                "action": "fix-json",
                "errors": errors,
                "instruction": "위 오류를 모두 해결해서 동일한 스키마(JSON)를 다시 출력해. 숫자 범위를 준수하고, key_levels.support < resistance 이어야 한다. 응답은 JSON만."
            }
        }
        new_messages = list(prompt) + [fix_instruction]
        return _call_openai_chat(new_messages)
    except Exception:
        return None
# ======================= /AI JSON 검증/수정 유틸 끝 =======================


def build_ai_prompt(df, rsi, macd, macd_signal, pattern, volume_comment, features, auto_triggers, scores, backtest, prompt_version="v2", symbol=None):
    change = compute_change_metrics(df)
    recent = df[["date", "open", "high", "low", "close", "volume"]].tail(7).to_dict("records")
    pattern_text = pattern if pattern and pattern != "none" else "감지됨 없음"

    system = {
        "role": "system",
        "content": (
            "너는 보수적인 한국어 크립토 애널리스트다. 투자 조언/매수·매도 지시는 절대 하지 말고, "
            "사실과 해석을 분리해라. 반드시 JSON으로만 응답하라."
        )
    }
    user = {
        "role": "user",
        "content": {
            "task": "하루 요약 리포트(JSON만 반환)",
            "promptVersion": prompt_version,
            "symbol": f"{(symbol or 'SYMBOL')}/KRW",
            "core_metrics": {
                "last_close": change["last_close"],
                "d1_return_pct": change["d1_return_pct"],
                "dev7_pct": change["dev7_pct"],
                "atr14": change["atr14"],
                "vol_z": change["vol_z"],
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "pattern": pattern_text
            },
            "advanced_features": features,
            "auto_triggers": auto_triggers,
            "rule_scores": scores,
            "backtest": backtest,
            "recent7": recent,
            "volume_comment": volume_comment,
            "required_json_schema": {
                "headline": "string",
                "regime": "string",
                "key_levels": {"support": "number", "resistance": "number"},
                "facts": ["string"],
                "interpretation": ["string"],
                "scenarios": {
                    "bull": {"trigger": "string", "invalidate": "string"},
                    "bear": {"trigger": "string", "invalidate": "string"},
                    "detail": ["string"]
                },
                "risks": ["string"],
                "confidence": "integer",
                "confidenceDrivers": ["string"],
                "unknowns_or_gaps": ["string"],
                "regime_breakdown": {"vol": "string", "bb_pctile_120d": "number"},
                "liquidity": {"turnover24hKrw": "number", "minTurnoverKrw": "number", "liquidityOk": "boolean"},
                "risk_model": {"expected_pullback_pct": "number", "stop_atr": "number", "drawdown_hint": "string"}
            },
            "style_guidelines": [
                "과장, 확정 표현 금지", "숫자·임계값을 명시", "모호한 표현 지양"
            ]
        }
    }
    return [system, user]

# 한글 주석: 간단한 지수 백오프 재시도 유틸
import time
def _with_retry(fn, max_try=2, base_delay=0.8):
    last = None
    for i in range(max_try):
        last = fn()
        if last:  # 성공 시 바로 반환
            return last
        time.sleep(base_delay * (2 ** i))
    return None

def _call_openai_chat(prompt):
    """
    한글 주석: OpenAI Chat Completions 호출(JSON 모드 강제).
    반환: dict 또는 None
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # prompt가 메시지 리스트면 그대로, 아니면 래핑
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": "You are a cautious Korean crypto analyst. Never give financial advice."},
                {"role": "user", "content": prompt}
            ]
        body = {
            "model": model,
            "temperature": 0.3,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": messages
        }
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        content = j["choices"][0]["message"]["content"].strip()
        # parse JSON
        return json.loads(content)
    except Exception:
        return None


# def generate_ai_summary(df, rsi, macd, macd_signal, pattern, volume_comment):
def generate_ai_summary(df, rsi, macd, macd_signal, pattern, volume_comment, symbol):
    """
    한글 주석: JSON 기반 AI 분석(→ 스키마/일관성 검증 → 필요 시 모델 기반 수정)
    실패 시 규칙 기반 폴백.
    반환: (text, json_dict_or_none, meta)
    """
    validate_inputs(df, rsi, macd, macd_signal, pattern, volume_comment)
    features = compute_advanced_features_v2(df)
    scores = compute_rule_based_scores(features, rsi_value=rsi)
    k_env = K_TRIGGER
    # backtest = backtest_auto_triggers(df, k=k_env, horizon_days=BT_HORIZON_DAYS, target_atr=BT_TARGET_ATR)
    backtest = compute_backtest_detailed(df, k=k_env, horizon_days=BT_HORIZON_DAYS, target_atr=BT_TARGET_ATR)
    auto_triggers = compute_auto_triggers(features, k=k_env)
    viz = build_viz_timeseries(df, limit=120)
    # 추가 산출물(시계열/트레이드/레벨)
    viz_plus = build_viz_plus(df, k=k_env, target_atr=BT_TARGET_ATR, lookback=120)
    bt_trd = backtest_trades(df, k=k_env, horizon_days=BT_HORIZON_DAYS, target_atr=BT_TARGET_ATR)
    levels_today = {
        "support": features.get("support_20d"),
        "resistance": features.get("resistance_20d"),
        "bullTrigger": auto_triggers.get("bull_trigger"),
        "bearTrigger": auto_triggers.get("bear_trigger"),
        "bullTarget": (auto_triggers.get("bull_trigger") + BT_TARGET_ATR*features.get("atr14", 0.0)) if features.get("atr14") else None,
        "bearTarget": (auto_triggers.get("bear_trigger") - BT_TARGET_ATR*features.get("atr14", 0.0)) if features.get("atr14") else None,
    }
    # prompt = build_ai_prompt(df, rsi, macd, macd_signal, pattern, volume_comment, features, auto_triggers, scores, backtest, prompt_version="v2")
    prompt = build_ai_prompt(df, rsi, macd, macd_signal, pattern, volume_comment, features, auto_triggers, scores, backtest, prompt_version="v2", symbol=symbol)

    # 1) OpenAI 호출 + 레이턴시 측정
    t0 = datetime.now()
    ai_json = _with_retry(lambda: _call_openai_chat(prompt), max_try=2, base_delay=0.8)
    latency_ms = int((datetime.now() - t0).total_seconds() * 1000)
    provider = 'openai_json' if ai_json else 'fallback'

    validation_errors = []
    # 2) 검증 + 경미한 코드단 보정
    if ai_json:
        ok, errs = validate_ai_json_against_schema(ai_json)
        if not ok:
            validation_errors.extend(errs)
            # (a) 보수적 기본값으로 1차 보정
            ai_json = _repair_ai_json_with_defaults(ai_json, features)
            ok2, errs2 = validate_ai_json_against_schema(ai_json)
            if not ok2:
                validation_errors.extend(errs2)
                # (b) 모델에 '수정 지시' 1회 요청
                repaired = _repair_ai_json_via_model(prompt, ai_json, errs2)
                if repaired:
                    ai_json = repaired
                    ok3, errs3 = validate_ai_json_against_schema(ai_json)
                    if not ok3:
                        validation_errors.extend(errs3)
                        ai_json = None
                else:
                    ai_json = None

    # === 초보 친화 필드 보강: 모델 JSON에 easy_summary/badges가 없으면 생성 ===
    if ai_json:
        regime_for_easy = ai_json.get("regime") or features.get("vol_regime") or "중립"
        try:
            rsi_zone = ("과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립"))
        except Exception:
            rsi_zone = "중립"
        macd_dir = "상승" if macd > macd_signal else "하락"
        vol_simple = (volume_comment or "").split(".")[0].strip() or "거래량 변화 정보 부족"
        # symbol_for_easy = sel_symbol if 'sel_symbol' in globals() else "SYMBOL"
        symbol_for_easy = symbol or "SYMBOL"
        if not ai_json.get("easy_summary"):
            ai_json["easy_summary"] = (
                f"현재 {symbol_for_easy}은 {regime_for_easy} 시장에 있으며, RSI {round(rsi,1)}로 {rsi_zone} 구간입니다. "
                f"MACD는 {macd_dir} 신호를 보이고 있고, {vol_simple}"
            )
        if not ai_json.get("badges"):
            badges = []
            if regime_for_easy: badges.append(regime_for_easy)
            if rsi > 70:
                badges.append("과매수")
            elif rsi < 30:
                badges.append("과매도")
            else:
                badges.append("RSI 중립")
            badges.append("상승 신호" if macd > macd_signal else "하락 신호")
            if any(k in (volume_comment or "") for k in ["급증", "높아", "강화"]):
                badges.append("거래량↑")
            ai_json["badges"] = badges

    # 3) 실패 시 규칙 기반 폴백
    if not ai_json:
        ov = (backtest or {}).get("overall", {})
        # 백슬래시가 들어가는 중첩 f-string을 피하기 위해, 패턴 문구를 먼저 만든다.
        pattern_phrase = "명확한 패턴 없음" if (not pattern or pattern == "none") else f"'{pattern}' 패턴 감지"
        # 문단을 줄단위로 구성 (각 줄은 단일 f-string)
        line1 = f"시장 요약: RSI {rsi}, MACD {macd} / 시그널 {macd_signal}. {pattern_phrase}."
        line2 = f"레짐: {features.get('vol_regime')} (ATR/가격 {features.get('vol_regime_ratio')})"
        line3 = f"지지/저항(20D): {features.get('support_20d')} / {features.get('resistance_20d')}"
        line4 = (
            f"볼린저 위치: {features.get('bb_pos_pct')}%, 모멘텀(7/14/28일): "
            f"{features.get('mom7_pct')}% / {features.get('mom14_pct')}% / {features.get('mom28_pct')}%"
        )
        line5 = f"거래량 코멘트: {volume_comment}"
        line6 = f"자동 트리거: 상방 {auto_triggers['bull_trigger']} / 하방 {auto_triggers['bear_trigger']}"
        line7 = (
            f"백테스트(최근): 상방 적중 {ov.get('hitRate_bull')}%, 하방 적중 {ov.get('hitRate_bear')}%, "
            f"어느 한쪽 {ov.get('hitRate_any')}% (표본 {ov.get('samples')})"
        )
        text = "\n".join([
            line1,
            line2,
            line3,
            line4,
            line5,
            line6,
            line7,
            "- 체크포인트: 변동성/거래량 흐름, EMA 크로스 상태, ADX/DI, MFI/OBV",
        ])
        disclaimer = "\n\n※ 본 내용은 정보 제공 목적이며, 투자 조언이 아닙니다."

        # 규칙기반 JSON 백업도 함께 생성하여 null 저장을 방지
        fallback_json, fallback_headline = _build_rule_based_ai_json(
            symbol=symbol, rsi=rsi, macd=macd, macd_signal=macd_signal,
            pattern=pattern, volume_comment=volume_comment,
            features=features, scores=scores, triggers=auto_triggers,
        )
        meta = {
            "promptVersion": "v2",
            "provider": provider,
            "modelName": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "latencyMs": latency_ms,
            "validationErrors": validation_errors,
            "features": features,
            "ruleScores": scores,
            "autoTriggers": auto_triggers,
            "backtest": backtest,
            "kTrigger": k_env,
            "btHorizonDays": BT_HORIZON_DAYS,
            "btTargetAtr": BT_TARGET_ATR,
            "indicatorsLast": {"rsi": rsi, "macd": macd, "macdSignal": macd_signal},
            "viz": viz,
            "vizPlus": viz_plus,
            "btTrades": bt_trd.get("trades"),
            "equityCurve": bt_trd.get("equityCurve"),
            "btStats": bt_trd.get("stats"),
            "levelsToday": levels_today,
        }
        try:
            print("[save.debug] headline=", fallback_json.get('headline'))
            print("[save.debug] regime=", fallback_json.get('regime'))
            print("[save.debug] facts.len=", len(fallback_json.get('facts', [])))
            print("[save.debug] interps.len=", len(fallback_json.get('interpretation', [])))
            print("[save.debug] backtest=", backtest)
        except Exception:
            pass
        return (text + disclaimer).strip(), fallback_json, meta

    # 4) JSON → 마크다운 렌더
    hd = ai_json.get("headline", "")
    regime = ai_json.get("regime", "")
    kl = ai_json.get("key_levels", {}) or {}
    facts = ai_json.get("facts", []) or []
    interp = ai_json.get("interpretation", []) or []
    scen = ai_json.get("scenarios", {}) or {}
    risks = ai_json.get("risks", []) or []
    conf = ai_json.get("confidence", None)

    lines = []
    if hd: lines.append(f"**한 줄 요약**: {hd}")
    if regime: lines.append(f"**시장 레짐**: {regime}")
    if kl: lines.append(f"**주요 레벨**: 지지 {kl.get('support')} / 저항 {kl.get('resistance')}")
    if facts:
        lines.append("**사실(Facts)**")
        for f in facts: lines.append(f"- {f}")
    if interp:
        lines.append("**해석(Interpretation)**")
        for it in interp: lines.append(f"- {it}")
    if scen:
        bull = scen.get("bull", {})
        bear = scen.get("bear", {})
        lines.append("**시나리오**")
        if bull: lines.append(f"- 상방: 트리거={bull.get('trigger')}, 무효화={bull.get('invalidate')}")
        if bear: lines.append(f"- 하방: 트리거={bear.get('trigger')}, 무효화={bear.get('invalidate')}")
    if risks:
        lines.append("**리스크**")
        for r in risks: lines.append(f"- {r}")
    if conf is not None:
        lines.append(f"**확신도**: {conf}/100")
    if backtest and isinstance(backtest, dict) and backtest.get("overall", {}).get("samples", 0) > 0:
        ov = backtest["overall"]
        lines.append(
            f"**백테스트(전체)**: 상방 {ov.get('hitRate_bull')}% / 하방 {ov.get('hitRate_bear')}% / 어느 한쪽 {ov.get('hitRate_any')}% (n={ov.get('samples')}, 95%CI {ov.get('ci_any')})")
        br = backtest.get("byRegime", {})
        for key in ["저변동", "중간", "고변동"]:
            if key in br:
                rr = br[key]
                lines.append(f"- {key}: any {rr.get('hitRate_any')}% (n={rr.get('samples')}, CI {rr.get('ci_any')})")

    disclaimer = "※ 본 내용은 정보 제공 목적이며, 투자 조언이 아닙니다."
    text = "\n".join(lines + ["", disclaimer]).strip()
    meta = {
        "promptVersion": "v2",
        "provider": provider,
        "modelName": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "latencyMs": latency_ms,
        "validationErrors": validation_errors,
        "features": features,
        "ruleScores": scores,
        "autoTriggers": auto_triggers,
        "backtest": backtest,
        "kTrigger": k_env,
        "btHorizonDays": BT_HORIZON_DAYS,
        "btTargetAtr": BT_TARGET_ATR,
        "indicatorsLast": {"rsi": rsi, "macd": macd, "macdSignal": macd_signal},
        "viz": viz,
        "vizPlus": viz_plus,
        "btTrades": bt_trd.get("trades"),
        "equityCurve": bt_trd.get("equityCurve"),
        "btStats": bt_trd.get("stats"),
        "levelsToday": levels_today,
    }
    try:
        print(f"AI provider used: {provider}, latency={latency_ms}ms")
    except Exception:
        pass
    return text, ai_json, meta

# === 오늘의 DAILY PICK (업비트 기준, 1개 심층 분석/저장) ===
# 한국어 주석:
# - 정책 변경: '하루 5개'가 아닌, '하루 1개(심층)'로 저장합니다.
# - 선정 규칙: select_daily_symbol_kst(date_str) 로 일자별 1개 심볼 동결.
# - 저장 포맷(필드/문서키)은 기존과 동일: HOT_{심볼}_{YYYY-MM-DD}

try:
    today_kst = kst_today_str()
    sel_symbol = select_daily_symbol_kst(today_kst) or HOT_SYMBOL_ENV
    print(f"[HOT1] Daily Pick(KST {today_kst}): {sel_symbol}")

    df = fetch_upbit_daily_candles(f"KRW-{sel_symbol}", 200)
    rsi = calculate_rsi(df["close"])
    macd, macd_signal = calculate_macd(df["close"])
    pattern = detect_pattern_enhanced(df["high"], df["low"])  # 데이터 부족 시 'none'
    volume_comment = analyze_volume(df)

    # RSI/MACD 해석
    if rsi < 30:
        rsi_comment = f"RSI가 {rsi}로 과매도 구간에 진입해 반등 가능성이 있습니다."
    elif rsi > 70:
        rsi_comment = f"RSI가 {rsi}로 과매수 구간에 위치해 조정 가능성이 있습니다."
    else:
        rsi_comment = f"RSI는 {rsi}로 중립 구간입니다."
    macd_comment = (f"MACD({macd})가 시그널선({macd_signal}) 위로 돌파하여 상승 전환 신호입니다." if macd > macd_signal
                    else f"MACD({macd})가 시그널선({macd_signal}) 아래로 유지되며 하락 지속 가능성을 시사합니다.")

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
    pattern_comment = (f"현재 '{pattern}' 패턴이 감지되었습니다. {pattern_explanations.get(pattern, '기술적 해석이 제공되지 않았습니다.')}"
                       if pattern != "none" else "현재 명확한 차트 패턴은 감지되지 않았습니다.")

    summary = f"{rsi_comment} {macd_comment} {pattern_comment} {volume_comment}"

    print("=== DEBUG: Inputs for generate_ai_summary ===")
    print("symbol:", sel_symbol)
    print("df.tail():\n", df.tail())
    print("rsi:", rsi)
    print("macd:", macd)
    print("macd_signal:", macd_signal)
    print("pattern:", pattern)
    print("volume_comment:", volume_comment)
    print("=============================================")

    # ai_summary_text, ai_summary_json, ai_meta = generate_ai_summary(df, rsi, macd, macd_signal, pattern, volume_comment)
    ai_summary_text, ai_summary_json, ai_meta = generate_ai_summary(df, rsi, macd, macd_signal, pattern, volume_comment,sel_symbol)

    doc_id = f"{HOT_PREFIX}_{sel_symbol}_{today_kst}"
    data = {
        "symbol": sel_symbol,
        "market": f"KRW-{sel_symbol}",
        "date": today_kst,
        "rsi": rsi,
        "macd": macd,
        "macdSignal": macd_signal,
        "pattern": pattern,
        "summary": summary,
        "aiSummary": ai_summary_text,
        "aiSummaryJson": ai_summary_json,
        "premiumOnly": False,
        "isDailyPick": True,
        "chartData": df.to_dict(orient='records'),
        "aiSummaryScores": ai_meta.get("ruleScores") if ai_meta else None,
        "autoTriggers": ai_meta.get("autoTriggers") if ai_meta else None,
        "promptVersion": ai_meta.get("promptVersion") if ai_meta else "v2",
        "provider": ai_meta.get("provider") if ai_meta else None,
        "modelName": ai_meta.get("modelName") if ai_meta else None,
        "latencyMs": ai_meta.get("latencyMs") if ai_meta else None,
        "validationErrors": ai_meta.get("validationErrors") if ai_meta else [],
        "features": ai_meta.get("features") if ai_meta else None,
        "backtest": ai_meta.get("backtest") if ai_meta else None,
        "kTrigger": ai_meta.get("kTrigger") if ai_meta else None,
        "btHorizonDays": ai_meta.get("btHorizonDays") if ai_meta else None,
        "btTargetAtr": ai_meta.get("btTargetAtr") if ai_meta else None,
        "viz": ai_meta.get("viz") if ai_meta else None,
        "indicatorsLast": ai_meta.get("indicatorsLast") if ai_meta else None,
        # 추가 저장 (고도화)
        "vizPlus": ai_meta.get("vizPlus") if ai_meta else None,
        "btTrades": ai_meta.get("btTrades") if ai_meta else None,
        "equityCurve": ai_meta.get("equityCurve") if ai_meta else None,
        "btStats": ai_meta.get("btStats") if ai_meta else None,
        "levelsToday": ai_meta.get("levelsToday") if ai_meta else None,
    }
    db.collection("daily_analysis").document(doc_id).set(data)
    print(f"✅ Firestore 저장 완료: {doc_id} (HOT1, symbol={sel_symbol})")
except Exception as e:
    print(f"[HOT1][ERROR] 처리 실패: {e}")