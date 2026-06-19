"""
dashboard.py
MLOps 모니터링 대시보드
실행: streamlit run dashboard.py  (app-review-classifier 폴더에서)
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from app.sheet_db import get_predictions_log, get_feedback

st.set_page_config(page_title="MLOps 모니터링 대시보드", page_icon="📊", layout="wide")
st.title("📊 MLOps 모니터링 대시보드")
st.caption("앱 리뷰 자동 분류기 — 실시간 운영 현황")

if st.button("🔄 새로고침"):
    st.cache_data.clear()


# ── 데이터 로드 ────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    return get_predictions_log(), get_feedback()

pred_df, fb_df = load_data()

# timestamp 컬럼 파싱
for df in [pred_df, fb_df]:
    if "timestamp" in df.columns and not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ── 섹션 1: 운영 지표 ─────────────────────────────────────────
st.subheader("운영 지표")
col1, col2, col3, col4 = st.columns(4)

total = len(pred_df)
avg_conf = round(pred_df["score"].astype(float).mean(), 4) if total > 0 else 0.0
low_conf = int((pred_df["score"].astype(float) < 0.65).sum()) if total > 0 else 0
label_counts = pred_df["predicted_label"].value_counts().to_dict() if total > 0 else {}

col1.metric("총 예측 건수", total)
col2.metric("평균 Confidence", f"{avg_conf:.2%}")
col3.metric("낮은 Confidence (<65%)", low_conf)
col4.metric("🐛 버그 / 💡 요청 / 👍 칭찬",
            f"{label_counts.get('bug',0)} / {label_counts.get('request',0)} / {label_counts.get('praise',0)}")

st.divider()

# ── 섹션 2: Confidence Trend ──────────────────────────────────
st.subheader("Confidence Trend")
if not pred_df.empty and "timestamp" in pred_df.columns:
    trend_df = pred_df[["timestamp", "score"]].dropna().set_index("timestamp")
    trend_df["score"] = trend_df["score"].astype(float)
    st.line_chart(trend_df, y="score", use_container_width=True)
    st.caption("예측 요청마다 기록된 confidence 추이")
else:
    st.info("예측 데이터가 없습니다. 서비스에서 리뷰를 분류해보세요.")

st.divider()

# ── 섹션 3: 모델별 사용 현황 ──────────────────────────────────
st.subheader("모델별 서빙 현황")
if not pred_df.empty and "serving_model" in pred_df.columns:
    col_a, col_b = st.columns(2)
    with col_a:
        st.bar_chart(pred_df["serving_model"].value_counts(), use_container_width=True)
        st.caption("champion / challenger 서빙 분포")
    with col_b:
        if "predicted_label" in pred_df.columns:
            st.bar_chart(pred_df["predicted_label"].value_counts(), use_container_width=True)
            st.caption("예측 라벨 분포")
else:
    st.info("서빙 데이터가 없습니다.")

st.divider()

# ── 섹션 4: 사용자 피드백 ────────────────────────────────────
st.subheader("사용자 피드백")
if not fb_df.empty:
    fb_total = len(fb_df)
    if "is_correct" in fb_df.columns:
        # True/False 또는 "True"/"False" 문자열 모두 처리
        correct_mask = fb_df["is_correct"].astype(str).str.lower() == "true"
        correct = int(correct_mask.sum())
        wrong = fb_total - correct
        wrong_rate = wrong / fb_total if fb_total > 0 else 0
    else:
        correct, wrong, wrong_rate = 0, 0, 0.0

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("피드백 총 건수", fb_total)
    fc2.metric("오분류 피드백", wrong)
    fc3.metric("오분류율", f"{wrong_rate:.1%}")

    if "correct_label" in fb_df.columns:
        st.bar_chart(fb_df["correct_label"].value_counts(), use_container_width=True)
        st.caption("사용자가 제출한 정답 라벨 분포")
else:
    st.info("아직 피드백 데이터가 없습니다. 웹 UI에서 피드백 버튼을 눌러보세요.")

st.divider()

# ── 섹션 5: 최근 예측 로그 ───────────────────────────────────
st.subheader("최근 예측 로그 (최신 20건)")
if not pred_df.empty:
    show_cols = [c for c in ["timestamp", "review_text", "predicted_label", "score", "serving_model", "model_type"]
                 if c in pred_df.columns]
    st.dataframe(pred_df[show_cols].tail(20).iloc[::-1], use_container_width=True)
else:
    st.info("예측 로그가 없습니다.")
