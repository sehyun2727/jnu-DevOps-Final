import json
import logging
import os
from datetime import datetime

import pandas as pd

import config

logger = logging.getLogger(__name__)

_SHEETS_AVAILABLE = False
_gc = None

try:
    import gspread
    from google.oauth2.service_account import Credentials

    _SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    _sheet_id_or_name = config.GOOGLE_SHEET_NAME or config.GOOGLE_SHEET_ID

    # 우선순위 1: JSON 환경변수 (Render / GitHub Actions)
    if config.GOOGLE_SERVICE_ACCOUNT_JSON and _sheet_id_or_name:
        _info = json.loads(config.GOOGLE_SERVICE_ACCOUNT_JSON)
        _creds = Credentials.from_service_account_info(_info, scopes=_SCOPES)
        _gc = gspread.authorize(_creds)
        _SHEETS_AVAILABLE = True
        logger.info("Google Sheets 인증 성공 (GOOGLE_SERVICE_ACCOUNT_JSON)")

    # 우선순위 2: 로컬 credentials.json 파일
    elif os.path.exists(config.GOOGLE_CREDENTIALS_FILE) and _sheet_id_or_name:
        _creds = Credentials.from_service_account_file(
            config.GOOGLE_CREDENTIALS_FILE, scopes=_SCOPES
        )
        _gc = gspread.authorize(_creds)
        _SHEETS_AVAILABLE = True
        logger.info("Google Sheets 인증 성공 (credentials.json)")

    else:
        logger.warning("Google Sheets 인증 정보 없음 → 로컬 CSV 폴백 모드")

except Exception as e:
    logger.warning(f"Google Sheets 초기화 실패 → 로컬 CSV 폴백 모드: {e}")


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────

def _open_spreadsheet():
    """시트 이름(GOOGLE_SHEET_NAME) 우선, 없으면 ID(GOOGLE_SHEET_ID)로 열기."""
    if config.GOOGLE_SHEET_NAME:
        return _gc.open(config.GOOGLE_SHEET_NAME)
    return _gc.open_by_key(config.GOOGLE_SHEET_ID)


def _get_sheet(sheet_name: str):
    if not _SHEETS_AVAILABLE:
        raise RuntimeError("Google Sheets 사용 불가")
    return _open_spreadsheet().worksheet(sheet_name)


def _local_csv(filename: str) -> str:
    base = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base, "ml", "data", filename)


# ──────────────────────────────────────────────
# train_data 시트 — 읽기
# ──────────────────────────────────────────────

def get_train_data() -> pd.DataFrame:
    """학습 데이터를 Sheets 또는 로컬 CSV에서 반환."""
    if _SHEETS_AVAILABLE:
        try:
            df = pd.DataFrame(_get_sheet("train_data").get_all_records())
            logger.info(f"Google Sheets train_data 로드: {len(df)}건")
            return df
        except Exception as e:
            logger.warning(f"Sheets 읽기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("train_data.csv")
    df = pd.read_csv(path)
    logger.info(f"로컬 CSV train_data 로드: {len(df)}건")
    return df


# ──────────────────────────────────────────────
# prediction_logs 시트 — 쓰기 / 읽기
# ──────────────────────────────────────────────

def log_prediction(
    review_text: str,
    predicted_label: str,
    score: float,
    model_type: str,
    run_id: str,
    serving_model: str = "champion",
) -> None:
    """예측 결과를 Sheets 또는 로컬 CSV에 기록."""
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review_text": review_text,
        "predicted_label": predicted_label,
        "score": round(score, 4),
        "model_type": model_type,
        "run_id": run_id,
        "serving_model": serving_model,
    }

    if _SHEETS_AVAILABLE:
        try:
            _get_sheet("prediction_logs").append_row(list(row.values()))
            logger.info(f"prediction_logs 기록: {predicted_label} ({score:.2%})")
            return
        except Exception as e:
            logger.warning(f"Sheets 쓰기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("predictions_log.csv")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    logger.info("로컬 CSV predictions_log 기록 완료")


def get_predictions_log() -> pd.DataFrame:
    """예측 로그를 Sheets 또는 로컬 CSV에서 반환."""
    if _SHEETS_AVAILABLE:
        try:
            df = pd.DataFrame(_get_sheet("prediction_logs").get_all_records())
            logger.info(f"Google Sheets prediction_logs 로드: {len(df)}건")
            return df
        except Exception as e:
            logger.warning(f"Sheets 읽기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("predictions_log.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["timestamp", "review_text", "predicted_label", "score",
                                  "model_type", "run_id", "serving_model"])


# ──────────────────────────────────────────────
# feedback_logs 시트 — 쓰기 / 읽기
# ──────────────────────────────────────────────

def log_feedback(
    review_text: str,
    predicted_label: str,
    correct_label: str,
    score: float = 0.0,
    serving_model: str = "champion",
) -> None:
    """사용자 피드백을 Sheets 또는 로컬 CSV에 기록."""
    is_correct = predicted_label == correct_label
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review_text": review_text,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "score": round(score, 4),
        "serving_model": serving_model,
        "is_correct": is_correct,
    }

    if _SHEETS_AVAILABLE:
        try:
            _get_sheet("feedback_logs").append_row(list(row.values()))
            logger.info(f"feedback_logs 기록: correct={is_correct}")
            return
        except Exception as e:
            logger.warning(f"Sheets 쓰기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("feedback.csv")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    logger.info("로컬 CSV feedback 기록 완료")


def get_feedback() -> pd.DataFrame:
    """사용자 피드백을 Sheets 또는 로컬 CSV에서 반환."""
    if _SHEETS_AVAILABLE:
        try:
            df = pd.DataFrame(_get_sheet("feedback_logs").get_all_records())
            logger.info(f"Google Sheets feedback_logs 로드: {len(df)}건")
            return df
        except Exception as e:
            logger.warning(f"Sheets 읽기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("feedback.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["timestamp", "review_text", "predicted_label",
                                  "correct_label", "score", "serving_model", "is_correct"])
