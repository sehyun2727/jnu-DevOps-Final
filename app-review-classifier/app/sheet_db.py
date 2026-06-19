import logging
import os
from datetime import datetime

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Google Sheets 사용 가능 여부 (credentials 없으면 로컬 CSV 폴백)
_SHEETS_AVAILABLE = False
_gc = None

try:
    import gspread
    from google.oauth2.service_account import Credentials

    _SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    if os.path.exists(config.GOOGLE_CREDENTIALS_FILE) and config.GOOGLE_SHEET_ID:
        _creds = Credentials.from_service_account_file(
            config.GOOGLE_CREDENTIALS_FILE, scopes=_SCOPES
        )
        _gc = gspread.authorize(_creds)
        _SHEETS_AVAILABLE = True
        logger.info("Google Sheets 인증 성공")
    else:
        logger.warning("credentials.json 또는 GOOGLE_SHEET_ID 미설정 → 로컬 CSV 폴백 모드")
except Exception as e:
    logger.warning(f"Google Sheets 초기화 실패 → 로컬 CSV 폴백 모드: {e}")


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────

def _get_sheet(sheet_name: str):
    if not _SHEETS_AVAILABLE:
        raise RuntimeError("Google Sheets 사용 불가")
    spreadsheet = _gc.open_by_key(config.GOOGLE_SHEET_ID)
    return spreadsheet.worksheet(sheet_name)


def _local_csv(filename: str) -> str:
    base = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base, "ml", "data", filename)


# ──────────────────────────────────────────────
# train_data 시트
# ──────────────────────────────────────────────

def get_train_data() -> pd.DataFrame:
    """학습 데이터를 Sheets 또는 로컬 CSV에서 DataFrame으로 반환."""
    if _SHEETS_AVAILABLE:
        try:
            sheet = _get_sheet("train_data")
            records = sheet.get_all_records()
            df = pd.DataFrame(records)
            logger.info(f"Google Sheets train_data 로드 완료: {len(df)}건")
            return df
        except Exception as e:
            logger.warning(f"Sheets 읽기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("train_data.csv")
    df = pd.read_csv(path)
    logger.info(f"로컬 CSV train_data 로드 완료: {len(df)}건")
    return df


# ──────────────────────────────────────────────
# predictions_log 시트
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
            sheet = _get_sheet("predictions_log")
            sheet.append_row(list(row.values()))
            logger.info(f"predictions_log 기록 완료: {predicted_label} ({score:.2%})")
            return
        except Exception as e:
            logger.warning(f"Sheets 쓰기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("predictions_log.csv")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    logger.info("로컬 CSV predictions_log 기록 완료")


# ──────────────────────────────────────────────
# feedback 시트
# ──────────────────────────────────────────────

def log_feedback(
    review_text: str,
    predicted_label: str,
    correct_label: str,
) -> None:
    """사용자 피드백을 Sheets 또는 로컬 CSV에 기록."""
    is_correct = predicted_label == correct_label
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review_text": review_text,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "is_correct": is_correct,
    }

    if _SHEETS_AVAILABLE:
        try:
            sheet = _get_sheet("feedback")
            sheet.append_row(list(row.values()))
            logger.info(f"feedback 기록 완료: correct={is_correct}")
            return
        except Exception as e:
            logger.warning(f"Sheets 쓰기 실패, 로컬 CSV 폴백: {e}")

    path = _local_csv("feedback.csv")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    logger.info("로컬 CSV feedback 기록 완료")
