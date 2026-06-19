import os
from dotenv import load_dotenv

load_dotenv()

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
CHAMPION_MODEL_URI  = os.getenv("CHAMPION_MODEL_URI",  "models:/review-classifier@champion")
CHALLENGER_MODEL_URI = os.getenv("CHALLENGER_MODEL_URI", "models:/review-classifier@challenger")

# 서비스 모드: "rules" = 더미 규칙, "ml" = MLflow 모델
MODEL_MODE = os.getenv("MODEL_MODE", "rules")

# 카나리 배포
CANARY_ENABLED = os.getenv("CANARY_ENABLED", "false").lower() == "true"
CANARY_RATIO   = float(os.getenv("CANARY_RATIO", "0.25"))   # challenger 비율

# Drift 감지
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.65"))
LOW_CONFIDENCE_LIMIT     = int(os.getenv("LOW_CONFIDENCE_LIMIT", "5"))

# GitHub Issue 자동 생성
GH_TOKEN = os.getenv("GH_TOKEN", "")
GH_REPO  = os.getenv("GH_REPO", "")   # 형식: "owner/repo"

# Google Sheets
# 인증: 우선순위 1 = JSON 문자열 환경변수 (Render/CI), 우선순위 2 = credentials.json 파일 (로컬)
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_CREDENTIALS_FILE     = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
# 시트 지정: GOOGLE_SHEET_NAME(이름) 또는 GOOGLE_SHEET_ID(ID) 중 하나 설정
GOOGLE_SHEET_NAME           = os.getenv("GOOGLE_SHEET_NAME", "")   # 강의 방식 (이름으로 열기)
GOOGLE_SHEET_ID             = os.getenv("GOOGLE_SHEET_ID", "")     # ID로 열기 (대체 방식)

# 라벨
LABELS = ["bug", "request", "praise"]
LABEL_KO = {"bug": "🐛 버그신고", "request": "💡 기능요청", "praise": "👍 칭찬"}
