# 앱스토어 리뷰 자동 분류 서비스

앱스토어 사용자 리뷰를 입력하면 **🐛 버그신고 / 💡 기능요청 / 👍 칭찬** 중 하나로 자동 분류하는 MLOps 웹 서비스

## 프로젝트 구조

```
app-review-classifier/
├── app/
│   ├── main.py            # FastAPI 앱 (엔드포인트)
│   ├── model_loader.py    # MLflow champion 모델 로드
│   ├── sheet_db.py        # Google Sheets 연동
│   └── retrain_issue.py   # drift 감지 → GitHub Issue 생성
├── ml/
│   ├── train.py           # 모델 학습 + MLflow 기록
│   ├── model_promoter.py  # champion 자동 교체
│   └── data/              # 학습/테스트 데이터
├── templates/             # Jinja2 HTML 템플릿
├── static/                # CSS
├── tests/                 # pytest
├── .github/workflows/     # GitHub Actions
├── config.py              # 환경변수 기반 설정
└── requirements.txt
```

## 로컬 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

접속: http://localhost:8000

## 환경 변수 (.env)

```
MODEL_MODE=rules              # rules | ml
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
CHAMPION_MODEL_URI=models:/review-classifier@champion
CANARY_ENABLED=false
GH_TOKEN=<GitHub Personal Access Token>
GH_REPO=<owner/repo>
GOOGLE_SHEET_ID=<Sheet ID>
GOOGLE_CREDENTIALS_FILE=credentials.json
```

## MLflow 서버 실행

```bash
mlflow server --host 0.0.0.0 --port 6430 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts
```

## 테스트

```bash
pytest tests/ -v
```
