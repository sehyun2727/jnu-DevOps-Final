# =============================================================
#  scripts/start_mlflow.ps1
#  MLflow Tracking Server를 로컬에서 시작합니다.
#  사용법: .\scripts\start_mlflow.ps1
#  (app-review-classifier 폴더에서 실행)
# =============================================================

$PORT = 5000
$DB   = "sqlite:///mlflow.db"
$ROOT = "./mlartifacts"

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  MLflow Tracking Server 시작" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  포트     : $PORT"
Write-Host "  DB       : $DB"
Write-Host "  Artifact : $ROOT"
Write-Host ""
Write-Host "[STEP 1] MLflow 서버를 백그라운드에서 시작합니다..."
Write-Host "  → 브라우저: http://localhost:$PORT"
Write-Host ""

# MLflow 서버 시작 (별도 창)
Start-Process powershell -ArgumentList `
    "-NoExit", "-Command", `
    "mlflow server --backend-store-uri $DB --default-artifact-root $ROOT --host 0.0.0.0 --port $PORT"

Start-Sleep -Seconds 3

Write-Host "[STEP 2] ngrok으로 외부 공개합니다..."
Write-Host "  → ngrok이 설치되어 있어야 합니다."
Write-Host "     설치: https://ngrok.com/download  또는  winget install ngrok"
Write-Host ""
Write-Host "  아래 명령어를 새 터미널에서 실행하세요:"
Write-Host ""
Write-Host "    ngrok http $PORT" -ForegroundColor Yellow
Write-Host ""
Write-Host "[STEP 3] ngrok 실행 후 나타나는 Forwarding URL을 복사하세요."
Write-Host "  예시: https://abcd-1234.ngrok-free.app"
Write-Host ""
Write-Host "[STEP 4] .env 파일의 MLFLOW_TRACKING_URI를 ngrok URL로 업데이트하세요:"
Write-Host ""
Write-Host "    MLFLOW_TRACKING_URI=https://abcd-1234.ngrok-free.app" -ForegroundColor Yellow
Write-Host ""
Write-Host "[STEP 5] 모델 학습 (ngrok URL이 MLFLOW_TRACKING_URI에 설정된 상태):"
Write-Host ""
Write-Host "    python -m ml.train" -ForegroundColor Yellow
Write-Host ""
Write-Host "[STEP 6] ML 모드로 서비스 실행:"
Write-Host ""
Write-Host "    `$env:MODEL_MODE='ml'; uvicorn app.main:app --reload" -ForegroundColor Yellow
Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  MLflow UI: http://localhost:$PORT" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Cyan
