import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from app.sheet_db import log_prediction, log_feedback

# ── 로깅 설정 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d (%(funcName)s) | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── 앱 시작 시 ML 모델 로드 ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.MODEL_MODE == "ml":
        logger.info("MODEL_MODE=ml → MLflow champion 모델 로드 시작")
        from app import model_loader
        model_loader.load_models()
        if model_loader.is_loaded():
            logger.info("ML 모델 로드 완료")
        else:
            logger.warning("ML 모델 로드 실패 — /classify 호출 시 오류 반환")
    else:
        logger.info("MODEL_MODE=rules → 키워드 더미 분류기 사용")
    yield


# ── FastAPI 앱 ─────────────────────────────────────────────────────────
app = FastAPI(title="앱 리뷰 자동 분류기", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── 요청/응답 모델 ──────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    review_text: str
    predicted_label: str
    correct_label: str


# ── 더미 분류기 (PHASE 2에서 ML 모델로 교체) ───────────────────────────
def _dummy_classify(text: str) -> tuple[str, dict]:
    """키워드 기반 규칙 분류 — MODEL_MODE='rules' 일 때 사용."""
    bug_kw = ["튕", "꺼", "오류", "안 됩", "버그", "에러", "멈춤", "죽", "작동 안",
              "안되", "안돼", "느려", "배터리", "사라", "깨져", "흰 화면", "검은 화면"]
    req_kw = ["추가", "해주세요", "원해요", "기능", "있으면", "지원", "넣어주",
              "바랍니다", "요청", "해줬으면", "하면 좋", "추가해", "부탁"]

    bug_score = sum(1 for kw in bug_kw if kw in text)
    req_score = sum(1 for kw in req_kw if kw in text)

    if bug_score == 0 and req_score == 0:
        return "praise", {"bug": 0.10, "request": 0.15, "praise": 0.75}

    if bug_score >= req_score:
        conf = min(0.65 + bug_score * 0.06, 0.93)
        rest = round(1 - conf, 4)
        probs = {"bug": round(conf, 4), "request": round(rest * 0.4, 4), "praise": round(rest * 0.6, 4)}
    else:
        conf = min(0.65 + req_score * 0.06, 0.93)
        rest = round(1 - conf, 4)
        probs = {"bug": round(rest * 0.3, 4), "request": round(conf, 4), "praise": round(rest * 0.7, 4)}

    label = max(probs, key=probs.get)
    return label, probs


# ── 엔드포인트 ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "model_mode": config.MODEL_MODE, "version": "1.0.0"}


@app.post("/classify")
async def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "리뷰 텍스트를 입력해주세요."}, status_code=422)

    logger.info(f"[classify] 입력: {text[:60]!r}")

    try:
        if config.MODEL_MODE == "ml":
            from app import model_loader
            label, probs, model_info = model_loader.classify(text)
        else:
            label, probs = _dummy_classify(text)
            model_info = {
                "model_type": "rules",
                "run_id": "N/A",
                "serving_model": "dummy",
                "version": "N/A",
            }

        score = probs[label]
        result = {
            "label": label,
            "label_ko": config.LABEL_KO[label],
            "confidence": round(score, 4),
            "probabilities": probs,
            "model_info": model_info,
        }

        logger.info(f"[classify] 결과: {label} ({score:.2%}) [{model_info['serving_model']}]")
        log_prediction(
            text, label, score,
            model_info["model_type"],
            model_info["run_id"],
            model_info["serving_model"],
        )
        return result

    except Exception as e:
        logger.exception(f"[classify] 오류 발생: {e}")
        return JSONResponse({"error": "분류 중 오류가 발생했습니다."}, status_code=500)


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if not all([req.review_text, req.predicted_label, req.correct_label]):
        return JSONResponse({"error": "필수 필드 누락"}, status_code=422)

    try:
        log_feedback(req.review_text, req.predicted_label, req.correct_label)
        logger.info(f"[feedback] predicted={req.predicted_label}, correct={req.correct_label}")
        return {"status": "ok", "message": "피드백 감사합니다!"}
    except Exception as e:
        logger.exception(f"[feedback] 오류 발생: {e}")
        return JSONResponse({"error": "피드백 저장 중 오류"}, status_code=500)
