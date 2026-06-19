import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── /health ────────────────────────────────────────────
def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ── /classify 정상 입력 ────────────────────────────────
def test_classify_bug():
    res = client.post("/classify", json={"text": "앱이 자꾸 튕겨요"})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "bug"
    assert 0 < data["confidence"] <= 1
    assert set(data["probabilities"].keys()) == {"bug", "request", "praise"}


def test_classify_request():
    res = client.post("/classify", json={"text": "다크모드 기능을 추가해주세요"})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "request"
    assert data["confidence"] > 0.5


def test_classify_praise():
    res = client.post("/classify", json={"text": "정말 편하고 좋은 앱이에요"})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "praise"


# ── /classify 빈 입력 ─────────────────────────────────
def test_classify_empty():
    res = client.post("/classify", json={"text": ""})
    assert res.status_code == 422
    assert "error" in res.json()


def test_classify_whitespace():
    res = client.post("/classify", json={"text": "   "})
    assert res.status_code == 422


# ── /classify 응답 구조 검증 ──────────────────────────
def test_classify_response_structure():
    res = client.post("/classify", json={"text": "앱이 자꾸 꺼져요"})
    assert res.status_code == 200
    data = res.json()
    assert "label" in data
    assert "label_ko" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "model_info" in data
    assert data["label"] in ["bug", "request", "praise"]


# ── /feedback ──────────────────────────────────────────
def test_feedback_correct():
    res = client.post("/feedback", json={
        "review_text": "앱이 자꾸 튕겨요",
        "predicted_label": "bug",
        "correct_label": "bug",
    })
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_feedback_missing_field():
    res = client.post("/feedback", json={
        "review_text": "앱이 자꾸 튕겨요",
        "predicted_label": "bug",
    })
    assert res.status_code == 422
