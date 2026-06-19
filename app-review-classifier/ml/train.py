"""
ml/train.py
앱 리뷰 다중 분류 모델 학습 + MLflow 실험 기록
실행: python -m ml.train (프로젝트 루트에서)
"""
import logging
import sys
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 프로젝트 루트를 sys.path에 추가 (모듈 방식 실행 시 필요)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from app.sheet_db import get_train_data
from ml.model_promoter import promote_if_better

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── MLflow 설정 ────────────────────────────────────────────────────────
EXPERIMENT_NAME = "review-classifier"
MODEL_NAME      = "review-classifier"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ── 학습할 모델 목록 ────────────────────────────────────────────────────
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "NaiveBayes":         MultinomialNB(),
    "DecisionTree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range":  (1, 2),
    "sublinear_tf": True,
}


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   model),
    ])


def train():
    # ── 1. 데이터 로드 ──────────────────────────────────────────────────
    logger.info("학습 데이터 로드 중...")
    df = get_train_data()

    if "review_text" not in df.columns or "label" not in df.columns:
        raise ValueError("train_data에 'review_text', 'label' 컬럼이 필요합니다.")

    df = df.dropna(subset=["review_text", "label"])
    X = df["review_text"].astype(str).tolist()
    y = df["label"].tolist()
    logger.info(f"데이터 로드 완료: {len(X)}건 | 클래스: {sorted(set(y))}")

    # ── 2. Train/Test 분리 ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train {len(X_train)}건 / Test {len(X_test)}건")

    # ── 3. MLflow 설정 ──────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow Tracking URI: {config.MLFLOW_TRACKING_URI}")

    # ── 4. 모델별 학습 & 기록 ────────────────────────────────────────────
    best_run_id  = None
    best_acc     = 0.0
    best_model_name = None

    for model_name, model in MODELS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"학습 시작: {model_name}")

        with mlflow.start_run(run_name=model_name) as run:
            pipeline = build_pipeline(model)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="weighted")

            # 파라미터 기록
            mlflow.log_param("model_type",    model_name)
            mlflow.log_param("test_size",     TEST_SIZE)
            mlflow.log_param("tfidf_max_features", TFIDF_PARAMS["max_features"])
            mlflow.log_param("tfidf_ngram_range",  str(TFIDF_PARAMS["ngram_range"]))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples",  len(X_test))

            # 메트릭 기록
            mlflow.log_metric("test_accuracy", round(acc, 4))
            mlflow.log_metric("test_f1_weighted", round(f1, 4))

            # classification_report를 artifact로 저장
            report = classification_report(y_test, y_pred, target_names=config.LABELS)
            report_path = f"classification_report_{model_name}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Model: {model_name}\n\n{report}")
            mlflow.log_artifact(report_path)
            os.remove(report_path)

            # 모델 등록
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

            run_id = run.info.run_id
            logger.info(f"[{model_name}] accuracy={acc:.4f} | f1={f1:.4f} | run_id={run_id[:8]}")

            if acc > best_acc:
                best_acc        = acc
                best_run_id     = run_id
                best_model_name = model_name

    # ── 5. 최고 모델 → champion 승격 ─────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"Best Model: {best_model_name} | accuracy={best_acc:.4f}")

    promoted = promote_if_better(
        model_name=MODEL_NAME,
        new_run_id=best_run_id,
        new_accuracy=best_acc,
        new_model_type=best_model_name,
    )

    if promoted:
        logger.info(f"champion 교체 완료: {best_model_name} (accuracy={best_acc:.4f})")
    else:
        logger.info("현재 champion이 더 우수합니다. 교체하지 않습니다.")

    logger.info("\n학습 파이프라인 완료!")
    return best_run_id, best_acc


if __name__ == "__main__":
    train()
