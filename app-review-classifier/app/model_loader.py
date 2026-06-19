"""
app/model_loader.py
MLflow Model Registry에서 champion / challenger 모델을 로드하고 추론합니다.
"""
import logging
import random

import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient

import config

logger = logging.getLogger(__name__)

_champion_model = None
_challenger_model = None
_champion_meta: dict = {}
_challenger_meta: dict = {}


def _load_model(model_uri: str):
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"모델 로드 성공: {model_uri}")
        return model
    except Exception as e:
        logger.warning(f"모델 로드 실패 ({model_uri}): {e}")
        return None


def _fetch_meta(model_name: str, alias: str) -> dict:
    try:
        client = MlflowClient()
        version = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(version.run_id)
        return {
            "run_id": version.run_id,
            "model_type": run.data.params.get("model_type", "unknown"),
            "version": version.version,
        }
    except Exception as e:
        logger.warning(f"메타데이터 조회 실패 (alias={alias}): {e}")
        return {"run_id": "N/A", "model_type": "unknown", "version": "N/A"}


def load_models() -> None:
    """앱 시작 시 champion(+필요시 challenger) 모델을 MLflow에서 로드합니다."""
    global _champion_model, _challenger_model, _champion_meta, _challenger_meta

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    _champion_model = _load_model(config.CHAMPION_MODEL_URI)
    if _champion_model is not None:
        # URI 형식: models:/review-classifier@champion
        model_name = config.CHAMPION_MODEL_URI.split("/")[1].split("@")[0]
        _champion_meta = _fetch_meta(model_name, "champion")
        logger.info(f"champion 메타: {_champion_meta}")

    if config.CANARY_ENABLED:
        _challenger_model = _load_model(config.CHALLENGER_MODEL_URI)
        if _challenger_model is not None:
            model_name = config.CHALLENGER_MODEL_URI.split("/")[1].split("@")[0]
            _challenger_meta = _fetch_meta(model_name, "challenger")
            logger.info(f"challenger 메타: {_challenger_meta}")


def classify(text: str) -> tuple[str, dict, dict]:
    """
    텍스트를 분류합니다.
    반환: (label, probabilities, model_info)
    CANARY_ENABLED=True 이면 CANARY_RATIO 확률로 challenger 모델을 사용합니다.
    """
    use_challenger = (
        config.CANARY_ENABLED
        and _challenger_model is not None
        and random.random() < config.CANARY_RATIO
    )

    if use_challenger:
        model = _challenger_model
        meta = _challenger_meta
        serving = "challenger"
    else:
        model = _champion_model
        meta = _champion_meta
        serving = "champion"

    if model is None:
        raise RuntimeError(
            "ML 모델이 로드되지 않았습니다. "
            "먼저 `python -m ml.train`으로 모델을 학습하거나 MODEL_MODE=rules로 실행하세요."
        )

    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    label = classes[int(np.argmax(proba))]

    model_info = {
        "model_type": meta.get("model_type", "unknown"),
        "run_id": meta.get("run_id", "N/A"),
        "serving_model": serving,
        "version": meta.get("version", "N/A"),
    }

    return label, probs, model_info


def is_loaded() -> bool:
    return _champion_model is not None
