"""
ml/model_promoter.py
MLflow Model Registry에서 champion / challenger alias를 관리합니다.
"""
import logging
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

CHAMPION_ALIAS   = "champion"
CHALLENGER_ALIAS = "challenger"


def _get_champion_accuracy(client: MlflowClient, model_name: str) -> float:
    """현재 champion 모델의 test_accuracy를 반환. champion 없으면 0.0."""
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        run     = client.get_run(version.run_id)
        return run.data.metrics.get("test_accuracy", 0.0)
    except Exception:
        return 0.0


def _latest_version_for_run(client: MlflowClient, model_name: str, run_id: str) -> str:
    """특정 run_id로 등록된 모델 버전 중 가장 최신 버전 번호를 반환."""
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
    if not versions:
        raise ValueError(f"run_id={run_id!r} 에 해당하는 모델 버전을 찾을 수 없습니다.")
    return sorted(versions, key=lambda v: int(v.version))[-1].version


def promote_if_better(
    model_name: str,
    new_run_id: str,
    new_accuracy: float,
    new_model_type: str,
) -> bool:
    """
    새 모델이 현재 champion보다 test_accuracy가 높으면 champion alias를 교체합니다.
    기존 champion은 challenger로 강등됩니다.
    Returns: True if promoted, False otherwise.
    """
    client      = MlflowClient()
    current_acc = _get_champion_accuracy(client, model_name)

    logger.info(
        f"[promoter] champion 비교: current={current_acc:.4f} vs new={new_accuracy:.4f} ({new_model_type})"
    )

    if new_accuracy <= current_acc:
        logger.info("[promoter] 현재 champion이 더 우수합니다. 교체 건너뜀.")
        return False

    new_version = _latest_version_for_run(client, model_name, new_run_id)

    # 기존 champion → challenger 강등
    try:
        old = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        client.set_registered_model_alias(model_name, CHALLENGER_ALIAS, old.version)
        logger.info(f"[promoter] 기존 champion v{old.version} → challenger 강등")
    except Exception:
        pass  # 기존 champion 없음 — 첫 번째 등록

    # 새 모델 → champion 승격
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, new_version)
    logger.info(
        f"[promoter] v{new_version} ({new_model_type}) → champion 승격 (accuracy={new_accuracy:.4f})"
    )
    return True
