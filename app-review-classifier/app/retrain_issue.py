"""
app/retrain_issue.py
confidence가 낮은 예측이 반복될 때 GitHub Issue를 자동으로 생성합니다.
"""
import logging
import requests
import config

logger = logging.getLogger(__name__)

_low_confidence_count = 0


def check_and_trigger(confidence: float) -> bool:
    """
    confidence가 LOW_CONFIDENCE_THRESHOLD 이하이면 카운터를 증가시킵니다.
    LOW_CONFIDENCE_LIMIT 횟수에 도달하면 GitHub Issue를 생성합니다.
    Returns: True if issue was created.
    """
    global _low_confidence_count

    if confidence >= config.LOW_CONFIDENCE_THRESHOLD:
        return False

    _low_confidence_count += 1
    logger.warning(
        f"[drift] 낮은 confidence={confidence:.2%} "
        f"(누적 {_low_confidence_count}/{config.LOW_CONFIDENCE_LIMIT})"
    )

    if _low_confidence_count >= config.LOW_CONFIDENCE_LIMIT:
        _low_confidence_count = 0
        return _create_github_issue()

    return False


def _create_github_issue() -> bool:
    if not config.GH_TOKEN or not config.GH_REPO:
        logger.warning("[drift] GH_TOKEN / GH_REPO 미설정 — GitHub Issue 생성 건너뜀")
        return False

    url = f"https://api.github.com/repos/{config.GH_REPO}/issues"
    headers = {
        "Authorization": f"token {config.GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "title": "[자동] 모델 드리프트 감지 — 재학습 필요",
        "body": (
            f"## 모델 드리프트 감지\n\n"
            f"최근 **{config.LOW_CONFIDENCE_LIMIT}건** 예측에서 "
            f"confidence가 **{config.LOW_CONFIDENCE_THRESHOLD:.0%}** 이하로 반복 감지되었습니다.\n\n"
            "### 권장 조치\n"
            "- [ ] 최근 예측 로그(`predictions_log`) 확인\n"
            "- [ ] 새 학습 데이터 추가 (`train_data` 시트)\n"
            "- [ ] `python -m ml.train` 재실행 또는 GitHub Actions 수동 트리거\n"
            "- [ ] 새 champion 모델 교체 여부 검토\n\n"
            "_이 이슈는 drift 감지 시스템에 의해 자동으로 생성되었습니다._"
        ),
        "labels": ["retraining", "drift"],
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        issue_url = resp.json().get("html_url", "")
        logger.info(f"[drift] GitHub Issue 생성 완료: {issue_url}")
        return True
    except Exception as e:
        logger.error(f"[drift] GitHub Issue 생성 실패: {e}")
        return False
