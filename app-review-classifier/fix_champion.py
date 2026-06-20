"""
champion alias를 로컬에 model.pkl이 있는 최신 버전으로 강제 교체
"""
import os
import sqlite3
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# SQLite에서 model version → storage_location 매핑 조회
conn = sqlite3.connect("mlflow.db")
cur = conn.cursor()
cur.execute("SELECT version, storage_location, run_id FROM model_versions WHERE name='review-classifier' ORDER BY version")
rows = cur.fetchall()
conn.close()

print("=== 모델 버전 & 로컬 artifact 확인 ===")
local_versions = []

for version, storage_location, run_id in rows:
    # storage_location 예: ml/artifacts/1/models/m-xxx/artifacts
    if storage_location:
        model_path = os.path.join(storage_location.lstrip("/").lstrip("\\"), "model.pkl")
        # 절대경로 시도
        abs_path = model_path if os.path.isabs(model_path) else os.path.join(os.getcwd(), model_path)
        exists = os.path.exists(abs_path)
    else:
        exists = False

    run = client.get_run(run_id)
    acc = run.data.metrics.get("test_accuracy", 0)
    model_type = run.data.params.get("model_type", "?")
    print(f"  v{version} | {model_type:<20} | acc={acc} | local={exists}")
    if exists:
        local_versions.append((version, float(acc), model_type))

if not local_versions:
    print("\n로컬 model.pkl을 찾을 수 없습니다.")
    print("storage_location 확인:")
    conn2 = sqlite3.connect("mlflow.db")
    cur2 = conn2.cursor()
    cur2.execute("SELECT version, storage_location FROM model_versions WHERE name='review-classifier' ORDER BY version DESC LIMIT 5")
    for r in cur2.fetchall():
        print(f"  v{r[0]}: {r[1]}")
    conn2.close()
else:
    best = max(local_versions, key=lambda x: x[1])
    best_version, best_acc, best_type = best
    client.set_registered_model_alias("review-classifier", "champion", str(best_version))
    print(f"\nchampion → v{best_version} ({best_type}, accuracy={best_acc:.4f}) 으로 교체 완료!")
