"""
confidence가 낮은 입력 찾기 테스트
실행: python test_confidence.py
"""
import os
os.environ["MODEL_MODE"] = "ml"

import config
import mlflow.sklearn
import numpy as np

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(config.CHAMPION_MODEL_URI)

candidates = [
    "1234567890",
    "!!!???###",
    "hello world this is test",
    "뫄솨뤄쏴횃",
    "abc def ghi",
    "음... 글쎄요",
    "모르겠어요",
    "그냥요",
    "ㅋㅋㅋㅋ",
    "이건 뭔가 좀 그런데 아닌가",
    "별로인것같기도하고아닌것같기도하고",
    "앱",
    "good bad nice bug",
    "アプリ 불편해요",
    "뭔가 이상하면서도 좋은것같은",
]

print(f"{'입력':<35} {'라벨':<10} {'confidence':>12}")
print("-" * 60)
for text in candidates:
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    label = classes[int(np.argmax(proba))]
    confidence = max(proba)
    marker = " ← 65% 이하!" if confidence < 0.65 else ""
    print(f"{text:<35} {label:<10} {confidence:>11.1%}{marker}")
