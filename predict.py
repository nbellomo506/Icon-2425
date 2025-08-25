from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import joblib

from config import MODEL_PATH
from data import add_domain_features


def _load_payload(model_path=MODEL_PATH) -> Dict[str, Any]:
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "thresholds": {"f1_weighted": 0.5, "youden": 0.5}, "features": None}


def _ensure_columns(X_new: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    for c in expected:
        if c not in X_new.columns:
            X_new[c] = np.nan
    return X_new[expected]


def predict_patient(
    patient: Dict[str, Any],
    model_path=MODEL_PATH,
    thr_mode: str = "f1",
    thr_value: Optional[float] = None
) -> Tuple[int, float, float, str]:
    data = _load_payload(model_path)
    model = data["model"]
    thresholds = data.get("thresholds", {"f1_weighted": 0.5, "youden": 0.5})
    expected = data.get("features")

    if thr_mode == "f1":
        threshold = float(thresholds.get("f1_weighted", 0.5))
    elif thr_mode == "youden":
        threshold = float(thresholds.get("youden", 0.5))
    elif thr_mode == "fixed":
        threshold = float(thr_value) if thr_value is not None else 0.5
    else:
        threshold = 0.5
        thr_mode = "fixed"

    X_new = pd.DataFrame([patient])
    X_new = add_domain_features(X_new)
    if expected is not None:
        X_new = _ensure_columns(X_new, expected)

    proba = model.predict_proba(X_new)[:, 1][0]
    label = int(proba >= threshold)
    return label, proba, threshold, thr_mode
