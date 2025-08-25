from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd

from data import guess_target_column


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def canonical_feature_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}

    def find_contains(*subs: str) -> Optional[str]:
        for low, orig in cols.items():
            if all(s in low for s in subs):
                return orig
        return None

    jitter_pct = None
    for low, orig in cols.items():
        if "jitter" in low and "%" in low:
            jitter_pct = orig
            break
    shimmer_db = find_contains("shimmer", "db")
    hnr_db = find_contains("hnr")
    return {"jitter_pct": jitter_pct, "shimmer_db": shimmer_db, "hnr_db": hnr_db}


class SeverityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class Threshold:
    feature_canonical: str
    operator: str
    value: float
    source: str
    note: str = ""
    severity: SeverityLevel = SeverityLevel.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


class MiniKB:
    def __init__(self):
        self._store: Dict[str, Threshold] = {}
        self._mapping: Dict[str, Optional[str]] = {}

    def set_mapping(self, mapping: Dict[str, Optional[str]]):
        self._mapping = dict(mapping)

    def insert_threshold(self, t: Threshold):
        self._store[t.feature_canonical] = t

    def get_threshold(self, feat: str) -> Optional[Threshold]:
        return self._store.get(feat)

    def to_json(self) -> Dict[str, Any]:
        return {"thresholds": {k: v.to_dict() for k, v in self._store.items()}, "mapping": self._mapping}

    def query(self, feature: Optional[str] = None, operator: Optional[str] = None, source_like: Optional[str] = None) -> pd.DataFrame:
        rows = []
        for feat, thr in self._store.items():
            if feature is not None and feat != feature:
                continue
            if operator is not None and thr.operator != operator:
                continue
            if source_like is not None and (source_like.lower() not in thr.source.lower()):
                continue
            rows.append({
                "feature": feat,
                "operator": thr.operator,
                "value": thr.value,
                "source": thr.source,
                "severity": thr.severity.name,
                "note": thr.note,
                "dataset_column": self._mapping.get(feat),
            })
        return pd.DataFrame(rows, columns=["feature", "operator", "value", "source", "severity", "note", "dataset_column"])


class ExtendedKB(MiniKB):
    def __init__(self):
        super().__init__()
        self.composite_rules: List[Dict[str, Any]] = []

    def add_composite_rule(self, name: str, conditions: List[Tuple[str, str, float]], severity: SeverityLevel):
        self.composite_rules.append({"name": name, "conditions": conditions, "severity": severity})

    def evaluate_patient(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        alerts: List[Dict[str, Any]] = []
        for feat_key, thr in self._store.items():
            dataset_col = self._mapping.get(feat_key)
            if not dataset_col or dataset_col not in patient:
                continue
            val = patient[dataset_col]
            if val is None:
                continue
            triggered = (thr.operator == ">" and val > thr.value) or (thr.operator == "<" and val < thr.value)
            if triggered:
                alerts.append({
                    "type": "threshold",
                    "feature": feat_key,
                    "value": val,
                    "threshold": thr.value,
                    "operator": thr.operator,
                    "severity": thr.severity.name,
                    "note": thr.note,
                    "source": thr.source,
                })
        for rule in self.composite_rules:
            met = 0
            for (feat_key, op, thr_val) in rule["conditions"]:
                dataset_col = self._mapping.get(feat_key)
                if not dataset_col or dataset_col not in patient or patient[dataset_col] is None:
                    continue
                val = patient[dataset_col]
                if (op == ">" and val > thr_val) or (op == "<" and val < thr_val):
                    met += 1
            if met == len(rule["conditions"]) and met > 0:
                alerts.append({
                    "type": "composite",
                    "rule": rule["name"],
                    "severity": rule["severity"].name,
                    "note": "Tutte le condizioni soddisfatte",
                })
        sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        max_sev = max([sev_map.get(a.get("severity", "LOW"), 1) for a in alerts], default=0)
        risk_level = {0: "None", 1: "Low", 2: "Medium", 3: "High"}[max_sev]
        return {"alerts": alerts, "risk_level": risk_level}


def populate_defaults(kb: MiniKB):
    kb.insert_threshold(Threshold("jitter_pct", ">", 0.5,  "literature_placeholder", note="Esempio 0.5%"))
    kb.insert_threshold(Threshold("shimmer_db", ">", 0.35, "literature_placeholder", note="Esempio 0.35 dB"))
    kb.insert_threshold(Threshold("hnr_db",     "<", 20.0, "literature_placeholder", note="Esempio 20 dB"))


def refine_from_data(kb: MiniKB, df: pd.DataFrame, target_col: Optional[str]):
    mapping = kb._mapping or canonical_feature_mapping(df)
    kb.set_mapping(mapping)
    base = df
    if target_col in df.columns and (df[target_col] == 0).sum() >= 15:
        base = df[df[target_col] == 0]
    if mapping.get("jitter_pct"):
        s = _num(base[mapping["jitter_pct"]]).dropna()
        if len(s) >= 10:
            kb.insert_threshold(Threshold("jitter_pct", ">", float(np.quantile(s, 0.95)), "data_quantile_0.95", note="95th perc healthy/all"))
    if mapping.get("shimmer_db"):
        s = _num(base[mapping["shimmer_db"]]).dropna()
        if len(s) >= 10:
            kb.insert_threshold(Threshold("shimmer_db", ">", float(np.quantile(s, 0.95)), "data_quantile_0.95", note="95th perc healthy/all"))
    if mapping.get("hnr_db"):
        s = _num(base[mapping["hnr_db"]]).dropna()
        if len(s) >= 10:
            kb.insert_threshold(Threshold("hnr_db", "<", float(np.quantile(s, 0.05)), "data_quantile_0.05", note="5th perc healthy/all"))


def build_kb_from_dataset(df: pd.DataFrame) -> ExtendedKB:
    kb = ExtendedKB()
    kb.set_mapping(canonical_feature_mapping(df))
    populate_defaults(kb)
    try:
        target = guess_target_column(df)
    except Exception:
        target = None
    refine_from_data(kb, df, target_col=target)
    j = kb.get_threshold("jitter_pct")
    s = kb.get_threshold("shimmer_db")
    if j and s:
        kb.add_composite_rule(
            name="Jitter e Shimmer elevati",
            conditions=[("jitter_pct", j.operator, j.value), ("shimmer_db", s.operator, s.value)],
            severity=SeverityLevel.HIGH,
        )
    return kb


def fuzzy_evaluate(value: float, threshold: float, operator: str, margin: float = 0.10) -> float:
    if value is None:
        return 0.0
    if operator == ">":
        if value <= threshold:
            return 0.0
        upper = threshold * (1 + margin)
        return 1.0 if value >= upper else (value - threshold) / (upper - threshold)
    if operator == "<":
        if value >= threshold:
            return 0.0
        lower = threshold * (1 - margin)
        return 1.0 if value <= lower else (threshold - value) / (threshold - lower)
    return 0.0


def fuzzy_risk_score(patient: Dict[str, Any], kb: MiniKB) -> float:
    scores: List[float] = []
    for feat_key, thr in kb._store.items():
        dataset_col = kb._mapping.get(feat_key)
        if not dataset_col or dataset_col not in patient:
            continue
        val = patient[dataset_col]
        scores.append(fuzzy_evaluate(val, thr.value, thr.operator))
    return float(np.mean(scores)) if scores else 0.0
