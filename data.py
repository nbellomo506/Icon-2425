from pathlib import Path
import pandas as pd

from config import DATASET_PATH


def carica_dataset(percorso: Path = DATASET_PATH) -> pd.DataFrame:
    if not percorso.exists():
        raise FileNotFoundError(f"Dataset non trovato in: {percorso.resolve()}")
    return pd.read_csv(percorso)


def guess_target_column(df: pd.DataFrame) -> str:
    candidates = ("parkinsons_status", "status", "target", "label", "diagnosis", "class")
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    raise ValueError("Colonna target non trovata (attese: parkinsons_status/status/target/label/diagnosis/class).")


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "UPDRS_Score" in df.columns:
        df["UPDRS_High"] = (df["UPDRS_Score"] > 80).astype(int)
    return df
