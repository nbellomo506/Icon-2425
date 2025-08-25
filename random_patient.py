import os
import json
import random
import pandas as pd


def genera_paziente_random(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset '{csv_path}' non trovato.")

    df = pd.read_csv(csv_path)

    for col in ["parkinsons_status", "status", "target", "label", "diagnosis", "class"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            break

    def random_value_for_feature(feature):
        f = feature.lower()
        if "jitter" in f:
            return round(random.uniform(0.1, 1.5), 3)
        if "shimmer" in f:
            return round(random.uniform(0.01, 0.5), 3)
        if "hnr" in f:
            return round(random.uniform(10, 35), 2)
        if "age" in f:
            return random.randint(40, 85)
        if "updrs" in f:
            return random.randint(0, 120)

        s = df[feature]
        if s.dtype.kind in "biufc":
            min_val = s.min(); max_val = s.max()
            if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                return round(random.uniform(0, 1), 3)
            return round(random.uniform(float(min_val), float(max_val)), 3)

        vals = s.dropna().unique().tolist()
        if not vals:
            return "unknown"
        return random.choice(vals)

    paziente_random = {col: random_value_for_feature(col) for col in df.columns}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(paziente_random, f, indent=4, ensure_ascii=False)

    print(f"✅ Creato '{out_path}'.")
    print(f"Esegui: python -m parkinson_classifier.main predict --json {out_path} --thr youden")
