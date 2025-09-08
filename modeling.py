from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from config import RANDOM_STATE, CV_SPLITS, MODEL_PATH
from data import add_domain_features, guess_target_column


def build_preprocessor(df: pd.DataFrame):
    target = guess_target_column(df)
    cat_cols = [c for c in df.columns if (df[c].dtype == "object" or str(df[c].dtype).startswith("category")) and c != target]
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]

    numeric_t = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_t = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_t, num_cols),
        ("cat", cat_t, cat_cols),
    ])

    selector = SelectKBest(score_func=f_classif, k="all")
    return preprocessor, selector, target, num_cols + cat_cols


def reliability_curve(y_true: np.ndarray, y_proba: np.ndarray, bins: int = 10):
    y_true = np.asarray(y_true).astype(float)
    y_proba = np.asarray(y_proba).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(y_proba, edges[1:-1], right=False)
    preds, obs = [], []
    for b in range(bins):
        mask = (idx == b)
        if np.any(mask):
            preds.append(float(np.mean(y_proba[mask])))
            obs.append(float(np.mean(y_true[mask])))
    return np.array(preds), np.array(obs)


def train_model(df: pd.DataFrame, save_path=MODEL_PATH):
    df = add_domain_features(df)
    pre, selector, target, feats = build_preprocessor(df)

    X, y = df[feats], df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train_full, y_train_full, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"ROC-AUC media (CV={CV_SPLITS}): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    pipe.fit(X_fit, y_fit)

    method = "isotonic" if len(y_cal) >= 100 else "sigmoid"
    cal = CalibratedClassifierCV(pipe, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    print(f"Calibrazione probabilita: metodo = {method}")

    proba_cal = cal.predict_proba(X_cal)[:, 1]

    ths = np.linspace(0.0, 1.0, 1001)
    f1s = [f1_score(y_cal, (proba_cal >= t).astype(int), average="weighted", zero_division=0) for t in ths]
    best_f1_thr = float(ths[int(np.argmax(f1s))])

    fpr, tpr, roc_thrs = roc_curve(y_cal, proba_cal)
    youden = tpr - fpr
    best_youden_thr = float(roc_thrs[int(np.argmax(youden))])

    print(f"Soglia ottima (validation) F1-weighted: {best_f1_thr:.3f}")
    print(f"Soglia ottima (validation) Youden J:    {best_youden_thr:.3f}")

    proba_test = cal.predict_proba(X_test)[:, 1]
    y_pred_05 = (proba_test >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred_05)
    auc = roc_auc_score(y_test, proba_test)
    print("\n=== Performance su test (probabilita calibrate) ===")
    print(f"Accuracy @0.50: {acc:.3f}\nROC-AUC (calibr.): {auc:.3f}\n")
    print(classification_report(y_test, y_pred_05, target_names=["Sano", "Parkinson"]))

    cm = confusion_matrix(y_test, y_pred_05)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sano", "Parkinson"])
    disp.plot()
    plt.title("Confusion matrix (test) - soglia 0.50 (prob. calibrate)")
    plt.show()

    px, ox = reliability_curve(getattr(y_test, "values", y_test), proba_test, bins=10)
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    if len(px) > 0:
        plt.plot(px, ox, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability diagram (test)")
    plt.grid(True, alpha=0.3)
    plt.show()

    best_thresholds = {"f1_weighted": best_f1_thr, "youden": best_youden_thr}
    payload = {
        "model": cal,
        "thresholds": best_thresholds,
        "features": feats,
        "calibrated": True,
        "calibration": {"method": method}
    }

    joblib.dump(payload, save_path)
    print(f"Modello calibrato e soglie salvati in {save_path.resolve()}")






from dataclasses import asdict
from sklearn.model_selection import StratifiedKFold

def _metrics_at(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> dict:
    y_pred = (y_proba >= thr).astype(int)
    return {
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }

def _best_thresholds(y_val: np.ndarray, p_val: np.ndarray, grid_points: int = 201) -> dict:
    # Soglia ottima per F1-weighted
    grid = np.linspace(0.0, 1.0, grid_points)
    f1s = [f1_score(y_val, (p_val >= t).astype(int), average="weighted", zero_division=0) for t in grid]
    thr_f1 = float(grid[int(np.argmax(f1s))])

    # Soglia ottima Youden J = TPR - FPR (su ROC)
    fpr, tpr, roc_thrs = roc_curve(y_val, p_val)
    youden = tpr - fpr
    thr_youden = float(roc_thrs[int(np.argmax(youden))])

    return {"fixed_050": 0.50, "best_f1": thr_f1, "best_youden": thr_youden}

def _build_calibrated(df: pd.DataFrame, method: str, n_estimators: int, seed: int):
    pre, selector, target, feats = build_preprocessor(df)
    X = df[feats]
    y = df[target]

    # split train -> fit/calib (come in train_model)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )
    pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
    pipe.fit(X_fit, y_fit)

    cal = CalibratedClassifierCV(pipe, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal, feats, target

def kfold_metrics_matrix(
    df: pd.DataFrame,
    splits: int = 5,
    method: str = "sigmoid",      # oppure "isotonic"
    n_estimators: int = 300,
    grid_points: int = 201,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restituisce:
      - df_detailed: matrice 5x3 (fold x soglie) con F1_weighted e Accuracy
      - df_summary: media e std per soglia (su 5 fold)
    """
    df = add_domain_features(df)
    pre, selector, target, feats = build_preprocessor(df)
    X = df[feats]
    y = df[target].astype(int)

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    rows = []

    fold_idx = 0
    for tr_idx, te_idx in skf.split(X, y):
        fold_idx += 1
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values

        # Dentro a train: faccio fit/cal come in train_model
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed + fold_idx
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed + fold_idx,
        )
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        pipe.fit(X_fit, y_fit)

        cal = CalibratedClassifierCV(pipe, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)

        # Soglie dalla calibrazione (validation interno)
        p_cal = cal.predict_proba(X_cal)[:, 1]
        th = _best_thresholds(y_cal, p_cal, grid_points=grid_points)

        # Valutazione sul fold di test
        p_te = cal.predict_proba(X_te)[:, 1]
        for name, tval in th.items():
            met = _metrics_at(y_te, p_te, tval)
            rows.append({
                "fold": fold_idx,
                "threshold_name": name,
                "threshold": float(tval),
                "F1_weighted": met["F1_weighted"],
                "Accuracy": met["Accuracy"],
            })

    df_detailed = pd.DataFrame(rows).sort_values(["fold", "threshold_name"]).reset_index(drop=True)

    # Aggregazione (media, std) sui 5 fold
    agg = df_detailed.groupby("threshold_name").agg(
        F1_mean=("F1_weighted", "mean"),
        F1_std =("F1_weighted", "std"),
        ACC_mean=("Accuracy", "mean"),
        ACC_std =("Accuracy", "std"),
        thr_mean=("threshold", "mean"),
        thr_std =("threshold", "std"),
    ).reset_index()

    return df_detailed, agg

def repeated_cv_table(
    df: pd.DataFrame,
    repeats: int = 5,
    splits: int = 5,
    method: str = "sigmoid",
    n_estimators: int = 300,
    grid_points: int = 201,
    outdir: str | None = None,          # <--- NEW
    out_csv: str | None = None,
    out_json: str | None = None,
) -> None:
    """
    Esegue più run (semi diversi) e salva:
      - CSV: media/std aggregata su *tutti* i fold e run per ogni soglia
      - JSON: dettagli (repeat × fold × soglia)

    Se 'outdir' è fornita, crea la cartella e salva lì i file.
    Se 'out_csv' / 'out_json' sono solo nomi (senza path), li salva dentro outdir.
    Se 'out_csv' / 'out_json' includono già un percorso, quello viene rispettato.
    """
    k = int(splits)
    R = int(repeats)

    all_rows = []
    for r in range(R):
        seed = RANDOM_STATE + r
        df_det, _ = kfold_metrics_matrix(
            df, splits=k, method=method, n_estimators=n_estimators, grid_points=grid_points, seed=seed
        )
        df_det = df_det.copy()
        df_det["repeat"] = r + 1
        all_rows.append(df_det)

    detailed = pd.concat(all_rows, ignore_index=True)

    summary = detailed.groupby("threshold_name").agg(
        F1_mean=("F1_weighted", "mean"),
        F1_std =("F1_weighted", "std"),
        ACC_mean=("Accuracy", "mean"),
        ACC_std =("Accuracy", "std"),
        thr_mean=("threshold", "mean"),
        thr_std =("threshold", "std"),
        n=("threshold", "count"),
    ).reset_index()

    # --- Path handling
    dir_path = Path(outdir) if outdir else None
    if dir_path:
        dir_path.mkdir(parents=True, exist_ok=True)

    # fallback dinamici se non passati
    auto_csv_name  = f"k{k}_r{R}_summary.csv"
    auto_json_name = f"k{k}_r{R}_detailed.json"

    # CSV
    if out_csv is None:
        csv_path = (dir_path / auto_csv_name) if dir_path else Path(auto_csv_name)
    else:
        given = Path(out_csv)
        csv_path = given if given.parent != Path("") else ((dir_path / given.name) if dir_path else given)

    # JSON
    if out_json is None:
        json_path = (dir_path / auto_json_name) if dir_path else Path(auto_json_name)
    else:
        given = Path(out_json)
        json_path = given if given.parent != Path("") else ((dir_path / given.name) if dir_path else given)

    # --- Salvataggi
    summary.to_csv(csv_path, index=False)
    payload = {
        "repeats": R,
        "splits": k,
        "method": method,
        "n_estimators": n_estimators,
        "grid_points": grid_points,
        "rows": detailed.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] Salvati:\n- CSV:  {csv_path.resolve()}\n- JSON: {json_path.resolve()}")
