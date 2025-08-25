import json
import argparse
from pathlib import Path

from config import DATASET_PATH, KB_JSON_PATH
from data import carica_dataset
from eda import analisi_esplorativa
from kb import build_kb_from_dataset, fuzzy_risk_score
from modeling import train_model
from predict import predict_patient
from random_patient import genera_paziente_random
from rdf_kb import run_full_pipeline


def cli() -> None:
    p = argparse.ArgumentParser(description="Classifier Parkinson (Ricerca-Ragionamento-Apprendimento)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_all = sub.add_parser("all-in-one", help="Build KB + Reason + HTML in one go")
    s_all.add_argument("--csv", default=str(DATASET_PATH))
    s_all.add_argument("--json", required=True)
    s_all.add_argument("--model", default="parkinson_model.joblib")
    s_all.add_argument("--thr", choices=["f1", "youden", "fixed"], default="f1")
    s_all.add_argument("--thr_value", type=float, default=None)
    s_all.add_argument("--ttl", default="kb.ttl")
    s_all.add_argument("--out", default="report.html")

    s_eda = sub.add_parser("eda", help="Analisi esplorativa")
    s_eda.add_argument("--grafici", action="store_true")

    sub.add_parser("kb", help="Costruisci/mostra KB e salva JSON")
    sub.add_parser("train", help="Addestra il modello calibrato")

    s_pred = sub.add_parser("predict", help="Predice (ML) per un nuovo paziente")
    s_pred.add_argument("--json", required=True)
    s_pred.add_argument("--thr", choices=["f1", "youden", "fixed"], default="f1")
    s_pred.add_argument("--thr_value", type=float, default=None)

    s_reason = sub.add_parser("reason", help="Ragionamento KB per un paziente")
    s_reason.add_argument("--json", required=True)

    s_random = sub.add_parser("random_patient", help="Genera un paziente random dal CSV")
    s_random.add_argument("--csv", default=str(DATASET_PATH))
    s_random.add_argument("--out", default="paziente_random.json")

    args = p.parse_args()

    if args.cmd == "eda":
        df = carica_dataset()
        analisi_esplorativa(df, grafici=args.grafici)
        try:
            kb = build_kb_from_dataset(df)
            print("\n=== Mapping dataset -> canonical features ===")
            print(kb._mapping)
            print("\n=== Soglie in KB (con provenance) ===")
            print(kb.query())
        except Exception as e:
            print(f"[KB] Avviso: {e}")

    elif args.cmd == "kb":
        df = carica_dataset()
        kb = build_kb_from_dataset(df)
        print("\n=== Mapping dataset -> canonical features ===")
        print(kb._mapping)
        print("\n=== Soglie (tutte) ===")
        print(kb.query())
        with open(KB_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(kb.to_json(), f, indent=2, ensure_ascii=False)
        print(f"\nKB salvata in: {KB_JSON_PATH}")

    elif args.cmd == "train":
        df = carica_dataset()
        train_model(df)

    elif args.cmd == "predict":
        with open(args.json, "r", encoding="utf-8") as fp:
            patient = json.load(fp)
        label, proba, thr_used, thr_mode_used = predict_patient(
            patient,
            thr_mode=args.thr,
            thr_value=args.thr_value,
        )
        stato = "Parkinson" if label else "Sano"
        print(f"\n[ML] Il paziente è classificato come: {stato} (probabilita {proba:.2%}, soglia {thr_used:.2f}, modalita {thr_mode_used})")

    elif args.cmd == "reason":
        df = carica_dataset()
        kb = build_kb_from_dataset(df)
        with open(args.json, "r", encoding="utf-8") as fp:
            patient = json.load(fp)
        report = kb.evaluate_patient(patient)
        fuzzy_score = fuzzy_risk_score(patient, kb)
        print("\n[KB] Report ragionamento:")
        print(json.dumps({
            "risk_level": report["risk_level"],
            "fuzzy_score": round(float(fuzzy_score), 3),
            "alerts": report["alerts"]
        }, indent=2, ensure_ascii=False))

    elif args.cmd == "random_patient":
        genera_paziente_random(args.csv, args.out)

    elif args.cmd == "all-in-one":
        html = run_full_pipeline(
            csv_path=args.csv,
            patient_json_path=args.json,
            model_path=args.model,
            kb_ttl=args.ttl,
            report_html=args.out,
            thr_mode=args.thr,
            thr_value=args.thr_value,
        )
        print(f"Report generated: {Path(html).resolve()}")


if __name__ == "__main__":
    cli()
