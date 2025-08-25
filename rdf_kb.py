from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import json
import os

from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD, URIRef

from data import carica_dataset
from kb import build_kb_from_dataset, SeverityLevel
from predict import predict_patient

EX = Namespace("http://example.org/pd#")
SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")


@dataclass
class KBRule:
    name: str
    conditions: List[Tuple[str, str, float]]
    severity: str


def kb_to_graph(kb) -> Graph:
    g = Graph()
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("prov", PROV)

    g.add((EX.Feature, RDF.type, RDFS.Class))
    g.add((EX.Threshold, RDF.type, RDFS.Class))
    g.add((EX.Rule, RDF.type, RDFS.Class))
    g.add((EX.Condition, RDF.type, RDFS.Class))
    g.add((EX.SeverityLevel, RDF.type, RDFS.Class))

    for sev in ["LOW", "MEDIUM", "HIGH"]:
        g.add((EX[sev], RDF.type, EX.SeverityLevel))
        g.add((EX[sev], RDFS.label, Literal(sev)))

    g.add((EX.hasThreshold, RDF.type, RDF.Property))
    g.add((EX.operator, RDF.type, RDF.Property))
    g.add((EX.thresholdValue, RDF.type, RDF.Property))
    g.add((EX.severity, RDF.type, RDF.Property))
    g.add((EX.source, RDF.type, RDF.Property))
    g.add((EX.note, RDF.type, RDF.Property))
    g.add((EX.datasetColumn, RDF.type, RDF.Property))
    g.add((EX.hasCondition, RDF.type, RDF.Property))
    g.add((EX.onFeature, RDF.type, RDF.Property))
    g.add((EX.value, RDF.type, RDF.Property))

    for canon, thr in kb._store.items():
        f_uri = EX[canon]
        g.add((f_uri, RDF.type, EX.Feature))
        if kb._mapping.get(canon):
            g.add((f_uri, EX.datasetColumn, Literal(kb._mapping[canon])))

        tnode = URIRef(f"{EX}thr_{canon}")
        g.add((tnode, RDF.type, EX.Threshold))
        g.add((tnode, EX.operator, Literal(thr.operator)))
        g.add((tnode, EX.thresholdValue, Literal(float(thr.value), datatype=XSD.double)))
        g.add((tnode, EX.severity, EX[str(thr.severity.name)]))
        g.add((tnode, EX.source, Literal(thr.source)))
        if thr.note:
            g.add((tnode, EX.note, Literal(thr.note)))
        g.add((f_uri, EX.hasThreshold, tnode))

    for rule in getattr(kb, "composite_rules", []):
        r_uri = URIRef(f"{EX}rule_{_slug(rule['name'])}")
        g.add((r_uri, RDF.type, EX.Rule))
        g.add((r_uri, RDFS.label, Literal(rule["name"])))
        g.add((r_uri, EX.severity, EX[str(rule["severity"].name)]))
        for i, (feat, op, val) in enumerate(rule["conditions"]):
            cnode = URIRef(f"{r_uri}/cond/{i}")
            g.add((cnode, RDF.type, EX.Condition))
            g.add((cnode, EX.onFeature, EX[feat]))
            g.add((cnode, EX.operator, Literal(op)))
            g.add((cnode, EX.value, Literal(float(val), datatype=XSD.double)))
            g.add((r_uri, EX.hasCondition, cnode))

    return g


def _slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name)


def build_rdf_from_dataset(csv_path: str, out_ttl: str = "kb.ttl") -> str:
    try:
        from pathlib import Path
        df = carica_dataset(Path(csv_path)) if csv_path else carica_dataset()
    except Exception:
        df = carica_dataset()

    kb = build_kb_from_dataset(df)

    j = kb.get_threshold("jitter_pct")
    s = kb.get_threshold("shimmer_db")
    h = kb.get_threshold("hnr_db")
    if h and (j or s):
        if j:
            kb.add_composite_rule(
                name="HNR basso AND Jitter alto",
                conditions=[("hnr_db", h.operator, h.value), ("jitter_pct", j.operator, j.value)],
                severity=SeverityLevel.HIGH
            )
        if s:
            kb.add_composite_rule(
                name="HNR basso AND Shimmer alto",
                conditions=[("hnr_db", h.operator, h.value), ("shimmer_db", s.operator, s.value)],
                severity=SeverityLevel.HIGH
            )

    g = kb_to_graph(kb)
    g.serialize(destination=out_ttl, format="turtle")
    return out_ttl


def _get_one(g: Graph, subj, pred):
    for _, _, obj in g.triples((subj, pred, None)):
        return obj
    return None


def _get_label(g: Graph, subj):
    for _, _, lab in g.triples((subj, RDFS.label, None)):
        return str(lab)
    return None


def _thresholds_from_graph(g: Graph) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f_uri, _, _ in g.triples((None, RDF.type, EX.Feature)):
        canon = str(f_uri).split("#")[-1]
        for _, _, tnode in g.triples((f_uri, EX.hasThreshold, None)):
            op = str(_get_one(g, tnode, EX.operator))
            val = float(_get_one(g, tnode, EX.thresholdValue))
            sev_uri = _get_one(g, tnode, EX.severity)
            sev = str(sev_uri).split("#")[-1] if isinstance(sev_uri, URIRef) else str(sev_uri)
            dcol = _get_one(g, f_uri, EX.datasetColumn)
            out[canon] = {"operator": op, "value": val, "severity": sev, "datasetColumn": str(dcol) if dcol else None}
    return out


@dataclass
class _Rule:
    name: str
    conditions: List[Tuple[str, str, float]]
    severity: str


def _rules_from_graph(g: Graph) -> List[_Rule]:
    rules: List[_Rule] = []
    for r_uri, _, _ in g.triples((None, RDF.type, EX.Rule)):
        name = _get_label(g, r_uri) or str(r_uri).split("#")[-1]
        sev_uri = _get_one(g, r_uri, EX.severity)
        sev = str(sev_uri).split("#")[-1] if isinstance(sev_uri, URIRef) else str(sev_uri)
        conds: List[Tuple[str, str, float]] = []
        for _, _, cnode in g.triples((r_uri, EX.hasCondition, None)):
            feat_uri = _get_one(g, cnode, EX.onFeature)
            feat = str(feat_uri).split("#")[-1]
            op = str(_get_one(g, cnode, EX.operator))
            val = float(_get_one(g, cnode, EX.value))
            conds.append((feat, op, val))
        if conds:
            rules.append(_Rule(name=name, conditions=conds, severity=sev))
    return rules


def _fuzzy(value: float, threshold: float, operator: str, margin: float = 0.10) -> float:
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


def evaluate_patient_with_rdf(graph_path: str, patient: Dict[str, Any]) -> Dict[str, Any]:
    g = Graph()
    g.parse(graph_path, format="turtle")

    thresholds = _thresholds_from_graph(g)
    rules = _rules_from_graph(g)

    alerts: List[Dict[str, Any]] = []
    for canon, cfg in thresholds.items():
        col = cfg.get("datasetColumn")
        if not col or col not in patient:
            continue
        try:
            v = float(patient[col])
        except Exception:
            continue
        op = cfg["operator"]
        thr = float(cfg["value"])
        abnormal = (v > thr) if op == ">" else (v < thr)
        if abnormal:
            alerts.append({"type": "threshold", "feature": canon, "operator": op, "value": thr, "observed": v, "severity": "MEDIUM"})

    for rule in rules:
        met = 0
        for feat, op, thr in rule.conditions:
            col = thresholds.get(feat, {}).get("datasetColumn")
            if not col or col not in patient:
                break
            try:
                v = float(patient[col])
            except Exception:
                break
            cond_ok = (v > thr) if op == ">" else (v < thr)
            if cond_ok:
                met += 1
        if met == len(rule.conditions) and met > 0:
            alerts.append({"type": "composite", "rule": rule.name, "severity": rule.severity})

    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    max_sev = max([sev_map.get(a.get("severity", "LOW"), 1) for a in alerts], default=0)
    risk_level = {0: "None", 1: "Low", 2: "Medium", 3: "High"}[max_sev]

    scores: List[float] = []
    for canon, cfg in thresholds.items():
        col = cfg.get("datasetColumn")
        if not col or col not in patient:
            continue
        try:
            v = float(patient[col])
        except Exception:
            continue
        op = cfg["operator"]
        thr = float(cfg["value"])
        scores.append(_fuzzy(v, thr, op))
    fuzzy_score = float(sum(scores) / len(scores)) if scores else 0.0

    return {"alerts": alerts, "risk_level": risk_level, "fuzzy_score": fuzzy_score}


def render_html_report(
    patient: Dict[str, Any],
    ml_summary: Optional[Dict[str, Any]],
    kb_summary: Dict[str, Any],
    outfile: str = "report.html",
    title: str = "Parkinson – ML + KB Report"
) -> str:
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 14px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    h1 { font-size: 22px; margin-top: 0; }
    h2 { font-size: 18px; margin: 0 0 12px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eee; font-size: 14px; }
    .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; }
    .risk-None { background: #e8f0fe; color: #1a73e8; }
    .risk-Low { background: #e6f4ea; color: #137333; }
    .risk-Medium { background: #fff6e6; color: #b06000; }
    .risk-High { background: #fce8e6; color: #c5221f; }
    code { background: #f6f8fa; padding: 0 6px; border-radius: 6px; }
    """
    risk = kb_summary.get("risk_level", "None")
    fuzzy = kb_summary.get("fuzzy_score", 0.0)

    def _row(k, v):
        return f"<tr><th>{k}</th><td>{v}</td></tr>"

    ml_html = ""
    if ml_summary:
        ml_html = f"""
        <div class="card">
          <h2>ML Prediction</h2>
          <table>
            {_row("Predicted label", ml_summary.get("label_readable"))}
            {_row("Probability (Parkinson)", f"{ml_summary.get('probability', 0.0):.3f}")}
            {_row("Threshold used", ml_summary.get("threshold"))}
            {_row("Threshold mode", ml_summary.get("threshold_mode"))}
            {_row("Model path", f"<code>{ml_summary.get('model_path')}</code>")}
          </table>
        </div>
        """

    alerts_rows = "".join(
        f"<tr><td>{a.get('type')}</td><td>{a.get('feature', a.get('rule', ''))}</td><td>{a.get('severity')}</td></tr>"
        for a in kb_summary.get("alerts", [])
    ) or "<tr><td colspan='3'>No alerts</td></tr>"

    html = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="card">
    <h1>{title} <span class="pill risk-{risk}">{risk}</span></h1>
    <div style="font-size:14px;color:#555">Fuzzy risk score: <b>{fuzzy:.3f}</b></div>
  </div>

  {ml_html}

  <div class="card">
    <h2>KB Alerts</h2>
    <table>
      <thead><tr><th>Type</th><th>Feature / Rule</th><th>Severity</th></tr></thead>
      <tbody>
        {alerts_rows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Patient features (excerpt)</h2>
    <table>
      <tbody>
        {"".join(_row(k, v) for k, v in list(patient.items())[:30])}
      </tbody>
    </table>
  </div>
</body>
</html>"""
    with open(outfile, "w", encoding="utf-8") as fp:
        fp.write(html)
    return outfile


def run_full_pipeline(
    csv_path: str,
    patient_json_path: str,
    model_path: str = "parkinson_model.joblib",
    kb_ttl: str = "kb.ttl",
    report_html: str = "report.html",
    thr_mode: str = "f1",
    thr_value: Optional[float] = None,
) -> str:
    out_ttl = build_rdf_from_dataset(csv_path, kb_ttl)

    with open(patient_json_path, "r", encoding="utf-8") as fp:
        patient = json.load(fp)

    ml_summary = None
    if os.path.exists(model_path):
        label, proba, thr_used, thr_mode_used = predict_patient(
            patient,
            model_path=model_path,
            thr_mode=thr_mode,
            thr_value=thr_value,
        )
        ml_summary = {
            "label": int(label),
            "label_readable": "Parkinson" if label else "Sano",
            "probability": float(proba),
            "threshold": float(thr_used),
            "threshold_mode": thr_mode_used,
            "model_path": model_path,
        }

    kb_summary = evaluate_patient_with_rdf(out_ttl, patient)

    html_path = render_html_report(patient, ml_summary, kb_summary, outfile=report_html)
    return html_path
