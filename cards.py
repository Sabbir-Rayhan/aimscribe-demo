"""Load canonical cards and turn them into (a) text to embed and (b) text for the model prompt."""
import hashlib
import json
from config import CARDS_PATH

SHORT_SOURCE = {
    "cholera": "WHO cholera tools v1.0 (2024)",
    "chikungunya": "BD chikungunya guideline (2017)",
    "dengue": "BD dengue guideline, 5th ed.",
    "hypertension": "BD hypertension guideline, 2nd ed. (2023)",
}


def load_cards(path=CARDS_PATH):
    with open(path, encoding="utf-8") as f:
        cards = [json.loads(line) for line in f if line.strip()]
    return cards


def cards_hash(cards):
    return hashlib.sha256(json.dumps(cards, sort_keys=True).encode("utf-8")).hexdigest()[:16]


TEXT_VERSION = 2


def retrieval_text_v1(c):
    """Text that represents a card in the vector index.

    Built from what a clinician would type or an NER step would extract:
    findings, patient factors, title and the action. The evidence passage is
    included so wording in the protocol can match wording in the case.
    """
    p = c["prerequisites"]
    parts = [
        c["disease"], c["clinical_category"], c["title"],
        "Findings: " + "; ".join(p["symptoms_or_findings"]),
        "Patient: " + "; ".join(p["patient_factors"]) if p["patient_factors"] else "",
        "Timing: " + p["timing_or_phase"] if p.get("timing_or_phase") else "",
        "Action: " + c["recommended_action"],
        "Evidence: " + " ".join(e["passage"] for e in c["evidence"][:2]),
    ]
    return " | ".join(x for x in parts if x)


def retrieval_text(c):
    """v2: short and finding-centred. The long action and evidence text of v1 made every card
    look similar to every case (scores clustered), so generic cards were retrieved for all diseases."""
    p = c["prerequisites"]
    parts = [
        f"{c['disease']} {c['clinical_category']}: {c['title']}",
        "Findings: " + "; ".join(p["symptoms_or_findings"]),
        "Patient: " + "; ".join(p["patient_factors"]) if p["patient_factors"] else "",
        "Timing: " + p["timing_or_phase"] if p.get("timing_or_phase") else "",
        "Action: " + c["recommended_action"][:200],
    ]
    return " | ".join(x for x in parts if x)


def page_label(c):
    pages = c["source"]["pdf_pages"]
    return "p" + ",".join(str(p) for p in pages)


def short_ref(c):
    return f"{c['card_id']} ({SHORT_SOURCE[c['disease']]}, PDF {page_label(c)})"


def _trim(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def card_for_prompt(c, annotation=None):
    """Compact, readable card block for the model."""
    p = c["prerequisites"]
    lines = [
        f"[{c['card_id']}] {c['title']}  | disease: {c['disease']} | category: {c['clinical_category']} | source: {SHORT_SOURCE[c['disease']]}, PDF {page_label(c)}",
        f"  findings/prerequisites ({p['match_logic']}): " + "; ".join(p["symptoms_or_findings"]),
    ]
    if p["patient_factors"]:
        lines.append("  patient factors: " + "; ".join(p["patient_factors"]))
    if p.get("timing_or_phase"):
        lines.append("  timing: " + p["timing_or_phase"])
    lines.append("  action: " + c["recommended_action"])
    for m in c["medication"]:
        fields = {k: m[k] for k in ["name", "dose", "route", "frequency", "duration", "instructions"]}
        txt = "; ".join(f"{k}={v if v else 'NOT STATED'}" for k, v in fields.items())
        lines.append("  medication: " + txt)
    for v in c["decision_variants"]:
        note = f" (note: {v['note']})" if v.get("note") else ""
        lines.append(f"  variant [{v['id']}]: IF {v['applies_if']} THEN {v['action']}{note}")
    if c.get("lookup_table"):
        lines.append("  table: " + _trim(json.dumps(c["lookup_table"], ensure_ascii=False), 1800))
    ex = c["exclusions_or_missing_info"]
    if ex["exclusions"]:
        lines.append("  exclusions: " + "; ".join(ex["exclusions"]))
    if ex["missing_info_to_ask"]:
        lines.append("  needs from patient: " + "; ".join(ex["missing_info_to_ask"]))
    if ex["caveats"]:
        lines.append("  caveats: " + _trim("; ".join(ex["caveats"]), 400))
    for e in c["evidence"][:3]:
        lines.append(f"  evidence (PDF p{e['pdf_page']}): \"{_trim(e['passage'], 320)}\"")
    if annotation:
        lines.append(
            f"  >> APPLICABILITY CHECK: {annotation.get('verdict', '?').upper()}"
            + (f" | matched variant: {annotation['matched_variant']}" if annotation.get("matched_variant") else "")
            + f" | reason: {annotation.get('reason', '')}"
            + (f" | missing: {', '.join(annotation['missing'])}" if annotation.get("missing") else "")
        )
    return "\n".join(lines)
