"""All prompts. The baseline and improved modes share SYSTEM_GEN and the output schema;
the only difference is the applicability annotation added to the card blocks (see pipeline.py)."""
from cards import card_for_prompt

PATIENT_FIELDS = [
    ("complaints", "Chief complaints / symptoms"),
    ("duration", "Illness duration"),
    ("age", "Age"),
    ("sex", "Sex"),
    ("weight_kg", "Weight (kg)"),
    ("pregnancy", "Pregnancy status"),
    ("history", "Medical history"),
    ("medicines", "Current medicines"),
    ("allergies", "Allergies"),
    ("findings", "Examination findings / measurements / test results"),
]


def patient_block(p):
    lines = []
    for key, label in PATIENT_FIELDS:
        v = str(p.get(key, "") or "").strip()
        lines.append(f"- {label}: {v if v else 'NOT PROVIDED'}")
    return "\n".join(lines)


EMPTY_VALUES = {"none", "none known", "nil", "no", "n/a", "na", "not known", "unknown"}


def patient_query(p):
    """Free text used to retrieve cards. No disease name is required.
    Empty and 'none' answers are left out so they do not match cards that merely mention allergies or medicines."""
    parts = []
    for key, label in PATIENT_FIELDS:
        v = str(p.get(key, "") or "").strip()
        if v and v.lower().strip(". ") not in EMPTY_VALUES:
            parts.append(f"{label}: {v}")
    return ". ".join(parts)


SCHEMA = """{
  "probable_diagnosis": [
    {"disease": "...", "status": "probable | suspected | confirmed",
     "reason": "short reason", "supporting_findings": ["..."], "evidence": ["CARD-ID", "..."]}
  ],
  "investigations": [
    {"test": "...", "purpose": "...", "timing_or_conditions": "... or 'not stated'", "evidence": ["CARD-ID"]}
  ],
  "medication_plan": [
    {"name": "...", "dose": "... or 'not stated'", "route": "... or 'not stated'", "frequency": "... or 'not stated'",
     "duration": "... or 'not stated'", "instructions": "... or 'not stated'",
     "patient_restrictions": "restrictions from the protocol that apply (or exclusions that block it) for THIS patient, else 'none identified'",
     "evidence": ["CARD-ID"]}
  ],
  "advice": [
    {"kind": "self_care | follow_up | warning_sign | referral", "text": "...", "evidence": ["CARD-ID"]}
  ],
  "missing_information": [
    {"item": "...", "why_it_matters": "...", "evidence": ["CARD-ID"]}
  ],
  "cannot_recommend": [
    {"item": "...", "reason": "not covered by the supplied cards / blocked by an exclusion / needs missing information"}
  ]
}"""

SYSTEM_GEN = f"""You are a clinical decision-SUPPORT tool for a general practitioner in Bangladesh. You help the clinician; you do not replace them.
You are given a patient case and a set of protocol CARDS retrieved from four guidelines (dengue, chikungunya, cholera, hypertension). The disease name is NOT given: infer plausible diagnoses from the findings.

Rules:
1. Use ONLY the supplied cards. Do not use outside medical knowledge, numbers, drugs or thresholds. If the cards do not cover something, put it under cannot_recommend.
2. Every recommendation must cite one or more card IDs from the supplied list in its "evidence" field. Never cite an ID that was not supplied.
3. Give diagnoses as "probable" or "suspected" unless the case itself contains findings or test results that the cards say confirm the diagnosis. Offer a short differential when the cards support more than one disease, and say which findings support each.
4. Respect every card's prerequisites, exclusions and decision variants. Check the patient's age, sex, weight, pregnancy, allergies, current medicines and comorbidities against them. If a variant or exclusion applies, say so in patient_restrictions and follow it.
5. When a patient detail needed to apply a card is missing, list it under missing_information (say why it matters) and still give what can safely be recommended from the evidence that is available. Do not guess the missing detail. List only details about THIS patient that are needed now: do not ask about pregnancy for males, children or when a card does not require it for this patient, and do not list later events such as discharge or follow-up results as missing. List at most the 6 most important missing details and never ask for something the case already states.
6. For medicines give name, dose, route, frequency, duration and instructions exactly as the cards state. Write "not stated" for any field the cards do not give. Never fill a field from general knowledge.
7. Investigations should say what they are for and the timing or conditions the card states. Advice should cover self-care, follow-up, warning signs and referral where the cards support it.
8. Do not recommend a medicine or action just because a card was retrieved: it must fit this patient, including any condition written on the card (for example a drug reserved for cases 'refractory to other drugs', or a step that only follows an earlier step).
9. If the findings point to a condition the cards do not cover (or only mention as a differential), say so under cannot_recommend and do not force the case into a covered disease.
10. When a card gives both a per-kg rule and a table, calculate from the per-kg rule using the patient's weight and show the calculation (for example 75 mL/kg x 12 kg = 900 mL), then choose the table row by the patient's exact age and weight; use the table only as a cross-check.
11. Return ONLY one JSON object with exactly these keys, and no other text:
{SCHEMA}
Use empty lists where nothing applies."""


def gen_prompt(patient, cards, annotations=None):
    annotations = annotations or {}
    blocks = "\n\n".join(card_for_prompt(c, annotations.get(c["card_id"])) for c in cards)
    extra = ""
    if annotations:
        extra = ("\nAn applicability check has been run on each card against this patient (see the '>> APPLICABILITY CHECK' line). "
                 "Do not base recommendations on cards marked NOT_APPLICABLE. For cards marked CONDITIONAL or INSUFFICIENT_INFO, "
                 "report the listed missing items under missing_information. Use the matched variant when one is given.\n")
    return (f"PATIENT CASE\n{patient_block(patient)}\n\n"
            f"PROTOCOL CARDS ({len(cards)} retrieved)\n{blocks}\n{extra}\n"
            "Return the JSON object now.")


SYSTEM_APPLIC = """You check whether protocol cards apply to one specific patient. You do not give treatment advice.
For each card, compare the card's findings/prerequisites, patient factors, exclusions, decision variants and "needs from patient" with the patient case.
Verdicts:
- "applies": the patient's stated findings satisfy the card's prerequisites and no exclusion is triggered.
- "conditional": the card could apply, but a detail it depends on is missing or unclear in the patient case. List the missing details.
- "not_applicable": the patient clearly does not match (findings absent or contradicted, wrong age/phase/severity, or an exclusion is triggered).
Rules: never assume a detail that is NOT PROVIDED; mark conditional instead. "missing" must list only patient details needed now, never later events (for example discharge, or results that cannot exist yet). A card about a later stage that does not apply yet is "not_applicable" with that reason. If a decision variant's condition is met, give its id in matched_variant. Judge only from the case and the card text.
Return ONLY this JSON object:
{"checks": [{"card_id": "...", "verdict": "applies | conditional | not_applicable", "reason": "one short sentence", "matched_variant": "variant id or null", "missing": ["..."]}]}
Include one entry for every card, in the order given."""


def applicability_prompt(patient, cards):
    blocks = "\n\n".join(card_for_prompt(c) for c in cards)
    return f"PATIENT CASE\n{patient_block(patient)}\n\nCARDS TO CHECK ({len(cards)})\n{blocks}\n\nReturn the JSON object now."
