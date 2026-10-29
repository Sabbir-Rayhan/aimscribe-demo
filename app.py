"""Streamlit prototype: enter a patient case, retrieve protocol cards, get recommendations.

Run:  streamlit run app.py
"""
import json
import pandas as pd
import streamlit as st

import config
from cards import load_cards
from index import load_index
from llm import GeminiClient
from pipeline import run_case

st.set_page_config(page_title="AIMScribe protocol recommender", layout="wide")

FIELDS = ["complaints", "duration", "age", "sex", "weight_kg", "pregnancy", "history", "medicines", "allergies", "findings"]
SEX_OPTIONS = ["", "Male", "Female", "Other"]
PREG_OPTIONS = ["", "Not pregnant", "Pregnant", "Unknown / may become pregnant"]


@st.cache_resource(show_spinner="Loading cards and index …")
def get_resources():
    return load_cards(), load_index()


@st.cache_resource(show_spinner=False)
def get_llm(model, use_cache):
    return GeminiClient(model=model, use_cache=use_cache)


def example_cases():
    out = {}
    for folder in (config.CASES_DIR, config.CASES_DEV_DIR):
        for p in sorted(folder.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out[f"{p.stem}: {d.get('label', '')}".strip(": ")] = d["input"]
            except Exception:
                continue
    return out


def load_example():
    name = st.session_state.get("example_pick")
    ex = example_cases().get(name)
    if ex:
        for f in FIELDS:
            st.session_state[f] = ex.get(f, "")


def clear_form():
    for f in FIELDS:
        st.session_state[f] = ""


for f in FIELDS:
    st.session_state.setdefault(f, "")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["baseline", "improved"], help=(
        "baseline: retrieve top cards and generate.\n\n"
        "improved: same, plus an applicability check of each retrieved card against the patient before generation (one extra model call)."))
    k = st.slider("Cards to retrieve (top-k)", 3, 15, config.TOP_K)
    model = st.text_input("Model", config.MODEL_NAME)
    use_cache = st.checkbox("Reuse cached model responses", value=False)
    st.caption("Same model, corpus, retriever and generation prompt in both modes.")
    st.divider()
    examples = example_cases()
    if examples:
        st.selectbox("Load a saved example case", [""] + list(examples), key="example_pick", on_change=load_example)
    st.button("Clear form", on_click=clear_form)

st.title("Protocol-grounded recommendation engine")
st.caption("Decision support for a clinician; probable diagnoses only unless findings confirm. Built on four supplied protocols (dengue, chikungunya, cholera, hypertension). Fictional demo, not for real patients.")

# ---------------- input form ----------------
st.subheader("1. Patient information")
c1, c2 = st.columns([3, 2])
with c1:
    st.text_area("Chief complaints / symptoms", key="complaints", height=110,
                 placeholder="e.g. fever with severe body aches and rash for 3 days")
    st.text_area("Examination findings, measurements, test results (optional)", key="findings", height=90,
                 placeholder="e.g. BP 150/95 and 152/96; platelet 90,000; pulse weak")
    st.text_area("Medical history", key="history", height=70)
with c2:
    a, b, c = st.columns(3)
    a.text_input("Illness duration", key="duration", placeholder="3 days")
    b.text_input("Age", key="age", placeholder="years")
    c.selectbox("Sex", SEX_OPTIONS, key="sex")
    d, e = st.columns(2)
    d.text_input("Weight (kg)", key="weight_kg")
    e.selectbox("Pregnancy status", PREG_OPTIONS, key="pregnancy")
    st.text_area("Current medicines", key="medicines", height=70)
    st.text_area("Allergies", key="allergies", height=70)

run = st.button("Get recommendations", type="primary")


def evidence_chips(ids, ref_by_id):
    if not ids:
        return ":red[no evidence cited]"
    return " ".join(f"`{ref_by_id.get(i, i + ' (not in retrieved set)')}`" for i in ids)


def render(res):
    ref_by_id = {r["card_id"]: r["ref"] for r in res["retrieved"]}
    out = res["output"]
    t = res["timing"]

    st.subheader("2. Patient information used")
    st.json({k: v for k, v in res["patient"].items() if v}, expanded=False)

    st.subheader("3. Retrieved evidence")
    df = pd.DataFrame([{"card": r["card_id"], "score": r["score"], "disease": r["disease"], "type": r["category"], "title": r["title"], "source page": r["pages"]}
                       for r in res["retrieved"]])
    st.dataframe(df, hide_index=True)
    cards_by_id = {c["card_id"]: c for c in load_cards()}
    with st.expander("Evidence passages"):
        for r in res["retrieved"]:
            c = cards_by_id[r["card_id"]]
            st.markdown(f"**{r['ref']}** — {c['title']}")
            for ev in c["evidence"][:3]:
                st.caption(f"p{ev['pdf_page']}: {ev['passage']}")

    if res["mode"] == "improved":
        st.subheader("3b. Applicability check (improved mode)")
        if res["applicability"]:
            adf = pd.DataFrame([{"card": cid, "verdict": a["verdict"], "variant": a["matched_variant"] or "", "reason": a["reason"],
                                 "missing": "; ".join(a["missing"])} for cid, a in res["applicability"].items()])
            st.dataframe(adf, hide_index=True)
        else:
            st.warning(f"Applicability check unavailable: {res['applicability_error']}. Generation ran without it.")

    if res["generation_error"]:
        st.error(f"Model output problem: {res['generation_error']}")

    st.subheader("4. Probable diagnosis / differential")
    for d in out["probable_diagnosis"]:
        st.markdown(f"**{d.get('disease', '?')}** — _{d.get('status', 'probable')}_")
        st.write(d.get("reason", ""))
        if d.get("supporting_findings"):
            st.caption("Supporting findings: " + "; ".join(map(str, d["supporting_findings"])))
        st.markdown("Evidence: " + evidence_chips(d["evidence"], ref_by_id))
    if not out["probable_diagnosis"]:
        st.info("No diagnosis could be supported from the retrieved cards.")

    st.subheader("5. Investigations")
    for i in out["investigations"]:
        st.markdown(f"**{i.get('test', '?')}** — {i.get('purpose', '')}")
        st.caption(f"Timing / conditions: {i.get('timing_or_conditions', 'not stated')}")
        st.markdown("Evidence: " + evidence_chips(i["evidence"], ref_by_id))

    st.subheader("6. Medication planning")
    for m in out["medication_plan"]:
        st.markdown(f"**{m.get('name', '?')}**")
        st.table(pd.DataFrame([{k2: m.get(k2, "not stated") for k2 in ["dose", "route", "frequency", "duration", "instructions"]}]))
        st.write("Patient restrictions: " + str(m.get("patient_restrictions", "none identified")))
        st.markdown("Evidence: " + evidence_chips(m["evidence"], ref_by_id))

    st.subheader("7. Advice")
    for a in out["advice"]:
        st.markdown(f"**{a.get('kind', 'advice').replace('_', ' ')}:** {a.get('text', '')}")
        st.markdown("Evidence: " + evidence_chips(a["evidence"], ref_by_id))

    st.subheader("8. Missing information")
    for mi in out["missing_information"]:
        st.markdown(f"- **{mi.get('item', '?')}** — {mi.get('why_it_matters', '')}  " + evidence_chips(mi["evidence"], ref_by_id))
    if not out["missing_information"]:
        st.caption("None reported.")
    if out["cannot_recommend"]:
        with st.expander("Not covered / cannot recommend"):
            for cr in out["cannot_recommend"]:
                st.markdown(f"- **{cr.get('item', '?')}** — {cr.get('reason', '')}")

    st.divider()
    cc = res["citation_check"]
    if cc["not_in_retrieved_set"] or cc["uncited_items"]:
        st.warning(f"Citation check: {len(cc['not_in_retrieved_set'])} citation(s) not in the retrieved set; {len(cc['uncited_items'])} item(s) without a citation.")
    else:
        st.caption("Citation check: every item cites a retrieved card.")
    st.caption(f"Mode: {res['mode']} | model: {res['model']} | model calls: {t['model_calls']} | model time: {t['model_seconds']} s | total: {t['total_seconds']} s | tokens in/out: {t['input_tokens']}/{t['output_tokens']}")
    with st.expander("Raw result JSON"):
        st.json(res)


if run:
    patient = {f: st.session_state.get(f, "") for f in FIELDS}
    if not any(str(patient[f]).strip() for f in ["complaints", "findings"]):
        st.warning("Enter at least the chief complaints or some findings.")
    else:
        cards, idx = get_resources()
        try:
            llm = get_llm(model, use_cache)
        except Exception as ex:   # noqa: BLE001
            st.error(f"Model connection failed: {ex}")
            st.stop()
        with st.spinner("Retrieving evidence and generating recommendations …"):
            res = run_case(patient, mode, llm, idx, k=k, cards=cards)
        st.session_state["last_result"] = res

if st.session_state.get("last_result"):
    render(st.session_state["last_result"])
