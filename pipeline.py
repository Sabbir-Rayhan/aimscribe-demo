"""Recommendation pipeline with two modes of the SAME system.

baseline : retrieve top-k cards -> generate (one model call)
improved : retrieve top-k cards -> applicability check (extra model call) -> generate with the
           applicability annotations attached to each card

Everything else (corpus, index, retriever, model, temperature, generation prompt) is shared.
"""
import time
import config
from cards import load_cards, short_ref, page_label
from llm import parse_json
from prompts import SYSTEM_GEN, SYSTEM_APPLIC, gen_prompt, applicability_prompt, patient_query

SECTIONS = ["probable_diagnosis", "investigations", "medication_plan", "advice", "missing_information", "cannot_recommend"]
EVIDENCE_SECTIONS = ["probable_diagnosis", "investigations", "medication_plan", "advice", "missing_information"]


def _as_list(x):
    return x if isinstance(x, list) else ([] if x in (None, "") else [x])


def normalise_output(obj):
    obj = obj if isinstance(obj, dict) else {}
    out = {}
    for s in SECTIONS:
        items = []
        for it in _as_list(obj.get(s)):
            if not isinstance(it, dict):
                it = {"text": str(it)}
            if s in EVIDENCE_SECTIONS:
                ev = it.get("evidence")
                it["evidence"] = [str(e).strip() for e in _as_list(ev)]
            items.append(it)
        out[s] = items
    return out


def check_citations(output, retrieved_ids):
    """Deterministic post-check: every cited ID must be one of the retrieved cards."""
    bad, uncited = [], []
    for s in EVIDENCE_SECTIONS:
        for i, it in enumerate(output[s]):
            ev = it.get("evidence", [])
            if not ev:
                uncited.append(f"{s}[{i}]")
            for e in ev:
                if e not in retrieved_ids:
                    bad.append(f"{s}[{i}]: {e}")
    return dict(invalid_citations=bad, uncited_items=uncited)


def _call(llm, system, prompt):
    """One model call; if the reply is not valid JSON, ask once more (the retry is counted and reported)."""
    r = llm.generate(system, prompt)
    obj, err = parse_json(r.text) if not r.error else (None, r.error)
    if obj is None and not r.error:
        r2 = llm.generate(system, prompt + "\n\nYour previous reply was not valid JSON. Return ONLY one valid JSON object (double-quoted keys and strings, no trailing commas).")
        obj2, err2 = parse_json(r2.text) if not r2.error else (None, r2.error)
        r2.retries = (r.retries or 0) + 1
        r2.elapsed += r.elapsed
        r2.input_tokens = (r2.input_tokens or 0) + (r.input_tokens or 0)
        r2.output_tokens = (r2.output_tokens or 0) + (r.output_tokens or 0)
        return r2, obj2, err2
    return r, obj, err


def run_case(patient, mode, llm, index, k=None, cards=None):
    assert mode in ("baseline", "improved")
    k = k or config.TOP_K
    cards = cards or load_cards()
    by_id = {c["card_id"]: c for c in cards}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    hits = index.search(patient_query(patient), k)
    t_retrieve = time.perf_counter() - t0
    retrieved = [by_id[cid] for cid, _ in hits]
    scores = {cid: s for cid, s in hits}

    calls = []        # one record per model call
    annotations, applic_err = None, None
    if mode == "improved":
        r, obj, err = _call(llm, SYSTEM_APPLIC, applicability_prompt(patient, retrieved))
        calls.append(dict(step="applicability_check", elapsed=r.elapsed, input_tokens=r.input_tokens,
                          output_tokens=r.output_tokens, cached=r.cached, retries=r.retries))
        if obj and isinstance(obj.get("checks"), list):
            annotations = {}
            for ch in obj["checks"]:
                if isinstance(ch, dict) and ch.get("card_id") in by_id:
                    v = str(ch.get("verdict", "")).lower().strip()
                    annotations[ch["card_id"]] = dict(
                        verdict=v if v in ("applies", "conditional", "not_applicable") else "conditional",
                        reason=str(ch.get("reason", "")), matched_variant=ch.get("matched_variant") or None,
                        missing=[str(m) for m in _as_list(ch.get("missing"))])
        else:
            applic_err = err or "applicability check returned no usable checks"

    r, obj, err = _call(llm, SYSTEM_GEN, gen_prompt(patient, retrieved, annotations))
    calls.append(dict(step="generation", elapsed=r.elapsed, input_tokens=r.input_tokens,
                      output_tokens=r.output_tokens, cached=r.cached, retries=r.retries,
                      finish_reason=r.finish_reason))
    output = normalise_output(obj)
    cit = check_citations(output, set(by_id))
    cit_retrieved = check_citations(output, {c["card_id"] for c in retrieved})

    total = time.perf_counter() - t_start
    return dict(
        mode=mode, model=getattr(llm, "model", "?"), k=k,
        patient=patient,
        retrieved=[dict(card_id=c["card_id"], score=round(scores[c["card_id"]], 4), title=c["title"],
                        disease=c["disease"], category=c["clinical_category"], ref=short_ref(c), pages=page_label(c))
                   for c in retrieved],
        applicability=annotations,
        applicability_error=applic_err,
        output=output,
        citation_check=dict(not_in_retrieved_set=cit_retrieved["invalid_citations"],
                            not_in_corpus=cit["invalid_citations"], uncited_items=cit_retrieved["uncited_items"]),
        generation_error=err,
        timing=dict(total_seconds=round(total, 2), retrieval_seconds=round(t_retrieve, 3),
                    model_seconds=round(sum(c["elapsed"] for c in calls), 2), model_calls=len(calls),
                    input_tokens=sum(c["input_tokens"] or 0 for c in calls),
                    output_tokens=sum(c["output_tokens"] or 0 for c in calls), calls=calls),
    )
