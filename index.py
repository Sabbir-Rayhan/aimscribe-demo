"""Vector index over the canonical cards: embeddings in a NumPy matrix, cosine search.

With 79 cards a flat exact index is simpler and fully transparent. The same
interface can be swapped for Chroma or FAISS if the card set grows.
"""
import json
import math
import pickle
import re
from collections import Counter
import numpy as np
import config
from cards import load_cards, cards_hash, retrieval_text, TEXT_VERSION

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastEmbedder:
    name = "fastembed"

    def __init__(self, model):
        from fastembed import TextEmbedding   # ONNX runtime; no PyTorch needed
        self.model_name = model
        self.model = TextEmbedding(model_name=model)

    def fit(self, texts):
        pass

    def _norm(self, arr):
        arr = np.asarray(arr, dtype="float32")
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)

    def encode_docs(self, texts):
        return self._norm(list(self.model.embed(texts)))

    def encode_query(self, text):
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self._norm(list(self.model.embed([prefix + text])))[0]

    def state(self):
        return None


class TfidfEmbedder:
    """Offline fallback (no model download). Also used for quick tests."""
    name = "tfidf"
    model_name = "tfidf-1-2gram"

    def __init__(self, model=None, state=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = state or TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)

    def fit(self, texts):
        self.vec.fit(texts)

    def encode_docs(self, texts):
        return self.vec.transform(texts).toarray().astype("float32")

    def encode_query(self, text):
        return self.vec.transform([text]).toarray().astype("float32")[0]

    def state(self):
        return self.vec


def make_embedder(backend=None, model=None, state=None):
    backend = backend or config.EMBED_BACKEND
    model = model or config.EMBED_MODEL
    if backend == "tfidf":
        return TfidfEmbedder(state=state)
    return FastEmbedder(model)


STOP = set("a an and are as at be by for from has have in is it its of on or that the to was were with without per".split())


def tokenize(text):
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    """Plain Okapi BM25 over the card texts (no extra dependency)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype="float32")
        self.avg = float(self.len.mean()) or 1.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokenize(query)
        sc = np.zeros(len(self.docs), dtype="float32")
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in set(q):
                f = d.get(t)
                if f:
                    sc[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return sc


def _ranks(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r


class CardIndex:
    def __init__(self, ids, matrix, embedder, meta):
        self.ids, self.matrix, self.embedder, self.meta = ids, matrix, embedder, meta
        cards = {c["card_id"]: c for c in load_cards()}
        self.cards = [cards[i] for i in ids]
        self.bm25 = BM25([retrieval_text(c) for c in self.cards])

    def _dense(self, query):
        q = self.embedder.encode_query(query)
        n = np.linalg.norm(q)
        q = q / n if n else q
        return self.matrix @ q

    def search(self, query, k=None, method=None, route=None, ratio=None):
        """method: dense | bm25 | hybrid (reciprocal rank fusion of dense and BM25).
        route: keep only cards of up to N best-scoring diseases (0/None = off); ratio: extra diseases must score
        at least ratio x the best disease, so a clear single-disease case stays single and a close call keeps two."""
        k = k or config.TOP_K
        method = method or config.RETRIEVAL
        route = config.ROUTE_DISEASES if route is None else route
        ratio = config.ROUTE_RATIO if ratio is None else ratio
        if method == "dense":
            scores = self._dense(query)
        elif method == "bm25":
            scores = self.bm25.scores(query)
        else:
            rd, rb = _ranks(self._dense(query)), _ranks(self.bm25.scores(query))
            scores = (1.0 / (60 + rd) + 1.0 / (60 + rb)).astype("float32")
        order = list(np.argsort(-scores, kind="stable"))
        if route:
            by_dis = {}
            for i in order:
                by_dis.setdefault(self.cards[i]["disease"], []).append(float(scores[i]))
            dscore = {d: sum(v[:3]) for d, v in by_dis.items()}
            ranked = sorted(dscore, key=dscore.get, reverse=True)
            best = dscore[ranked[0]]
            keep = {ranked[0]} | {d for d in ranked[1:route] if dscore[d] >= ratio * best}
            order = [i for i in order if self.cards[i]["disease"] in keep]
        return [(self.ids[i], float(scores[i])) for i in order[:k]]


def build_index(backend=None, model=None, out_dir=config.INDEX_DIR):
    cards = load_cards()
    embedder = make_embedder(backend, model)
    texts = [retrieval_text(c) for c in cards]
    embedder.fit(texts)
    matrix = embedder.encode_docs(texts)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(backend=embedder.name, model=embedder.model_name, cards_hash=cards_hash(cards), n_cards=len(cards), dim=int(matrix.shape[1]), text_version=TEXT_VERSION)
    np.save(out_dir / "vectors.npy", matrix)
    (out_dir / "ids.json").write_text(json.dumps([c["card_id"] for c in cards]), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    st = embedder.state()
    if st is not None:
        with open(out_dir / "tfidf.pkl", "wb") as f:
            pickle.dump(st, f)
    return CardIndex([c["card_id"] for c in cards], matrix, embedder, meta)


def load_index(out_dir=config.INDEX_DIR, rebuild_if_stale=True):
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return build_index()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if rebuild_if_stale and (meta["cards_hash"] != cards_hash(load_cards()) or meta.get("text_version") != TEXT_VERSION):
        return build_index(meta["backend"], meta["model"])
    matrix = np.load(out_dir / "vectors.npy")
    ids = json.loads((out_dir / "ids.json").read_text(encoding="utf-8"))
    state = None
    if meta["backend"] == "tfidf":
        with open(out_dir / "tfidf.pkl", "rb") as f:
            state = pickle.load(f)
    embedder = make_embedder(meta["backend"], meta["model"], state)
    return CardIndex(ids, matrix, embedder, meta)
