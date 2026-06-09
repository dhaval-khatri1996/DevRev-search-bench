"""
Bare Metal DevRev Search Pipeline v2
--------------------------------------
Design philosophy (HRT-inspired):
  Stage 0 : Annotation boost — learned prior from labelled queries
  Stage 1 : BGE-M3 sparse (neural BM25) + classic BM25 (fallback/retrainable)
            → weighted RRF → top 150 candidates
  Stage 2 : BGE-M3 dense, POS-grouped queries
            → groups selected empirically on annotated queries
            → weighted RRF → top 30
  Stage 3 : bge-reranker-v2-m3 cross-encoder → top 10
            + article cap

"Bare metal" = transparent code, minimal black boxes,
               every component replaceable or retrainable.

Key design decisions:
  - BM25 is KEPT alongside BGE-M3 sparse. Reason: BM25 can be rebuilt
    instantly on any new/local corpus. BGE-M3 sparse is fixed (pre-trained).
    On domain-specific data not in BGE-M3 training, BM25 may outperform.
  - Annotation boost is multiplicative, not additive. This means the prior
    can only AMPLIFY existing retrieval signal, never create signal from
    nothing. Graceful degradation for unseen query terms.
  - POS groups are selected empirically on annotated queries, not fixed.
    Only groups that improve (or don't hurt) recall are kept.
  - All caches are stored to .cache/ — rebuild any component independently.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import argparse
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import spacy
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel   # pip install FlagEmbedding
from FlagEmbedding import FlagReranker
from tqdm import tqdm

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── constants ─────────────────────────────────────────────────────────────────
BGE_M3_MODEL  = "BAAI/bge-m3"
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"

BM25_TOP_K    = 150    # Stage 1 candidates
EMBED_TOP_K   = 30     # Stage 2 candidates
FINAL_K       = 10     # final results per query
ARTICLE_CAP   = 2      # max chunks per parent article
RRF_K         = 60

# Keep POS groups within this recall gap of full-text baseline
POS_TOLERANCE = 0.02

# Domain stopwords — high-frequency noise specific to DevRev corpus
# (derived from corpus token frequency analysis)
DOMAIN_STOPWORDS = {
    "devrev", "ai", "use", "with", "by", "in", "and", "the", "a", "an",
    "of", "to", "for", "is", "are", "that", "this", "it", "be", "was",
    "on", "at", "from", "or", "as", "not", "but", "we", "you", "they",
    "has", "have", "had", "will", "can", "do", "does", "did", "g",
}

# All candidate POS groups — evaluated empirically, best subset selected
# Applied to QUERY only (document titles/fragments don't parse reliably)
POS_GROUPS: dict[str, Optional[set[str]]] = {
    "noun_verb" : {"NOUN", "PROPN", "VERB"},
    "noun_only" : {"NOUN", "PROPN"},
    "noun_adj"  : {"NOUN", "PROPN", "ADJ"},
    "verb_adv"  : {"VERB", "ADV"},
    "full"      : None,   # always kept as baseline
}

# Linguistically motivated default weights
# overridden after empirical POS selection if needed
DEFAULT_WEIGHTS: dict[str, float] = {
    "noun_verb" : 1.2,
    "noun_only" : 1.0,
    "noun_adj"  : 0.8,
    "verb_adv"  : 0.5,
    "full"      : 1.0,
}

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def fix_byte_string(text: str) -> str:
    """Fix accidentally serialised Python byte-string literals like b'...'"""
    if text.startswith(("b'", 'b"')) and text.endswith(("'", '"')):
        try:
            literal = ast.literal_eval(text)
            if isinstance(literal, bytes):
                return literal.decode("utf-8", errors="replace")
        except Exception:
            pass
    return text


def clean_text(text: str) -> str:
    """Full preprocessing pass."""
    text = fix_byte_string(text)
    text = re.sub(r"(?i)^devrev\s*\|\s*", "", text)   # strip "DevRev | " prefix
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\|\s*", " ", text)              # remove remaining pipes
    text = re.sub(r"https?://\S+", "", text)           # remove URLs
    text = re.sub(r"[^\w\s\-.,!?]", " ", text)        # remove punctuation clusters
    return re.sub(r"\s+", " ", text).strip()


def tokenize_bm25(text: str) -> list[str]:
    """Lowercase word tokens with domain stopwords removed."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in DOMAIN_STOPWORDS and len(t) > 1]


# ══════════════════════════════════════════════════════════════════════════════
# 2. POS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def extract_pos_group(query: str, pos_tags: Optional[set[str]]) -> str:
    """Filter query tokens by POS. Falls back to full query if result is empty."""
    if pos_tags is None:
        return query
    doc    = nlp(query)
    tokens = [tok.text for tok in doc if tok.pos_ in pos_tags]
    return " ".join(tokens) if tokens else query


def get_all_pos_groups(query: str) -> dict[str, str]:
    """Return all POS group variants of a query in one spacy pass."""
    doc    = nlp(query)
    result = {}
    for name, pos_tags in POS_GROUPS.items():
        if pos_tags is None:
            result[name] = query
        else:
            tokens = [tok.text for tok in doc if tok.pos_ in pos_tags]
            result[name] = " ".join(tokens) if tokens else query
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. RRF + ARTICLE CAP
# ══════════════════════════════════════════════════════════════════════════════

def weighted_rrf(
    ranked_lists: list[tuple[list[int], float]],
    k: int = RRF_K,
) -> list[int]:
    fused: dict[int, float] = {}
    for ranked_ids, weight in ranked_lists:
        for rank, doc_idx in enumerate(ranked_ids, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)
    return sorted(fused, key=lambda i: fused[i], reverse=True)


def article_id(chunk_id: str) -> str:
    return chunk_id.split("_KNOWLEDGE_NODE")[0]


def apply_article_cap(
    items: list[dict],
    k_final: int = FINAL_K,
    cap: int = ARTICLE_CAP,
) -> list[dict]:
    results: list[dict] = []
    per_article: Counter = Counter()
    for item in items:
        art = article_id(item["id"])
        if per_article[art] >= cap:
            continue
        per_article[art] += 1
        results.append(item)
        if len(results) >= k_final:
            break
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANNOTATION BOOST  (Stage 0)
# ══════════════════════════════════════════════════════════════════════════════

class AnnotationBoost:
    """
    Learns a term → article relevance prior from annotated queries.

    Scoring:
        final_score = raw_score × (1 + λ × prior(term, article))

    Multiplicative design rationale:
        - Prior can only AMPLIFY existing signal, never create it from nothing.
          If raw_score = 0, final_score = 0 regardless of prior.
        - Graceful degradation: terms with no annotation → boost = 1.0 (no effect).
        - λ controls annotation influence: λ=0 → pure retrieval, λ=1 → strong nudge.

    Sparse data handling:
        - Count-based probability is appropriate here (not gradient descent).
        - λ tuned by grid search on annotated queries (not on blind test set).
        - Multiplicative form prevents annotation from dominating on rare terms.
    """

    def __init__(self, lambda_: float = 0.3) -> None:
        self.lambda_             = lambda_
        self.prior               : dict[str, dict[str, float]] = {}
        self.article_to_indices  : dict[str, list[int]] = {}

    def fit(self, annotated_queries: list[dict], doc_ids: list[str]) -> None:
        """Build prior from annotated queries."""
        self.article_to_indices = defaultdict(list)
        for idx, did in enumerate(doc_ids):
            self.article_to_indices[article_id(did)].append(idx)

        counts: dict[str, Counter] = defaultdict(Counter)
        for item in annotated_queries:
            terms       = set(tokenize_bm25(item["query"]))
            golden_arts = {article_id(r["id"]) for r in item["retrievals"]}
            for term in terms:
                for art in golden_arts:
                    counts[term][art] += 1

        self.prior = {}
        for term, art_counts in counts.items():
            total = sum(art_counts.values())
            self.prior[term] = {art: cnt / total for art, cnt in art_counts.items()}

    def boost_scores(
        self,
        raw_scores: np.ndarray,
        query_terms: list[str],
    ) -> np.ndarray:
        """Apply multiplicative annotation boost. O(terms × matched_articles)."""
        boost = np.ones(len(raw_scores), dtype=np.float32)
        for term in query_terms:
            if term not in self.prior:
                continue
            for art, prior_score in self.prior[term].items():
                for idx in self.article_to_indices.get(art, []):
                    boost[idx] += self.lambda_ * prior_score
        return raw_scores * boost

    def tune_lambda(
        self,
        annotated_queries: list[dict],
        score_fn,
        candidates: list[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    ) -> float:
        """Grid search λ. Optimises recall@150 — Stage 1's only job is not to miss."""
        print("Tuning annotation boost λ...")
        best_lambda, best_score = 0.0, -1.0
        for lam in candidates:
            self.lambda_ = lam
            mean = float(np.mean([score_fn(item) for item in annotated_queries]))
            print(f"  λ={lam:.2f}  recall@150={mean:.4f}")
            if mean > best_score:
                best_score, best_lambda = mean, lam
        self.lambda_ = best_lambda
        print(f"  → best λ={best_lambda:.2f}  recall@150={best_score:.4f}")
        return best_lambda

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"lambda": self.lambda_, "prior": self.prior,
                         "article_to_indices": dict(self.article_to_indices)}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.lambda_            = data["lambda"]
        self.prior              = data["prior"]
        self.article_to_indices = defaultdict(list, data["article_to_indices"])


# ══════════════════════════════════════════════════════════════════════════════
# 5. SPARSE RETRIEVER — BGE-M3 sparse + classic BM25
# ══════════════════════════════════════════════════════════════════════════════

class SparseRetriever:
    """
    Primary  : BGE-M3 learned sparse weights (neural BM25).
               Understands that 'AirSync' and 'sync' are related.
               Fixed — cannot be retrained.

    Fallback : Classic BM25.
               Can be rebuilt instantly on any new/local corpus.
               Better than BGE-M3 sparse on highly domain-specific
               terminology not seen during BGE-M3 training.

    Both signals run and are fused via RRF.
    Disable BGE-M3 sparse with use_bge_sparse=False to use BM25 only.
    """

    def __init__(self, use_bge_sparse: bool = True) -> None:
        self.use_bge_sparse = use_bge_sparse
        self.bm25            : Optional[BM25Okapi]   = None
        self.sparse_vecs     : Optional[list[dict]]  = None

    def build_bm25(self, documents: list[str]) -> None:
        print("Building BM25 index...")
        tok = [tokenize_bm25(d) for d in tqdm(documents, desc="BM25 tokenize")]
        self.bm25 = BM25Okapi(tok)

    def encode_corpus_sparse(
        self, model: BGEM3FlagModel, documents: list[str], batch_size: int = 32
    ) -> None:
        """One-time sparse encoding of all documents."""
        print("Encoding BGE-M3 sparse vectors (one-time)...")
        self.sparse_vecs = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Sparse encode"):
            out = model.encode(
                documents[i: i + batch_size],
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            self.sparse_vecs.extend(out["lexical_weights"])

    def bge_sparse_scores(
        self, model: BGEM3FlagModel, query: str
    ) -> np.ndarray:
        """Dot product of query sparse vec against all doc sparse vecs."""
        q_out = model.encode(
            [query],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        q_vec: dict = q_out["lexical_weights"][0]
        scores = np.zeros(len(self.sparse_vecs), dtype=np.float32)
        for token, q_w in q_vec.items():
            for idx, doc_vec in enumerate(self.sparse_vecs):
                if token in doc_vec:
                    scores[idx] += float(q_w) * float(doc_vec[token])
        return scores

    def get_scores(
        self,
        model: Optional[BGEM3FlagModel],
        query: str,
    ) -> np.ndarray:
        if self.use_bge_sparse and self.sparse_vecs is not None and model:
            return self.bge_sparse_scores(model, query)
        assert self.bm25, "BM25 not built — call build_bm25() first"
        return np.array(self.bm25.get_scores(tokenize_bm25(query)), dtype=np.float32)

    def save(self, sparse_path: str, bm25_path: str) -> None:
        if self.sparse_vecs:
            with open(sparse_path, "wb") as f:
                pickle.dump(self.sparse_vecs, f)
        if self.bm25:
            with open(bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)

    def load(self, sparse_path: str, bm25_path: str) -> None:
        if Path(sparse_path).exists():
            with open(sparse_path, "rb") as f:
                self.sparse_vecs = pickle.load(f)
        if Path(bm25_path).exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 6. POS GROUP SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

class POSGroupSelector:
    """
    Evaluates each POS group independently on annotated queries.
    Keeps groups within `tolerance` recall of the full-text baseline.
    'full' is always kept.

    Why empirical selection matters:
        - 'verb_adv' group may hurt if DevRev queries are mostly noun-heavy.
        - 'noun_adj' may help for queries like "slow API response".
        - Let the data decide, not assumptions.
    """

    def __init__(self, tolerance: float = POS_TOLERANCE) -> None:
        self.tolerance       = tolerance
        self.selected_groups = list(POS_GROUPS.keys())
        self.group_scores    : dict[str, float] = {}

    def select(
        self,
        annotated_queries: list[dict],
        score_fn,   # (query_variant: str, golden: set[str]) → recall@10
    ) -> list[str]:
        print("\nEvaluating POS groups on annotated queries...")
        for group_name, pos_tags in POS_GROUPS.items():
            recalls = []
            for item in annotated_queries:
                variant = extract_pos_group(item["query"], pos_tags)
                golden  = {r["id"] for r in item["retrievals"]}
                recalls.append(score_fn(variant, golden))
            self.group_scores[group_name] = float(np.mean(recalls))
            print(f"  {group_name:<15}  recall@10 = {self.group_scores[group_name]:.4f}")

        baseline = self.group_scores["full"]
        self.selected_groups = [
            name for name, sc in self.group_scores.items()
            if sc >= baseline - self.tolerance
        ]
        if "full" not in self.selected_groups:
            self.selected_groups.append("full")

        print(f"\n  Baseline (full text): {baseline:.4f}")
        print(f"  Selected groups:      {self.selected_groups}")
        return self.selected_groups

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"selected": self.selected_groups,
                       "scores":   self.group_scores}, f, indent=2)

    def load(self, path: str) -> None:
        data = json.load(open(path))
        self.selected_groups = data["selected"]
        self.group_scores    = data["scores"]


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class BareMetalSearch:

    def __init__(
        self,
        knowledge_base_path : str,
        cache_dir            : str = ".cache",
        weights              : Optional[dict[str, float]] = None,
        annotation_lambda    : float = 0.3,
        use_bge_sparse       : bool  = True,
    ) -> None:
        self.cache   = Path(cache_dir)
        self.cache.mkdir(exist_ok=True)
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # transparent, replaceable components
        self.sparse    = SparseRetriever(use_bge_sparse=use_bge_sparse)
        self.ann_boost = AnnotationBoost(lambda_=annotation_lambda)
        self.pos_sel   = POSGroupSelector()

        self._load_corpus(knowledge_base_path)
        self._load_models()
        self._build_indexes()

    # ── corpus ────────────────────────────────────────────────────────────────

    def _load_corpus(self, path: str) -> None:
        print("Loading corpus...")
        if path == "hf":
            from datasets import load_dataset
            raw = list(load_dataset("devrev/search", "knowledge_base", split="corpus"))
        else:
            with open(path) as f:
                raw = json.load(f)

        self.doc_ids    : list[str] = []
        self.doc_titles : list[str] = []
        self.doc_texts  : list[str] = []
        self.documents  : list[str] = []

        for item in tqdm(raw, desc="Preprocessing"):
            title = clean_text(item.get("title", ""))
            text  = clean_text(item.get("text",  ""))
            self.doc_ids.append(item["id"])
            self.doc_titles.append(title)
            self.doc_texts.append(text)
            self.documents.append(f"{title}\n\n{text}".strip())

        print(f"Corpus: {len(self.doc_ids):,} chunks")

    # ── models ────────────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        print("Loading BGE-M3 (use_fp16 halves memory, negligible quality loss)...")
        self.bge      = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)
        print("Loading reranker...")
        self.reranker = FlagReranker(RERANK_MODEL, use_fp16=True)

    # ── indexes ───────────────────────────────────────────────────────────────

    def _build_indexes(self) -> None:
        sparse_path = str(self.cache / "sparse_vecs.pkl")
        bm25_path   = str(self.cache / "bm25.pkl")
        dense_path  = str(self.cache / "dense_embeddings.npy")
        ids_path    = str(self.cache / "doc_ids.pkl")

        # check if cached ids match current corpus
        ids_match = False
        if Path(ids_path).exists():
            with open(ids_path, "rb") as f:
                ids_match = (pickle.load(f) == self.doc_ids)

        # BM25 — always built (fast, retrainable on new data)
        if Path(bm25_path).exists() and ids_match:
            print("Loading BM25 from cache...")
            with open(bm25_path, "rb") as f:
                self.sparse.bm25 = pickle.load(f)
        else:
            self.sparse.build_bm25(self.documents)
            with open(bm25_path, "wb") as f:
                pickle.dump(self.sparse.bm25, f)

        # BGE-M3 sparse — one-time encode
        if Path(sparse_path).exists() and ids_match:
            print("Loading BGE-M3 sparse vectors from cache...")
            with open(sparse_path, "rb") as f:
                self.sparse.sparse_vecs = pickle.load(f)
        else:
            self.sparse.encode_corpus_sparse(self.bge, self.documents)
            with open(sparse_path, "wb") as f:
                pickle.dump(self.sparse.sparse_vecs, f)

        # BGE-M3 dense — one-time encode
        if Path(dense_path).exists() and ids_match:
            print("Loading dense embeddings from cache...")
            self.embeddings = np.load(dense_path)
        else:
            self._encode_dense(dense_path)

        # save ids for future cache validation
        with open(ids_path, "wb") as f:
            pickle.dump(self.doc_ids, f)

    def _encode_dense(self, dense_path: str) -> None:
        print("Encoding dense embeddings (BGE-M3, one-time)...")
        batches, batch_size = [], 32
        for i in tqdm(range(0, len(self.documents), batch_size), desc="Dense encode"):
            out = self.bge.encode(
                self.documents[i: i + batch_size],
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            batches.append(out["dense_vecs"])
        emb = np.vstack(batches).astype(np.float32)
        # L2 normalise so cosine sim = dot product at search time
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb  /= np.clip(norms, 1e-9, None)
        self.embeddings = emb
        np.save(dense_path, emb)

    # ── annotation boost fitting ──────────────────────────────────────────────

    def fit_annotation_boost(self, annotated_queries: list[dict]) -> None:
        """Fit prior and tune λ. Saves result to cache."""
        cache_path = str(self.cache / "annotation_boost.pkl")
        if Path(cache_path).exists():
            print("Loading annotation boost from cache...")
            self.ann_boost.load(cache_path)
            return

        self.ann_boost.fit(annotated_queries, self.doc_ids)

        def recall_at_150(item: dict) -> float:
            terms   = tokenize_bm25(item["query"])
            raw     = self.sparse.get_scores(self.bge, item["query"])
            boosted = self.ann_boost.boost_scores(raw, terms)
            top150  = set(np.argsort(boosted)[::-1][:150].tolist())
            golden  = {i for i, d in enumerate(self.doc_ids)
                       if d in {r["id"] for r in item["retrievals"]}}
            return len(golden & top150) / len(golden) if golden else 0.0

        self.ann_boost.tune_lambda(annotated_queries, recall_at_150)
        self.ann_boost.save(cache_path)

    # ── POS group selection ───────────────────────────────────────────────────

    def select_pos_groups(self, annotated_queries: list[dict]) -> list[str]:
        """Evaluate and select POS groups. Saves result to cache."""
        cache_path = str(self.cache / "selected_groups.json")
        if Path(cache_path).exists():
            print("Loading POS group selection from cache...")
            self.pos_sel.load(cache_path)
            return self.pos_sel.selected_groups

        def score_fn(query_variant: str, golden: set[str]) -> float:
            out   = self.bge.encode(
                [query_variant],
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            q_emb = out["dense_vecs"][0].astype(np.float32)
            q_emb /= max(np.linalg.norm(q_emb), 1e-9)
            sims   = self.embeddings @ q_emb
            top10  = set(np.argsort(sims)[::-1][:10].tolist())
            golden_idx = {i for i, d in enumerate(self.doc_ids) if d in golden}
            return len(golden_idx & top10) / len(golden_idx) if golden_idx else 0.0

        selected = self.pos_sel.select(annotated_queries, score_fn)
        self.pos_sel.save(cache_path)
        return selected

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k_final: int = FINAL_K) -> list[dict]:
        query = clean_text(query)
        terms = tokenize_bm25(query)

        # ── Stage 1: sparse + BM25 fused, both annotation-boosted ────────
        # BGE-M3 sparse (primary — neural, understands term relationships)
        sparse_scores  = self.sparse.get_scores(self.bge, query)
        sparse_boosted = self.ann_boost.boost_scores(sparse_scores, terms)
        sparse_ranked  = list(np.argsort(sparse_boosted)[::-1][:BM25_TOP_K])

        # classic BM25 (secondary — transparent, retrainable)
        bm25_scores  = np.array(
            self.sparse.bm25.get_scores(terms), dtype=np.float32
        )
        bm25_boosted = self.ann_boost.boost_scores(bm25_scores, terms)
        bm25_ranked  = list(np.argsort(bm25_boosted)[::-1][:BM25_TOP_K])

        # RRF: sparse weighted higher, BM25 as supporting signal
        stage1 = weighted_rrf([
            (sparse_ranked, 1.5),
            (bm25_ranked,   0.8),
        ])[:BM25_TOP_K]

        # ── Stage 2: POS-grouped dense embeddings (selected groups only) ─
        candidate_embs = self.embeddings[stage1]   # (150, 1024)
        pos_groups     = get_all_pos_groups(query)
        ranked_lists   : list[tuple[list[int], float]] = []

        for group_name in self.pos_sel.selected_groups:
            group_query = pos_groups.get(group_name, query)
            if not group_query.strip():
                continue
            out   = self.bge.encode(
                [group_query],
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            q_emb = out["dense_vecs"][0].astype(np.float32)
            q_emb /= max(np.linalg.norm(q_emb), 1e-9)

            sims  = candidate_embs @ q_emb              # (150,)
            ranked = [stage1[i] for i in np.argsort(sims)[::-1]]
            ranked_lists.append((ranked, self.weights.get(group_name, 1.0)))

        stage2 = weighted_rrf(ranked_lists)[:EMBED_TOP_K]

        # ── Stage 3: cross-encoder rerank ────────────────────────────────
        pairs  = [(query, self.documents[i]) for i in stage2]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        reranked = sorted(zip(stage2, scores), key=lambda x: x[1], reverse=True)
        results  = [
            {"id"   : self.doc_ids[i],
             "title": self.doc_titles[i],
             "text" : self.doc_texts[i],
             "score": float(s)}
            for i, s in reranked
        ]
        return apply_article_cap(results, k_final=k_final, cap=ARTICLE_CAP)


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    pipeline: BareMetalSearch,
    annotated_queries: list[dict],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    metrics: dict[str, list[float]] = defaultdict(list)
    ndcg_scores: list[float] = []

    for item in tqdm(annotated_queries, desc="Evaluating"):
        golden = {r["id"] for r in item["retrievals"]}
        preds  = [r["id"] for r in pipeline.search(item["query"])]

        for k in ks:
            top_k = preds[:k]
            hits  = len(golden & set(top_k))
            metrics[f"hit_rate@{k}"].append(1.0 if hits > 0 else 0.0)
            metrics[f"recall@{k}"].append(hits / len(golden) if golden else 0.0)
            metrics[f"precision@{k}"].append(hits / k)
            rr = next(
                (1.0 / (r + 1) for r, pid in enumerate(top_k) if pid in golden),
                0.0,
            )
            metrics[f"mrr@{k}"].append(rr)

        gains = [1.0 if pid in golden else 0.0 for pid in preds[:10]]
        dcg   = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
        ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(golden), 10)))
        ndcg_scores.append(dcg / ideal if ideal > 0 else 0.0)

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary["ndcg@10"]         = float(np.mean(ndcg_scores))
    summary["selection_score"] = (summary["recall@10"] + summary["precision@10"]) / 2
    return summary


def print_summary(summary: dict[str, float]) -> None:
    print("=" * 44)
    for key in ("hit_rate@10", "recall@10", "precision@10",
                "mrr@10", "ndcg@10", "selection_score"):
        print(f"  {key:<18}  {summary[key]:.4f}")
    print("=" * 44)


# ══════════════════════════════════════════════════════════════════════════════
# 9. SUBMISSION WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_submission(
    pipeline: BareMetalSearch,
    test_queries: list[dict],
    output_path: str = "submission.json",
) -> None:
    results = []
    for item in tqdm(test_queries, desc="Generating submission"):
        preds = pipeline.search(item["query"])
        results.append({
            "query_id"  : item["query_id"],
            "query"     : item["query"],
            "retrievals": [
                {"id": r["id"], "title": r["title"], "text": r["text"]}
                for r in preds
            ],
        })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Submission written → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Bare Metal DevRev Search v2")
    parser.add_argument("--kb",            default="hf",
                        help="Path to knowledge_base.json or 'hf'")
    parser.add_argument("--cache-dir",     default=".cache")
    parser.add_argument("--no-bge-sparse", action="store_true",
                        help="Disable BGE-M3 sparse, use BM25 only for Stage 1")
    parser.add_argument("--skip-tune",     action="store_true",
                        help="Skip annotation boost + POS selection (use cache)")
    parser.add_argument("--output",        default="submission.json")
    args = parser.parse_args()

    pipeline = BareMetalSearch(
        knowledge_base_path=args.kb,
        cache_dir=args.cache_dir,
        use_bge_sparse=not args.no_bge_sparse,
    )

    print("Loading queries...")
    from datasets import load_dataset
    annotated = list(load_dataset("devrev/search", "annotated_queries", split="train"))
    test_q    = list(load_dataset("devrev/search", "test_queries",      split="test"))

    if not args.skip_tune:
        pipeline.fit_annotation_boost(annotated)
        pipeline.select_pos_groups(annotated)
    else:
        sel_cache = Path(args.cache_dir) / "selected_groups.json"
        if sel_cache.exists():
            pipeline.pos_sel.load(str(sel_cache))
        boost_cache = Path(args.cache_dir) / "annotation_boost.pkl"
        if boost_cache.exists():
            pipeline.ann_boost.load(str(boost_cache))

    print("\nSanity check on annotated queries...")
    print_summary(evaluate(pipeline, annotated))

    write_submission(pipeline, test_q, output_path=args.output)


if __name__ == "__main__":
    main()
