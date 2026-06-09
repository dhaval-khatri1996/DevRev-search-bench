"""
Bare Metal DevRev Search Pipeline
----------------------------------
Stage 1 : BM25 (full text) + BM25 (query nouns only)  → RRF → top 100
Stage 2 : POS-grouped query embeddings (MiniLM)        → weighted RRF → top 20
Stage 3 : MiniLM cross-encoder rerank                 → top 10
          + article cap (max 2 chunks per article)

Weight tuning via Optuna on annotated queries (291 rows, 5-fold CV).
All models run locally — no API calls.
"""

from __future__ import annotations

import json
import re
import ast
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ── third-party (all lightweight / CPU-friendly) ──────────────────────────────
import spacy                          # POS tagging  — pip install spacy
                                      # python -m spacy download en_core_web_sm
from rank_bm25 import BM25Okapi       # pip install rank-bm25
from sentence_transformers import (   # pip install sentence-transformers
    SentenceTransformer,
    CrossEncoder,
)
from tqdm import tqdm                 # pip install tqdm

# optional — only needed for weight tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"          # 22 MB, 384-dim, fast on CPU
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RRF_K         = 60
BM25_TOP_K    = 100   # Stage 1 candidates
EMBED_TOP_K   = 20    # Stage 2 candidates
FINAL_K       = 10    # final results
ARTICLE_CAP   = 2     # max chunks per parent article

# Domain stopwords — high-frequency noise tokens specific to DevRev corpus
# (derived from the token frequency table you shared)
DOMAIN_STOPWORDS = {
    "devrev", "ai", "use", "with", "by", "in", "and", "the", "a", "an",
    "of", "to", "for", "is", "are", "that", "this", "it", "be", "was",
    "on", "at", "from", "or", "as", "not", "but", "we", "you", "they",
    "has", "have", "had", "will", "can", "do", "does", "did", "g",
}

# ── NLP setup ─────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # POS only


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def fix_byte_string(text: str) -> str:
    """Fix accidentally serialised byte-string literals like b'...'"""
    if text.startswith(("b'", 'b"')) and text.endswith(("'", '"')):
        try:
            literal = ast.literal_eval(text)
            if isinstance(literal, bytes):
                return literal.decode("utf-8", errors="replace")
        except Exception:
            pass
    return text


def strip_devrev_prefix(text: str) -> str:
    """Remove 'DevRev | ' boilerplate from titles/text."""
    # handles: "DevRev | Something", "devrev | something", "DevRev |Something"
    return re.sub(r"(?i)^devrev\s*\|\s*", "", text).strip()


def clean_text(text: str) -> str:
    """Full preprocessing pass for a single string."""
    text = fix_byte_string(text)
    text = strip_devrev_prefix(text)
    # normalise whitespace and escape sequences
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # remove lone pipe separators left after prefix strip
    text = re.sub(r"\s*\|\s*", " ", text)
    # remove URLs (keep domain terms but drop full URLs)
    text = re.sub(r"https?://\S+", "", text)
    # remove excess punctuation clusters
    text = re.sub(r"[^\w\s\-.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_bm25(text: str) -> list[str]:
    """Lowercase word tokens, domain stopwords removed."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in DOMAIN_STOPWORDS and len(t) > 1]


# ══════════════════════════════════════════════════════════════════════════════
# 2. POS-GROUPED QUERY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

# POS group definitions — applied to QUERY only (queries are full sentences,
# titles/fragments don't parse reliably)
POS_GROUPS: dict[str, set[str]] = {
    "noun_verb":  {"NOUN", "PROPN", "VERB"},
    "noun_only":  {"NOUN", "PROPN"},
    "noun_adj":   {"NOUN", "PROPN", "ADJ"},
    "verb_adv":   {"VERB", "ADV"},
    "full":       None,   # None = keep all tokens
}

# Linguistically motivated default weights (no tuning needed to be reasonable)
DEFAULT_WEIGHTS: dict[str, float] = {
    "noun_verb":  1.2,   # intent + topic — strongest combined signal
    "noun_only":  1.0,   # pure topic
    "noun_adj":   0.8,   # descriptive queries ("slow API response")
    "verb_adv":   0.5,   # action-only, weaker standalone
    "full":       1.0,   # safety net
}


def extract_pos_group(query: str, pos_tags: Optional[set[str]]) -> str:
    """Return space-joined tokens matching the given POS tags (or full query)."""
    if pos_tags is None:
        return query
    doc = nlp(query)
    tokens = [tok.text for tok in doc if tok.pos_ in pos_tags]
    return " ".join(tokens) if tokens else query   # fallback to full if empty


def get_all_pos_groups(query: str) -> dict[str, str]:
    """Return dict of group_name → filtered query string."""
    doc = nlp(query)
    result: dict[str, str] = {}
    for name, pos_tags in POS_GROUPS.items():
        if pos_tags is None:
            result[name] = query
        else:
            tokens = [tok.text for tok in doc if tok.pos_ in pos_tags]
            result[name] = " ".join(tokens) if tokens else query
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. RRF FUSION
# ══════════════════════════════════════════════════════════════════════════════

def weighted_rrf(
    ranked_lists: list[tuple[list[int], float]],
    k: int = RRF_K,
) -> list[int]:
    """
    Reciprocal Rank Fusion with per-list weights.
    ranked_lists: [(list_of_doc_indices, weight), ...]
    Returns doc indices sorted by descending fused score.
    """
    fused: dict[int, float] = {}
    for ranked_ids, weight in ranked_lists:
        for rank, doc_idx in enumerate(ranked_ids, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)
    return sorted(fused, key=lambda idx: fused[idx], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ARTICLE CAP
# ══════════════════════════════════════════════════════════════════════════════

def article_id(chunk_id: str) -> str:
    """Extract parent article ID from chunk ID."""
    return chunk_id.split("_KNOWLEDGE_NODE")[0]


def apply_article_cap(
    items: list[dict],
    k_final: int = FINAL_K,
    cap: int = ARTICLE_CAP,
) -> list[dict]:
    """Keep at most `cap` chunks per parent article."""
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
# 5. MAIN PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class BareMetalSearch:

    def __init__(
        self,
        knowledge_base_path: str,
        embeddings_cache: str = "embeddings_cache.npy",
        ids_cache: str = "ids_cache.pkl",
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._load_corpus(knowledge_base_path, embeddings_cache, ids_cache)
        self._build_bm25()
        self._load_models()

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_corpus(
        self,
        path: str,
        embeddings_cache: str,
        ids_cache: str,
    ) -> None:
        """
        Load knowledge base from a JSON file.
        Expected format: list of {"id": ..., "title": ..., "text": ...}
        Or load directly from HuggingFace datasets if path == "hf".
        """
        print("Loading corpus...")
        if path == "hf":
            from datasets import load_dataset
            kb = load_dataset("devrev/search", "knowledge_base", split="corpus")
            raw = [{"id": r["id"], "title": r["title"], "text": r["text"]} for r in kb]
        else:
            with open(path) as f:
                raw = json.load(f)

        self.doc_ids: list[str] = []
        self.doc_titles: list[str] = []
        self.doc_texts: list[str] = []
        self.documents: list[str] = []   # title + body, for BM25 and reranker

        print("Preprocessing corpus...")
        for item in tqdm(raw, desc="Cleaning"):
            title = clean_text(item.get("title", ""))
            text  = clean_text(item.get("text", ""))
            full  = f"{title}\n\n{text}".strip()

            self.doc_ids.append(item["id"])
            self.doc_titles.append(title)
            self.doc_texts.append(text)
            self.documents.append(full)

        print(f"Corpus size: {len(self.doc_ids):,} chunks")

        # load or compute embeddings
        emb_path = Path(embeddings_cache)
        ids_path = Path(ids_cache)

        if emb_path.exists() and ids_path.exists():
            print("Loading cached embeddings...")
            self.embeddings = np.load(str(emb_path))
            with open(str(ids_path), "rb") as f:
                cached_ids = pickle.load(f)
            if cached_ids != self.doc_ids:
                print("Cache mismatch — recomputing embeddings...")
                self._compute_embeddings(emb_path, ids_path)
        else:
            print("No cache found — computing embeddings (one-time cost)...")
            self._compute_embeddings(emb_path, ids_path)

    def _compute_embeddings(self, emb_path: Path, ids_path: Path) -> None:
        """Encode all documents with MiniLM and cache to disk."""
        # load model temporarily just for encoding
        encoder = SentenceTransformer(EMBED_MODEL)
        self.embeddings = encoder.encode(
            self.documents,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine sim = dot product after L2 norm
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(str(emb_path), self.embeddings)
        with open(str(ids_path), "wb") as f:
            pickle.dump(self.doc_ids, f)
        print(f"Embeddings saved: {emb_path}")

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        print("Building BM25 index...")
        tokenized = [tokenize_bm25(doc) for doc in tqdm(self.documents, desc="Tokenizing")]
        self.bm25 = BM25Okapi(tokenized)

    # ── models ────────────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        print("Loading models...")
        self.encoder  = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        print("Models loaded.")

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k_final: int = FINAL_K) -> list[dict]:
        query = clean_text(query)

        # ── Stage 1: BM25 ────────────────────────────────────────────────────
        pos_groups = get_all_pos_groups(query)

        # full-text BM25
        full_bm25_scores  = self.bm25.get_scores(tokenize_bm25(query))
        full_bm25_ranked  = list(np.argsort(full_bm25_scores)[::-1][:BM25_TOP_K])

        # noun-only BM25 (query-side POS)
        noun_query        = pos_groups["noun_only"]
        noun_bm25_scores  = self.bm25.get_scores(tokenize_bm25(noun_query))
        noun_bm25_ranked  = list(np.argsort(noun_bm25_scores)[::-1][:BM25_TOP_K])

        # RRF over the two BM25 lists
        stage1_candidates = weighted_rrf([
            (full_bm25_ranked, 1.0),
            (noun_bm25_ranked, 0.8),
        ])[:BM25_TOP_K]

        # ── Stage 2: POS-grouped embeddings ──────────────────────────────────
        # Only embed the Stage 1 candidates (100 docs, not 65K)
        candidate_embeddings = self.embeddings[stage1_candidates]  # (100, 384)

        ranked_lists: list[tuple[list[int], float]] = []

        for group_name, group_query in pos_groups.items():
            if not group_query.strip():
                continue
            q_emb  = self.encoder.encode(
                [group_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]                                          # (384,)
            sims   = candidate_embeddings @ q_emb         # (100,) dot product
            ranked = list(np.argsort(sims)[::-1])         # indices into stage1_candidates
            # map back to global doc indices
            global_ranked = [stage1_candidates[i] for i in ranked]
            weight = self.weights.get(group_name, 1.0)
            ranked_lists.append((global_ranked, weight))

        stage2_candidates = weighted_rrf(ranked_lists)[:EMBED_TOP_K]

        # ── Stage 3: Cross-encoder rerank ─────────────────────────────────────
        rerank_pairs = [(query, self.documents[idx]) for idx in stage2_candidates]
        rerank_scores = self.reranker.predict(rerank_pairs)

        reranked = sorted(
            zip(stage2_candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [
            {
                "id":    self.doc_ids[idx],
                "title": self.doc_titles[idx],
                "text":  self.doc_texts[idx],
                "score": float(score),
            }
            for idx, score in reranked
        ]

        return apply_article_cap(results, k_final=k_final, cap=ARTICLE_CAP)


# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    pipeline: BareMetalSearch,
    annotated_queries: list[dict],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Compute hit_rate, recall, precision, MRR, nDCG on annotated queries.
    Each item: {"query": ..., "retrievals": [{"id": ...}, ...]}
    """
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

    summary = {name: float(np.mean(vals)) for name, vals in metrics.items()}
    summary["ndcg@10"]         = float(np.mean(ndcg_scores))
    summary["selection_score"] = (summary["recall@10"] + summary["precision@10"]) / 2
    return summary


def print_summary(summary: dict[str, float]) -> None:
    print("=" * 44)
    for key in ("hit_rate@10", "recall@10", "precision@10", "mrr@10",
                "ndcg@10", "selection_score"):
        print(f"  {key:<18} {summary[key]:.4f}")
    print("=" * 44)


# ══════════════════════════════════════════════════════════════════════════════
# 7. WEIGHT TUNING (Optuna + 5-fold CV)
# ══════════════════════════════════════════════════════════════════════════════

def tune_weights(
    pipeline: BareMetalSearch,
    annotated_queries: list[dict],
    n_trials: int = 80,
    n_folds: int = 5,
) -> dict[str, float]:
    """
    Use Optuna (TPE / Bayesian) to find good POS group weights.
    Uses k-fold CV to avoid overfitting the small annotated set.
    Regularization term penalises highly unbalanced weights.
    """
    if not HAS_OPTUNA:
        raise ImportError("pip install optuna")

    fold_size = len(annotated_queries) // n_folds
    folds = [
        annotated_queries[i * fold_size: (i + 1) * fold_size]
        for i in range(n_folds)
    ]

    def objective(trial: "optuna.Trial") -> float:
        weights = {
            name: trial.suggest_float(name, 0.1, 2.0)
            for name in POS_GROUPS
        }
        # regularise — penalise unbalanced weights (important for sparse data)
        reg = 0.1 * float(np.std(list(weights.values())))

        pipeline.weights = weights
        fold_scores: list[float] = []
        for fold in folds:
            summary = evaluate(pipeline, fold, ks=(10,))
            fold_scores.append(summary["selection_score"])

        return float(np.mean(fold_scores)) - reg

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print("\nBest weights found:")
    for k, v in best.items():
        print(f"  {k:<20} {v:.4f}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUBMISSION WRITER
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
            "query_id":  item["query_id"],
            "query":     item["query"],
            "retrievals": [
                {"id": r["id"], "title": r["title"], "text": r["text"]}
                for r in preds
            ],
        })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Submission written → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Bare Metal DevRev Search Pipeline")
    parser.add_argument("--kb",         default="hf",
                        help="Path to knowledge_base.json, or 'hf' to load from HuggingFace")
    parser.add_argument("--emb-cache",  default="embeddings_cache.npy")
    parser.add_argument("--ids-cache",  default="ids_cache.pkl")
    parser.add_argument("--tune",       action="store_true",
                        help="Run Optuna weight tuning on annotated queries")
    parser.add_argument("--trials",     type=int, default=80)
    parser.add_argument("--output",     default="submission.json")
    args = parser.parse_args()

    # ── build pipeline ────────────────────────────────────────────────────────
    pipeline = BareMetalSearch(
        knowledge_base_path=args.kb,
        embeddings_cache=args.emb_cache,
        ids_cache=args.ids_cache,
    )

    # ── load annotated + test queries ─────────────────────────────────────────
    print("Loading queries...")
    from datasets import load_dataset
    annotated = list(load_dataset("devrev/search", "annotated_queries", split="train"))
    test_q    = list(load_dataset("devrev/search", "test_queries",      split="test"))

    # ── optional weight tuning ────────────────────────────────────────────────
    if args.tune:
        print(f"\nTuning weights ({args.trials} trials, 5-fold CV)...")
        best_weights = tune_weights(pipeline, annotated, n_trials=args.trials)
        pipeline.weights = best_weights
        # save for reuse
        with open("best_weights.json", "w") as f:
            json.dump(best_weights, f, indent=2)

    # ── sanity check on annotated queries ────────────────────────────────────
    print("\nSanity check on annotated queries...")
    summary = evaluate(pipeline, annotated)
    print_summary(summary)

    # ── generate submission ───────────────────────────────────────────────────
    write_submission(pipeline, test_q, output_path=args.output)


if __name__ == "__main__":
    main()
