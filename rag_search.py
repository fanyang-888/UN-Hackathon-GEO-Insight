"""
RAG Semantic Search for Geo-Insight Crisis Data.

Implements a lightweight vector search layer over the Gold crisis table using
TF-IDF embeddings + cosine similarity.  This mirrors the Databricks Vector Search
Index pattern shown in the Day-1 hackathon session, running locally over parquet
instead of a Delta table — the concept and API surface are identical.

Usage:
    from rag_search import CrisisRAG
    rag = CrisisRAG()          # builds index on first call, cached thereafter
    results = rag.search("find crises similar to Yemen with structural neglect", top_k=5)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GOLD_PATH = Path(__file__).parent / "data/gold/gold_ranked_crises.parquet"


def _build_document(row: pd.Series) -> str:
    """
    Convert a crisis row into a rich text document for TF-IDF indexing.

    Includes all semantically meaningful fields so natural-language queries
    like "high severity food crisis in East Africa" retrieve correct matches.
    """
    cov = row.get("coverage_pct", 0)
    pin = row.get("people_in_need", 0)
    neglect = row.get("neglect_type", "unknown")
    severity = row.get("inform_severity", 5)
    years_under = int(row.get("consecutive_years_underfunded", 0))
    has_hrp = "has HRP" if row.get("has_hrp") else "no HRP"
    low_conf = "low confidence data" if row.get("low_confidence") else "reliable data"
    gap = row.get("gap_score", 0)

    # Region inference from ISO-3 code
    iso3 = str(row.get("country_iso3", "")).upper()
    region = _iso3_to_region(iso3)

    return (
        f"{row.get('country_name', iso3)} {iso3} {row.get('year', '')} "
        f"{region} "
        f"gap score {gap:.3f} funding coverage {cov:.1f} percent "
        f"people in need {pin:,.0f} "
        f"INFORM severity {severity} "
        f"neglect type {neglect} "
        f"{'structural neglect' if neglect == 'structural' else ''} "
        f"{'acute crisis' if neglect == 'acute' else ''} "
        f"{'improving funding' if neglect == 'improving' else ''} "
        f"{years_under} consecutive years underfunded "
        f"{has_hrp} {low_conf} "
        f"{'severely underfunded' if cov < 20 else 'underfunded' if cov < 50 else 'partially funded'} "
        f"{'high severity' if severity >= 7 else 'medium severity' if severity >= 4 else 'low severity'} "
        f"{'large scale' if pin > 5_000_000 else 'medium scale' if pin > 1_000_000 else 'small scale'}"
    )


def _iso3_to_region(iso3: str) -> str:
    _map = {
        "SSD": "sub-saharan africa east africa",
        "SOM": "sub-saharan africa east africa horn of africa",
        "COD": "sub-saharan africa central africa",
        "CAF": "sub-saharan africa central africa",
        "ETH": "sub-saharan africa east africa horn of africa",
        "NER": "sub-saharan africa west africa sahel",
        "MLI": "sub-saharan africa west africa sahel",
        "TCD": "sub-saharan africa central africa sahel",
        "BFA": "sub-saharan africa west africa sahel",
        "NGA": "sub-saharan africa west africa",
        "CMR": "sub-saharan africa central africa",
        "YEM": "middle east",
        "SYR": "middle east",
        "IRQ": "middle east",
        "LBN": "middle east",
        "PSE": "middle east",
        "AFG": "south asia",
        "PAK": "south asia",
        "BGD": "south asia",
        "MMR": "south asia southeast asia",
        "HTI": "latin america caribbean",
        "VEN": "latin america",
        "COL": "latin america",
        "SDN": "north africa",
        "LBY": "north africa middle east",
        "MOZ": "sub-saharan africa southern africa",
        "ZWE": "sub-saharan africa southern africa",
        "UGA": "sub-saharan africa east africa",
        "KEN": "sub-saharan africa east africa",
        "TZA": "sub-saharan africa east africa",
        "MDG": "sub-saharan africa southern africa",
        "MWI": "sub-saharan africa southern africa",
    }
    return _map.get(iso3, "")


class CrisisRAG:
    """
    TF-IDF vector index over the Gold crisis table.

    Mirrors the Databricks Vector Search Index workflow:
      Bronze → Silver → Gold → [Vector Index] → Agent semantic_search tool

    On Databricks this would use:
      spark.sql("CREATE OR REPLACE VECTOR INDEX ...")
      or the Python SDK: VectorSearchClient().create_direct_access_index(...)

    Here we use sklearn TF-IDF + cosine similarity for local execution.
    The agent API surface (natural-language query → ranked crisis list) is identical.
    """

    def __init__(self):
        self._df: pd.DataFrame | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._docs: list[str] = []
        self._build_index()

    def _build_index(self):
        if not GOLD_PATH.exists():
            return

        df = pd.read_parquet(GOLD_PATH)
        # Use most-recent year per country for the index
        if "year" in df.columns:
            df = df.loc[df.groupby("country_iso3")["year"].idxmax()].reset_index(drop=True)

        self._df = df
        self._docs = [_build_document(row) for _, row in df.iterrows()]

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(self._docs)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search over crisis documents.

        Returns top_k crises ranked by TF-IDF cosine similarity to the query,
        merged with full crisis metadata from the Gold table.
        """
        if self._vectorizer is None or self._matrix is None or self._df is None:
            return []

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if sims[idx] < 0.01:   # skip near-zero matches
                continue
            row = self._df.iloc[idx]
            results.append({
                "country_iso3":   row.get("country_iso3"),
                "country_name":   row.get("country_name"),
                "year":           int(row.get("year", 0)),
                "gap_score":      float(row.get("gap_score", 0)),
                "coverage_pct":   float(row.get("coverage_pct", 0)),
                "people_in_need": float(row.get("people_in_need", 0)),
                "inform_severity":float(row.get("inform_severity", 0)),
                "neglect_type":   row.get("neglect_type"),
                "consecutive_years_underfunded": int(row.get("consecutive_years_underfunded", 0)),
                "has_hrp":        bool(row.get("has_hrp", False)),
                "low_confidence": bool(row.get("low_confidence", False)),
                "similarity_score": round(float(sims[idx]), 4),
            })

        return results

    def is_ready(self) -> bool:
        return self._vectorizer is not None


# Module-level singleton — built once, reused across agent calls
_rag_index: CrisisRAG | None = None


def get_rag_index() -> CrisisRAG:
    global _rag_index
    if _rag_index is None:
        _rag_index = CrisisRAG()
    return _rag_index
