"""
SOFi Cross-Encoder Reranker — Local, no-API, single-model singleton.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22MB, runs locally)
Loaded once at MemoryManager.setup() via load_reranker().
Applied to top-50 candidates after ACT-R heat scoring, before final tiering.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_model = None  # CrossEncoder singleton — loaded lazily at startup


def load_reranker() -> None:
    """Load the cross-encoder. Blocking ~200ms first call; cached after that."""
    global _model
    if _model is not None:
        return
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder reranker loaded (ms-marco-MiniLM-L-6-v2)")
    except ImportError:
        logger.warning("sentence-transformers not installed — reranker disabled")
    except Exception as exc:
        logger.warning(f"Reranker load failed: {exc} — reranker disabled")


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """
    Rerank up to top_n candidates by cross-encoder relevance to query.
    Returns original order if the model is unavailable or an error occurs.

    Expected latency on top-50 candidates: 10–15ms (warm model).
    """
    if _model is None or not candidates:
        return candidates[:top_n]

    try:
        pairs = [(query, str(m.get("content", ""))) for m in candidates]
        scores = _model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:top_n]]
    except Exception as exc:
        logger.warning(f"Reranker predict failed: {exc} — returning original order")
        return candidates[:top_n]
