"""
SOFi Cross-Encoder Reranker — Local, no-API, single-model singleton.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22MB, runs locally)
Loaded once at MemoryManager.setup() via load_reranker().
Applied to top-50 candidates after ACT-R heat scoring, before final tiering.
"""

from typing import List, Dict, Any

from memory.observability import observer

_model = None  # CrossEncoder singleton — loaded lazily at startup


def load_reranker() -> None:
    """
    Load the cross-encoder AND run one throwaway inference so PyTorch's lazy
    kernel compilation happens during setup, not on the user's first message.

    Without the dummy predict() below, the first real rerank pays a ~200ms
    JIT cost on top of normal inference time — verified empirically.
    """
    global _model
    if _model is not None:
        return
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Warm the kernels so the first real call lands at steady-state.
        _model.predict([("warmup", "warmup")])
        observer.info("cross-encoder reranker loaded", model="ms-marco-MiniLM-L-6-v2")
    except ImportError:
        observer.warning("sentence-transformers not installed — reranker disabled")
    except Exception as exc:
        observer.warning("reranker load failed", error=str(exc))


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """
    Rerank up to top_n candidates by cross-encoder relevance to query.
    Returns original order if the model is unavailable or an error occurs.
    """
    if _model is None or not candidates:
        return candidates[:top_n]

    try:
        pairs = [(query, str(m.get("content", ""))) for m in candidates]
        scores = _model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:top_n]]
    except Exception as exc:
        observer.warning("reranker predict failed", error=str(exc))
        return candidates[:top_n]
