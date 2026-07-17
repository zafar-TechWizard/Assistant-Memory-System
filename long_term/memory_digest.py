"""
Memory Digest Layer

Pure enrichment transforms applied to raw memory dicts produced by
_package() before they enter working context.  No I/O, no DB calls.
"""

from datetime import datetime
from typing import Dict, List, Optional


# ── Reliability thresholds ────────────────────────────────────────────────────

_WELL_ESTABLISHED_MIN_ACCESS = 8
_WELL_ESTABLISHED_MIN_CONF   = 0.7
_UNCERTAIN_MAX_CONF          = 0.45
_NEW_MAX_EVIDENCE            = 1
_NEW_MAX_ACCESS              = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _age_label(m: Dict) -> str:
    raw = m.get("timestamp") or m.get("created_date") or m.get("last_updated")
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", ""))
        days = max(0, (datetime.now() - dt.replace(tzinfo=None)).days)
        if days < 2:
            return ""
        if days < 14:
            return f"{days} days ago"
        if days < 60:
            return f"{days // 7} weeks ago"
        if days < 365:
            return f"{days // 30} months ago"
        return f"{days // 365} years ago"
    except Exception:
        return ""


def _reliability_label(m: Dict) -> str:
    access   = int(m.get("access_count")   or 0)
    conf     = float(m.get("confidence")   or 0.0)
    evidence = int(m.get("evidence_count") or 1)
    if access >= _WELL_ESTABLISHED_MIN_ACCESS and conf >= _WELL_ESTABLISHED_MIN_CONF:
        return "well-established"
    if evidence <= _NEW_MAX_EVIDENCE and access <= _NEW_MAX_ACCESS:
        return "new"
    if conf < _UNCERTAIN_MAX_CONF:
        return "uncertain"
    return "confirmed"


def _memory_type_label(m: Dict) -> str:
    labels = m.get("type") or m.get("labels") or []
    if isinstance(labels, list):
        for lbl in labels:
            lbl_l = lbl.lower()
            if "relationship" in lbl_l:
                return "person"
            if "knowledge" in lbl_l:
                return "fact"
            if "experience" in lbl_l:
                return "experience"
    if m.get("person_name"):
        return "person"
    if m.get("concept"):
        return "fact"
    return "experience"


def _relationship_profile(m: Dict) -> Optional[Dict]:
    """Build a structured profile from RelationshipMemoryNode fields. Returns None if not a person memory."""
    if not m.get("person_name"):
        return None

    trust     = float(m.get("trust_level")            or 0.0)
    emotional = float(m.get("emotional_connection")   or 0.0)
    strength  = float(m.get("relationship_strength")  or 0.0)
    freq      = float(m.get("interaction_frequency")  or 0.0)

    def _trust_label(v: float) -> str:
        return "high" if v >= 0.7 else ("moderate" if v >= 0.4 else "low")

    def _emotional_label(v: float) -> str:
        return "positive" if v >= 0.35 else ("strained" if v <= -0.35 else "neutral")

    def _freq_label(v: float) -> str:
        return "frequent" if v >= 0.65 else ("occasional" if v >= 0.3 else "rare")

    return {
        "person_name":           m.get("person_name", ""),
        "relationship_type":     m.get("relationship_type", ""),
        "trust_level":           round(trust, 2),
        "trust_label":           _trust_label(trust),
        "emotional_connection":  round(emotional, 2),
        "emotional_label":       _emotional_label(emotional),
        "relationship_strength": round(strength, 2),
        "interaction_frequency": round(freq, 2),
        "frequency_label":       _freq_label(freq),
        "interaction_contexts":  list(m.get("interaction_contexts") or []),
        "personality_traits":    list(m.get("personality_traits")   or []),
    }


# ── Core enrichment ───────────────────────────────────────────────────────────

def _enrich(m: Dict, predecessor: Optional[Dict] = None) -> None:
    """Apply all digest enrichments to a single memory dict in-place."""

    age = _age_label(m)
    if age:
        m["age_label"] = age

    m["reliability_label"]  = _reliability_label(m)
    m["memory_type_label"]  = _memory_type_label(m)

    # Edge context — fields set by _spreading_activation; pop them so internal
    # scoring fields don't leak into working context as bare floats/strings.
    edge_type     = m.pop("rel_type",      None)
    edge_strength = m.pop("edge_strength", None)
    distance      = m.pop("distance",      None)
    m.pop("path_count", None)
    m.pop("activation_score", None)

    if edge_type:
        m["edge_context"] = {
            "type":     edge_type,
            "strength": round(float(edge_strength or 0.5), 2),
        }

    if distance is not None:
        m["distance_label"] = "direct" if int(distance) == 1 else "indirect"

    profile = _relationship_profile(m)
    if profile:
        m["relationship_profile"] = profile

    if predecessor and predecessor.get("content"):
        m["supersession_context"] = predecessor["content"]


# ── Public API ────────────────────────────────────────────────────────────────

def apply_digest(
    memories: List[Dict],
    supersession_map: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """
    Enrich a list of memory dicts in-place and return them.
    supersession_map: {node_id -> predecessor_dict}, supplied for must_know tier.
    """
    sm = supersession_map or {}
    for m in memories:
        node_id = str(m.get("id", ""))
        _enrich(m, predecessor=sm.get(node_id))
    return memories


def build_relationship_profiles(
    must_know: List[Dict],
    context: List[Dict],
    associations: List[Dict],
) -> Dict[str, Dict]:
    """
    Aggregate RelationshipMemoryNode profiles across all tiers.
    Must be called after apply_digest() so relationship_profile fields exist.
    """
    profiles: Dict[str, Dict] = {}
    for m in must_know + context + associations:
        profile = m.get("relationship_profile")
        if profile:
            name = profile.get("person_name", "")
            if name and name not in profiles:
                profiles[name] = profile
    return profiles
