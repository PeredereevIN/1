"""Small synthetic smoke test for importable pipeline helpers."""
import numpy as np
from semantic_audit_pipeline import contiguous_ngrams, extract_research_text, compute_diagnostics


def test_contiguous_ngrams_do_not_bridge_stopwords():
    phrases = set(contiguous_ngrams("Graph-based retrieval for scientific papers"))
    assert "graph-based" in phrases
    assert "graph-based retrieval" in phrases
    assert "retrieval scientific" not in phrases


def test_extract_abstract_heading():
    text = "Title\n\nAbstract\nThis paper studies scientific graph search with experimental validation. The proposed analysis evaluates meaningful relationships between scientific publications and their semantic representations.\n\n1 Introduction\nThe rest of the paper starts here."
    abstract, source = extract_research_text(text)
    assert source == "abstract_heading"
    assert "scientific graph search" in abstract
    assert "rest of the paper" not in abstract


def test_compute_diagnostics_shapes():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(12, 8)).astype(np.float32)
    X = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    groups = np.array(["P00"] * 6 + ["P01"] * 6)
    ids = [f"d{i}" for i in range(12)]
    docs = [{"document_id": x, "title": x, "text_source": "title_only_or_missing"} for x in ids]
    result = compute_diagnostics(X, groups, ids, docs, 5)
    assert len(result[5]) == 2
    assert len(result[6]) == 12
    assert len(result[7]) == 1
    assert len(result[10]) == 60
