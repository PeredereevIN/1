from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_downloader():
    spec = importlib.util.spec_from_file_location("download_random_arxiv", ROOT / "download_random_arxiv.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_random_quotas_are_reproducible_and_sum_to_target():
    mod = load_downloader()
    first = mod.random_quotas(1870, mod.TOPICS, 20261119, 4.0)
    second = mod.random_quotas(1870, mod.TOPICS, 20261119, 4.0)
    assert first == second
    assert sum(first.values()) == 1870
    assert len(first) == len(mod.TOPICS)
    assert len(set(first.values())) > 1


def test_parse_arxiv_feed_primary_category_and_pdf_url():
    mod = load_downloader()
    xml = b'''<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>1</opensearch:totalResults>
      <entry>
        <id>https://arxiv.org/abs/2601.12345v2</id>
        <title> A sample scientific paper </title>
        <summary>Short abstract.</summary>
        <published>2026-01-01T00:00:00Z</published>
        <arxiv:primary_category term="cs.AI" />
        <arxiv:doi>10.1234/example</arxiv:doi>
        <category term="cs.AI" />
        <link title="pdf" href="https://arxiv.org/pdf/2601.12345" />
      </entry>
    </feed>'''
    rows, total = mod.parse_feed(xml)
    assert total == 1
    assert rows[0]["arxiv_id"] == "2601.12345"
    assert rows[0]["primary_category"] == "cs.AI"
    assert rows[0]["pdf_url"].endswith("2601.12345")
    assert rows[0]["doi"] == "10.1234/example"


def test_knn_graph_separates_two_clear_vector_groups():
    from sciintel.graph_clustering import build_knn_graph
    rng = np.random.default_rng(7)
    a = np.tile(np.array([[1.0, 0.0, 0.0]]), (12, 1)) + rng.normal(0, 0.015, (12, 3))
    b = np.tile(np.array([[0.0, 1.0, 0.0]]), (12, 1)) + rng.normal(0, 0.015, (12, 3))
    x = np.vstack([a, b]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    graph, edges = build_knn_graph(x, k=4, min_similarity=0.8)
    assert graph.number_of_nodes() == 24
    assert graph.number_of_edges() > 0
    assert len(edges) == graph.number_of_edges()
    assert nx.number_connected_components(graph) == 2
    assert all(float(v) >= 0.8 for v in edges["similarity"])
