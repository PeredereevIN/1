from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import networkx as nx

SCRIPT = Path(__file__).resolve().parents[1] / "compare_louvain_leiden.py"
spec = importlib.util.spec_from_file_location("stage1_compare", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class Stage1ComparisonTests(unittest.TestCase):
    def test_canonicalize_by_size_and_first_index(self):
        self.assertEqual(module.canonicalize_membership([9, 9, 4, 4, 4, 9, 7]), [0, 0, 1, 1, 1, 0, 2])

    def test_minimum_size_maps_small_communities_to_noise(self):
        self.assertEqual(module.apply_minimum_size([0, 0, 1, 2, 2, 2], 2), [0, 0, -1, 2, 2, 2])

    def test_connectivity_detects_disconnected_partition(self):
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b", "c", "d"])
        graph.add_edges_from([("a", "b"), ("c", "d")])
        diag = module.connectivity_diagnostics(graph, [0, 0, 0, 0])
        self.assertEqual(diag["disconnected_community_count"], 1)
        self.assertEqual(diag["extra_components_inside_communities"], 1)

    def test_connected_communities_pass(self):
        graph = nx.path_graph(["a", "b", "c", "d"])
        diag = module.connectivity_diagnostics(graph, [0, 0, 1, 1])
        self.assertEqual(diag["disconnected_community_count"], 0)

    def test_modularity_none_when_single_community(self):
        graph = nx.path_graph(["a", "b", "c"])
        nx.set_edge_attributes(graph, 1.0, "weight")
        self.assertIsNone(module.partition_modularity(graph, [0, 0, 0], 1.0))

    def test_modularity_for_two_communities(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 0.1), ("c", "d", 1.0)])
        value = module.partition_modularity(graph, [0, 0, 1, 1], 1.0)
        self.assertIsInstance(value, float)

    def test_sizes_ignore_noise(self):
        self.assertEqual(module.sizes_for_labels([-1, 0, 0, 2, 2, 2]), {"0": 2, "2": 3})


    def test_end_to_end_with_fake_igraph_api(self):
        class FakeEdges:
            def __setitem__(self, key, value):
                self.values = (key, value)

        class FakeResult:
            membership = [9, 9, 9, 4, 4, 4]

        class FakeGraph:
            def __init__(self, n, edges, directed=False):
                self.n = n
                self.edges = edges
                self.directed = directed
                self.es = FakeEdges()

            def community_leiden(self, **kwargs):
                self.kwargs = kwargs
                return FakeResult()

        fake_igraph = types.SimpleNamespace(
            __version__="test-double",
            Graph=FakeGraph,
            set_random_number_generator=lambda rng: None,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_id = "synthetic_run"
            run_dir = root / "artifacts" / "runs" / run_id
            run_dir.mkdir(parents=True)
            ids = [f"doc_{i}" for i in range(6)]
            clusters = pd.DataFrame({
                "document_id": ids,
                "cluster_id": [0, 0, 0, 1, 1, 1],
                "x": [0.0] * 6,
                "y": [0.0] * 6,
                "title": [f"Title {i}" for i in range(6)],
                "domain": ["a", "a", "a", "b", "b", "b"],
            })
            clusters.to_csv(run_dir / "clusters.csv", index=False, encoding="utf-8-sig")
            edges = pd.DataFrame({
                "source_id": [ids[0], ids[1], ids[0], ids[3], ids[4], ids[3], ids[2]],
                "target_id": [ids[1], ids[2], ids[2], ids[4], ids[5], ids[5], ids[3]],
                "similarity": [0.95, 0.94, 0.93, 0.95, 0.94, 0.93, 0.40],
            })
            edges.to_csv(run_dir / "graph_edges.csv", index=False, encoding="utf-8-sig")
            metrics = {"algorithm": "Louvain kNN graph", "graph_nodes": 6, "graph_edges": 7,
                       "clusters": 2, "resolution": 1.0, "min_community_size": 2}
            (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

            embedding_dir = root / "artifacts" / "embeddings" / "sentence-transformers_allenai-specter"
            embedding_dir.mkdir(parents=True)
            vecs = np.zeros((6, 6), dtype=np.float32)
            vecs[0:3, 0] = 1.0
            vecs[0:3, 1] = [0.01, 0.02, 0.03]
            vecs[3:6, 2] = 1.0
            vecs[3:6, 3] = [0.01, 0.02, 0.03]
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            np.save(embedding_dir / "vectors.npy", vecs)
            manifest = {"model": "test-model", "dimension": 6, "ids": ids, "normalized": True}
            (embedding_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with patch.dict(sys.modules, {"igraph": fake_igraph}):
                output = module.run_comparison(root, run_id, resolution=1.0, min_community_size=2, seed=42)

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["graph"]["nodes"], 6)
            self.assertEqual(summary["graph"]["edges"], 7)
            self.assertEqual(summary["partition_agreement"]["changed_documents"], 0)
            self.assertTrue((output / "REPORT_STAGE1_RU.md").exists())
            self.assertTrue((output / "changed_documents.csv").exists())


if __name__ == "__main__":
    unittest.main()

@unittest.skipUnless(importlib.util.find_spec("igraph") is not None, "igraph is installed by the PowerShell runner")
class RealIgraphSmokeTest(unittest.TestCase):
    def test_weighted_modularity_leiden_api(self):
        import random
        import igraph as ig

        graph = ig.Graph(
            n=8,
            edges=[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (6, 7), (2, 3), (5, 6)],
            directed=False,
        )
        graph.es["weight"] = [0.95, 0.94, 0.93, 0.95, 0.94, 0.93, 0.95, 0.40, 0.40]
        ig.set_random_number_generator(random.Random(42))
        result = graph.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=1.0,
            beta=0.01,
            n_iterations=-1,
        )
        self.assertEqual(len(result.membership), 8)
        self.assertGreaterEqual(len(set(result.membership)), 1)
