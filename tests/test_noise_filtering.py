import unittest
import pandas as pd

from cluster_labeler import AutomaticClusterLabeler, LabelerConfig


class NoiseFilteringTests(unittest.TestCase):
    def test_pdf_metadata_and_repeated_artifacts_do_not_become_labels(self):
        rows = []
        graph_titles = [
            "Dynamic Graph Algorithms for Shortest Paths",
            "Graph Algorithms for Incremental Connectivity",
            "Efficient Shortest Path Search in Large Graphs",
            "Dynamic Spanning Trees and Graph Connectivity",
            "Fast Graph Traversal with Edge Updates",
            "Lower Bounds for Graph Data Structures",
        ]
        speech_titles = [
            "Automatic Speech Recognition with Acoustic Models",
            "Robust Speech Recognition from Noisy Audio",
            "Word Error Rate Estimation for Speech Systems",
            "Streaming Automatic Speech Recognition",
            "Audio Language Models for Speech Understanding",
            "Speaker Diarization with Neural Audio Models",
        ]
        graph_body = (
            "Abstract\nWe study graph algorithms for dynamic connectivity, shortest paths, "
            "spanning trees, graph traversal, vertices, and edges. "
            "The proposed graph algorithm improves shortest path computation. "
            "ACM Classification Information Processing Systems. "
            "Repeated extraction artefact yt yt yt and log log log. "
            + ("Graph algorithm details and shortest path computation. " * 35)
            + "\nReferences\nWang Hopken Jannach. cs ds."
        )
        speech_body = (
            "Abstract\nAutomatic speech recognition uses acoustic models and audio signals. "
            "We evaluate word error rate, speaker diarization, speech transcription, "
            "and audio language models. "
            "Repeated extraction artefact yt yt yt and ri ri ri. "
            + ("Speech recognition and acoustic modeling details. " * 35)
            + "\nReferences\nWang Hopken Jannach. cs cl."
        )
        for i, title in enumerate(graph_titles):
            rows.append({
                "document_id": f"g{i}",
                "group_id": "arbitrary_group_17",
                "title": title,
                "text": (
                    "arXiv: 1234.56789\nAuthors: Wang Hopken Jannach\n"
                    "Department of Computer Science\ncs ds\n" + graph_body
                ),
            })
        for i, title in enumerate(speech_titles):
            rows.append({
                "document_id": f"s{i}",
                "group_id": "another_random_group",
                "title": title,
                "text": (
                    "arXiv: 1234.56789\nAuthors: Smith Jones\n"
                    "University of Somewhere\ncs cl\n" + speech_body
                ),
            })

        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.05, max_df=1.0)
        ).fit_predict(pd.DataFrame(rows))

        labels = result["labels"].set_index("group_id")
        candidates = result["candidates"]
        all_phrases = set(candidates["phrase"].str.lower())

        forbidden = {
            "cs ds", "cs cl", "yt yt", "yt yt yt", "log log log",
            "ri ri ri", "wang hopken jannach", "information processing systems",
            "review board irb",
        }
        self.assertFalse(forbidden.intersection(all_phrases))

        graph_label = labels.loc["arbitrary_group_17", "automatic_label"].lower()
        speech_label = labels.loc["another_random_group", "automatic_label"].lower()
        self.assertTrue(any(x in graph_label for x in ("graph", "shortest", "spanning")))
        self.assertTrue(any(x in speech_label for x in ("speech", "audio", "recognition")))

    def test_metadata_before_abstract_is_discarded(self):
        cleaned = AutomaticClusterLabeler._clean_text(
            "Authors: Wang Hopken Jannach\n"
            "Department of Computer Science\n"
            "cs ds\n"
            "Abstract\n"
            "Graph neural networks are used for graph representation learning.",
            5000,
        ).lower()
        self.assertNotIn("wang hopken", cleaned)
        self.assertNotIn("department", cleaned)
        self.assertIn("graph neural networks", cleaned)


if __name__ == "__main__":
    unittest.main()
