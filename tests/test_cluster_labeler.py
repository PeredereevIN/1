import unittest
import pandas as pd

from cluster_labeler import AutomaticClusterLabeler, LabelerConfig


class AutomaticClusterLabelerTests(unittest.TestCase):
    def test_discovers_topics_without_cluster_id_rules(self):
        rows = []
        docs = {
            "arbitrary_id_91": [
                ("Graph algorithms for dynamic shortest paths", "We study graph algorithms, shortest paths, vertices, edges, spanning trees and dynamic graph traversal."),
                ("Fast graph traversal and shortest path algorithms", "Graph traversal algorithms compute shortest paths in weighted graphs using vertices and edges."),
                ("Dynamic spanning trees in graph data structures", "We investigate dynamic graph data structures, spanning trees, graph connectivity and shortest paths."),
            ],
            "random_group_X": [
                ("Automatic speech recognition with acoustic models", "Speech recognition systems process audio signals and acoustic features for speech transcription."),
                ("Robust speech recognition from noisy audio", "Audio speech recognition uses acoustic modeling, speaker features and transcription of spoken language."),
                ("Neural text to speech synthesis and voice generation", "Speech synthesis generates voice audio with acoustic modeling and text to speech systems."),
            ],
            "topic_440": [
                ("Neural collaborative filtering for recommender systems", "Recommender systems predict user preferences from item interactions and recommendation rankings."),
                ("Recommendation systems with implicit user feedback", "Recommendation algorithms model user preferences, implicit feedback and item ranking."),
                ("Personalized recommender systems and user-item ranking", "Recommender systems use user item interactions for personalized recommendation and ranking."),
            ],
        }
        doc_id = 0
        for group, items in docs.items():
            for title, text in items:
                rows.append({
                    "document_id": f"doc_{doc_id}",
                    "group_id": group,
                    "title": title,
                    "text": text,
                    # This field must not influence the generated label.
                    "automatic_group_label": "MANUAL_LABEL_MUST_BE_IGNORED",
                })
                doc_id += 1

        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.01, max_df=1.0)
        ).fit_predict(pd.DataFrame(rows))

        labels = result["labels"].set_index("group_id")
        self.assertEqual(set(labels.index), set(docs))
        for group in docs:
            self.assertNotIn("MANUAL_LABEL", labels.loc[group, "automatic_label"])
            self.assertTrue(labels.loc[group, "key_phrases"])
            self.assertNotEqual(labels.loc[group, "automatic_label"], "Unspecified research topic")

        graph_label = labels.loc["arbitrary_id_91", "automatic_label"].lower()
        speech_label = labels.loc["random_group_X", "automatic_label"].lower()
        recommender_label = labels.loc["topic_440", "automatic_label"].lower()
        self.assertTrue(any(word in graph_label for word in ("graph", "shortest", "spanning")))
        self.assertTrue(any(word in speech_label for word in ("speech", "acoustic", "audio")))
        self.assertTrue(any(word in recommender_label for word in ("recommend", "ranking", "user")))


    def test_shared_generic_phrase_does_not_dominate_multiple_groups(self):
        rows = []
        topics = {
            "clinic_random": [
                ("Clinical Language Models for Radiology Report Summarization", "Clinical language models summarize radiology reports and extract patient findings from medical text. Language models support clinical documentation and patient record processing. " * 4),
                ("Medical Language Models for Patient Record Extraction", "Medical language models process patient records, clinical notes and radiology reports. Clinical text extraction supports hospital workflows. " * 4),
                ("Clinical Text Understanding with Large Language Models", "Large language models analyze clinical text, patient notes and radiology reports. Clinical language models support medical decision making. " * 4),
                ("Radiology Report Analysis with Clinical LLMs", "Clinical LLMs process radiology reports, medical records and patient findings. " * 4),
            ],
            "reasoning_random": [
                ("Language Models Learn Verifiable Rewards for Mathematical Reasoning", "Language models learn verifiable rewards for mathematical reasoning. Reward models evaluate correct answers and multi-step reasoning. " * 4),
                ("Verifiable Reward Models for LLM Reasoning", "Verifiable reward models train language models to solve mathematical problems and improve reasoning accuracy. " * 4),
                ("Correct Answers through Reinforcement Learning in Language Models", "Reinforcement learning teaches language models to produce correct answers with verifiable rewards. " * 4),
                ("Reward-Guided Reasoning in Large Language Models", "Large language models use reward-guided reasoning, answer verification and reinforcement learning. " * 4),
            ],
        }
        counter = 0
        for group_id, documents in topics.items():
            for title, text in documents:
                rows.append({
                    "document_id": f"shared_{counter}",
                    "group_id": group_id,
                    "title": title,
                    "text": text,
                })
                counter += 1

        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.03, max_df=1.0)
        ).fit_predict(pd.DataFrame(rows))
        labels = result["labels"].set_index("group_id")

        clinical = labels.loc["clinic_random", "automatic_label"].lower()
        reasoning = labels.loc["reasoning_random", "automatic_label"].lower()

        self.assertTrue(any(term in clinical for term in ("clinical", "radiology", "patient")))
        self.assertTrue(any(term in reasoning for term in ("correct", "reinforcement", "reward", "verifiable")))
        self.assertNotEqual(clinical, reasoning)
        self.assertFalse(clinical == "language models")
        self.assertFalse(reasoning == "language models")

    def test_noise_is_not_given_a_fake_topic(self):
        frame = pd.DataFrame([
            {"document_id": "1", "group_id": "A", "title": "Graph algorithms", "text": "Graph algorithms and graph traversal with graph edges and graph vertices."},
            {"document_id": "2", "group_id": "A", "title": "Graph traversal", "text": "Graph traversal and graph data structures with vertices and edges."},
            {"document_id": "3", "group_id": "NOISE", "title": "Unrelated paper", "text": "A completely separate subject with unrelated vocabulary."},
        ])
        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.01, max_df=1.0)
        ).fit_predict(frame)
        labels = result["labels"]
        self.assertEqual(labels["group_id"].tolist(), ["A"])
        assignments = result["assignments"].set_index("document_id")
        self.assertEqual(assignments.loc["3", "automatic_cluster_label"], "Unassigned noise")


    def test_shared_generic_phrase_is_not_selected_as_primary_topic(self):
        labeler = AutomaticClusterLabeler()
        candidates = pd.DataFrame([
            {"phrase": "language models", "length": 2, "score": 0.90,
             "group_contrast_score": 0.0, "group_contrast_margin": 0.0,
             "prevalence": 0.85, "title_support_docs": 8},
            {"phrase": "cross-lingual language models", "length": 3, "score": 0.72,
             "group_contrast_score": 0.70, "group_contrast_margin": 0.16,
             "prevalence": 0.22, "title_support_docs": 2},
            {"phrase": "multilingual tokenization", "length": 2, "score": 0.68,
             "group_contrast_score": 0.60, "group_contrast_margin": 0.13,
             "prevalence": 0.18, "title_support_docs": 2},
        ])
        selected = labeler._select_phrases(candidates)
        self.assertTrue(selected)
        self.assertNotEqual(selected[0]["phrase"], "language models")
        self.assertIn(selected[0]["phrase"], {
            "cross-lingual language models", "multilingual tokenization"
        })

    def test_rejects_malformed_extracted_ngrams(self):
        labeler = AutomaticClusterLabeler()
        self.assertFalse(labeler._phrase_is_valid("influence llm"))
        self.assertFalse(labeler._phrase_is_valid("models financial"))
        self.assertFalse(labeler._phrase_is_valid("challenges multimodal"))
        self.assertTrue(labeler._phrase_is_valid("question answering"))
        self.assertTrue(labeler._phrase_is_valid("lower bounds"))

    def test_modifier_is_attached_to_a_supported_noun_phrase(self):
        labeler = AutomaticClusterLabeler()
        label = labeler._compose_label([
            {"phrase": "financial"},
            {"phrase": "trading"},
        ])
        self.assertEqual(label, "Financial Trading")



    def test_representative_phrase_beats_rare_unique_modifier(self):
        labeler = AutomaticClusterLabeler()
        candidates = pd.DataFrame([
            {"phrase": "polynomial-time", "length": 1, "score": 0.99,
             "group_contrast_score": 1.0, "group_contrast_margin": 0.35,
             "prevalence": 0.36, "support_docs": 111, "title_support_docs": 1},
            {"phrase": "lower bounds", "length": 2, "score": 0.72,
             "group_contrast_score": 0.69, "group_contrast_margin": 0.33,
             "prevalence": 0.415, "support_docs": 127, "title_support_docs": 8},
            {"phrase": "approximation algorithms", "length": 2, "score": 0.68,
             "group_contrast_score": 1.0, "group_contrast_margin": 0.15,
             "prevalence": 0.154, "support_docs": 47, "title_support_docs": 4},
        ])
        selected = labeler._select_phrases(candidates)
        self.assertTrue(selected)
        self.assertEqual(selected[0]["phrase"], "lower bounds")
        self.assertNotIn("polynomial-time", [x["phrase"] for x in selected])

    def test_rare_unique_phrase_cannot_define_large_cluster(self):
        labeler = AutomaticClusterLabeler()
        candidates = pd.DataFrame([
            {"phrase": "noisy input", "length": 2, "score": 0.95,
             "group_contrast_score": 1.0, "group_contrast_margin": 0.09,
             "prevalence": 0.088, "support_docs": 5, "title_support_docs": 0},
            {"phrase": "physics-informed neural networks", "length": 3, "score": 0.90,
             "group_contrast_score": 0.72, "group_contrast_margin": 0.05,
             "prevalence": 0.053, "support_docs": 3, "title_support_docs": 1},
        ])
        selected = labeler._select_phrases(candidates)
        self.assertEqual(selected, [])

    def test_incidental_person_and_version_tokens_are_rejected(self):
        labeler = AutomaticClusterLabeler()
        self.assertFalse(labeler._phrase_is_valid("karpathy"))
        self.assertFalse(labeler._phrase_is_valid("internvl3"))
        self.assertFalse(labeler._phrase_is_valid("point-in-time"))
        self.assertTrue(labeler._phrase_is_valid("event sequences"))

    def test_academic_filler_words_are_removed_from_public_topics(self):
        labeler = AutomaticClusterLabeler()
        for phrase in ("set", "number", "fixed", "does", "given", "single", "directly"):
            self.assertFalse(labeler._phrase_is_valid(phrase), phrase)

    def test_public_labels_require_compound_concepts(self):
        labeler = AutomaticClusterLabeler()
        self.assertFalse(labeler._is_public_topic_phrase("brain"))
        self.assertFalse(labeler._is_public_topic_phrase("retrieval"))
        self.assertTrue(labeler._is_public_topic_phrase("semantic retrieval"))
        self.assertTrue(labeler._is_public_topic_phrase("GRPO"))

    def test_academic_compounds_are_not_destroyed_by_singleton_filter(self):
        labeler = AutomaticClusterLabeler()
        self.assertTrue(labeler._phrase_is_valid("set cover"))
        self.assertTrue(labeler._phrase_is_valid("number theory"))
        self.assertTrue(labeler._phrase_is_valid("fixed point"))
        self.assertFalse(labeler._is_public_topic_phrase("set"))

    def test_ngram_cannot_cross_clause_boundary_marker(self):
        labeler = AutomaticClusterLabeler()
        cleaned = labeler._clean_text(
            "We study tokenization, multilingual representations. "
            "The next method uses language models.",
            5000,
        )
        self.assertIn("zzboundarytokenzz", cleaned)
        self.assertFalse(
            labeler._phrase_is_valid(
                "tokenization zzboundarytokenzz multilingual"
            )
        )

    def test_contiguous_modifier_head_phrase_is_preserved_from_source_text(self):
        rows = []
        for i in range(5):
            rows.append({
                "document_id": f"multi_{i}",
                "group_id": "group_multilingual",
                "title": f"Multilingual language models for tokenization {i}",
                "text": (
                    "Multilingual language models support cross-lingual tokenization. "
                    "Language models learn multilingual representations and tokenization. " * 3
                ),
            })
        for i in range(5):
            rows.append({
                "document_id": f"clinic_{i}",
                "group_id": "group_clinical",
                "title": f"Clinical language models for medical text {i}",
                "text": (
                    "Clinical language models process medical text and clinical notes. "
                    "Language models support clinical documentation and radiology reports. " * 3
                ),
            })

        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.01, max_df=1.0)
        ).fit_predict(pd.DataFrame(rows))

        labels = result["labels"].set_index("group_id")
        candidates = result["candidates"]

        self.assertIn(
            "multilingual language models",
            set(candidates.loc[
                candidates["group_id"] == "group_multilingual", "phrase"
            ])
        )
        self.assertIn(
            "clinical language models",
            set(candidates.loc[
                candidates["group_id"] == "group_clinical", "phrase"
            ])
        )
        self.assertNotEqual(
            labels.loc["group_multilingual", "automatic_label"],
            labels.loc["group_clinical", "automatic_label"],
        )




    def test_nonadjacent_terms_are_not_synthesized_into_public_phrase(self):
        rows = []
        for i in range(6):
            rows.append({
                "document_id": f"finance_{i}",
                "group_id": "finance_group",
                "title": f"Financial risk and trading strategies in dynamic markets {i}",
                "text": (
                    "Financial risk is estimated using market information. "
                    "Trading strategies are evaluated under changing market conditions. "
                ) * 3,
            })
        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.01, max_df=1.0)
        ).fit_predict(pd.DataFrame(rows))
        candidates = result["candidates"]
        phrases = set(candidates.loc[
            candidates["group_id"] == "finance_group", "phrase"
        ].astype(str))
        self.assertNotIn("financial trading", phrases)
        self.assertFalse(LabelerConfig().synthesize_modifier_phrases)
        self.assertFalse(LabelerConfig().synthesize_concept_pairs)

    def test_generic_singletons_cannot_be_public_cluster_names(self):
        labeler = AutomaticClusterLabeler()
        for phrase in ("brain", "retrieval", "human", "audio", "tokenizer", "judge"):
            self.assertFalse(labeler._is_public_topic_phrase(phrase), phrase)

    def test_aliases_are_collapsed_to_canonical_topic_concepts(self):
        labeler = AutomaticClusterLabeler()
        self.assertEqual(labeler._canonical_topic_phrase("ASR"), "speech recognition")
        self.assertEqual(labeler._canonical_topic_phrase("VLMs"), "vision-language models")
        self.assertEqual(labeler._canonical_topic_phrase("GRPO"), "group relative policy optimization")
        self.assertEqual(labeler._phrase_overlap("ASR", "speech recognition"), 1.0)

    def test_overlapping_subtopics_are_not_both_used_in_public_title(self):
        labeler = AutomaticClusterLabeler()
        self.assertGreaterEqual(
            labeler._phrase_overlap("recommender systems", "next-poi recommendation"),
            0.5,
        )
        self.assertGreaterEqual(
            labeler._phrase_overlap("clinical language model", "large language models"),
            0.5,
        )

    def test_sentence_fragments_with_finite_verbs_are_rejected(self):
        labeler = AutomaticClusterLabeler()
        self.assertFalse(labeler._phrase_is_valid("agents perform tool"))
        self.assertFalse(labeler._phrase_is_valid("models demonstrate results"))
        self.assertTrue(labeler._phrase_is_valid("agent evaluation"))
        self.assertTrue(labeler._phrase_is_valid("financial trading"))

    def test_contiguous_noun_head_concepts_are_preserved_from_source_text(self):
        rows = []
        topics = {
            "finance": (
                "Financial trading strategies and financial market analysis",
                "Financial trading agents evaluate market effects and trading decisions. "
                "Financial models study financial trading and market risk. "
            ),
            "agents": (
                "LLM agent evaluation and tool use benchmark",
                "LLM agents perform tool use, agent evaluation and execution. "
                "Agent systems are evaluated for safety and long-horizon decisions. "
            ),
            "multilingual": (
                "Multilingual language models and tokenization",
                "Multilingual language models support cross-lingual processing, "
                "tokenization and multilingual representation. "
            ),
        }
        for group, (title, body) in topics.items():
            for i in range(6):
                rows.append({
                    "document_id": f"{group}_{i}",
                    "group_id": group,
                    "title": f"{title} {i}",
                    "text": body * 2,
                })

        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.03, max_df=1.0)
        ).fit_predict(pd.DataFrame(rows))
        candidates = result["candidates"]
        labels = result["labels"].set_index("group_id")

        finance_phrases = set(candidates.loc[candidates["group_id"] == "finance", "phrase"])
        agent_phrases = set(candidates.loc[candidates["group_id"] == "agents", "phrase"])
        multi_phrases = set(candidates.loc[candidates["group_id"] == "multilingual", "phrase"])

        self.assertIn("financial trading", finance_phrases)
        self.assertIn("agent evaluation", agent_phrases)
        self.assertIn("multilingual language models", multi_phrases)
        self.assertIn("financial", labels.loc["finance", "automatic_label"].lower())
        self.assertIn("agent", labels.loc["agents", "automatic_label"].lower())
        self.assertIn("multilingual", labels.loc["multilingual", "automatic_label"].lower())

    def test_small_cluster_evidence_is_not_near_perfect_from_uniqueness_alone(self):
        rows = []
        for i in range(7):
            rows.append({
                "document_id": f"rec_{i}",
                "group_id": "recommendations",
                "title": "Recommender systems for next POI recommendation" if i == 0 else f"Recommendation methods for user preferences {i}",
                "text": (
                    "Recommender systems model user preferences, item ranking, "
                    "next POI recommendation and multimodal representation. " * 2
                ),
            })
        result = AutomaticClusterLabeler(
            LabelerConfig(min_group_prevalence=0.03, max_df=1.0)
        ).fit_predict(pd.DataFrame(rows))
        score = float(result["labels"].iloc[0]["label_evidence_score"])
        self.assertLess(score, 0.95)



if __name__ == "__main__":
    unittest.main()
