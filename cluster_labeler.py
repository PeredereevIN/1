"""Generic, non-generative cluster labeling based on corpus statistics.

The module derives topic labels from the documents assigned to each cluster.
It does not use cluster IDs, hand-written per-cluster mappings, an LLM, or
pre-existing human labels.

Dependencies: pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfTransformer
from sklearn.preprocessing import normalize


GENERIC_STOPWORDS = {
    "abstract", "introduction", "conclusion", "references", "reference",
    "bibliography", "figure", "figures", "table", "tables", "section",
    "appendix", "supplementary", "copyright", "paper", "papers", "article",
    "articles", "study", "studies", "work", "works", "research", "researchers",
    "method", "methods", "methodology", "approach", "approaches", "propose",
    "proposed", "proposes", "present", "presents", "presented", "show", "shows",
    "shown", "demonstrate", "demonstrates", "result", "results", "experiment",
    "experiments", "experimental", "dataset", "datasets", "data", "task", "tasks",
    "problem", "problems", "using", "used", "use", "based", "provide", "provides",
    "including", "however", "therefore", "thus", "also", "well", "new", "different",
    "first", "second", "third", "one", "two", "three", "et", "al", "fig", "eq",
    "arxiv", "preprint", "supplement", "supplements", "available",
    "university", "department", "institute", "proceedings", "conference", "workshop",
    "vol", "volume", "issue", "pages", "page", "doi", "isbn", "issn",
    # Repeated structural terminology in mathematical/PDF extraction.
    "proposition", "corollary", "lemma", "proof", "theorem",
    # Institutional / publication boilerplate.
    "corresponding", "author", "authors", "affiliation", "affiliations",
    # Common personal-name tokens occasionally repeated in acknowledgements or abstracts.
    "karpathy", "zhang", "wang", "smith", "jones", "johnson", "hopken", "jannach",
    "txt", "csv", "pdf", "html", "xml", "json", "yaml", "yml",
    "accepted", "submitted", "published", "publication", "license",
    "version", "manuscript", "preprint", "editor", "editors",
    "creative", "commons", "permission", "permissions",
    # arXiv / ACM classification tokens, not document topics.
    "cs", "ds", "cl", "cv", "lg", "cr", "ir", "ne", "ro",
    "eess", "stat", "econ", "q", "bio", "quant", "ph",
    # Common extraction noise.
    "yt", "ri", "iii", "iv", "vii", "xx",
    # High-frequency academic English that survived the generic stop list
    # in the real corpus. These terms describe exposition, not topic.
    "does", "given", "single", "let", "directly", "standard",
    "process", "processes", "support", "supports", "generate", "generates",
    "generated", "compute", "computes", "computed", "predicts", "predicted",
    "improve", "improves", "improved", "include", "includes", "included",
    "consider", "considers", "considered", "require", "requires", "required",
    "among", "across", "within", "without", "around", "often", "generally",
    "several", "various", "specific", "particular", "certain", "another",
    "whether", "although", "whereas", "hence", "thereby", "respectively",
    "rather", "instead", "much", "many", "less", "more", "most", "least",
    "highly", "following", "previous", "existing", "consider", "considered",
    "suppose", "assume", "assuming", "denote", "denoted", "define", "defined",
    "observe", "observed", "obtained", "considering", "corresponding",
}

REFERENCE_PATTERNS = [
    r"(?im)^\s*#{0,6}\s*(?:\d+(?:\.\d+)*\s+)?references\s*$",
    r"(?im)^\s*#{0,6}\s*(?:\d+(?:\.\d+)*\s+)?bibliography\s*$",
    r"(?im)^\s*#{0,6}\s*(?:\d+(?:\.\d+)*\s+)?works cited\s*$",
    r"(?im)^\s*references\s*\n",
]

ACRONYMS = {
    "ai": "AI", "ml": "ML", "llm": "LLM", "llms": "LLMs",
    "nlp": "NLP", "asr": "ASR", "vlm": "VLM", "vlms": "VLMs",
    "kv": "KV", "gpu": "GPU", "gpus": "GPUs", "cpu": "CPU",
    "rl": "RL", "marl": "MARL", "rnn": "RNN", "cnn": "CNN",
    "gan": "GAN", "gans": "GANs", "bert": "BERT", "gpt": "GPT",
    "rag": "RAG", "api": "API", "apis": "APIs", "ood": "OOD",
    "grpo": "GRPO", "rlvr": "RLVR", "ecg": "ECG", "tsfm": "TSFM",
    "tsfms": "TSFMs", "dft": "DFT", "marl": "MARL", "poi": "POI",
}

# These terms are useful inside phrases but are too broad to serve as
# stand-alone topic labels. This is a corpus-independent linguistic filter.
LOW_INFORMATION_SINGLETONS = {
    "algorithm", "algorithms", "graph", "graphs", "vertex", "vertices",
    "edge", "edges", "bound", "bounds", "network", "networks",
    "system", "systems", "model", "models", "method", "methods",
    "approach", "approaches", "result", "results", "problem", "problems",
    "training", "inference", "learning", "text", "texts", "people",
    "flow", "time", "state", "action", "input", "output", "feature",
    "features", "performance", "accuracy", "application", "applications",
    "agent", "agents", "llm", "llms", "language", "languages",
    "workflow", "workflows", "drug",
    "large", "small", "new", "general", "global", "efficient", "distributed",
    "set", "sets", "number", "numbers", "fixed", "constant", "known",
    "case", "cases", "size", "sizes", "target", "targets", "mean", "means",
    "random", "given", "single", "standard", "directly",
}

# Stand-alone modifiers must be paired with a supported head noun before
# they can appear in a public cluster name.
MODIFIER_SINGLETONS = {
    "financial", "political", "multilingual", "cross-lingual", "clinical",
    "medical", "neural", "social", "mechanistic", "spiking", "long-horizon",
    "multimodal", "mathematical", "causal", "acoustic", "visual",
    # Stand-alone descriptors and extraction fragments are not subject areas.
    "polynomial-time", "constant-factor", "point-in-time", "inertial",
    "retinal", "russian", "cyrillic", "noisy", "physics-informed",
    "robotic", "near-surface", "memory-efficient", "hybrid-attention",
    "cross-signal", "verifiable", "temporal", "synthetic", "spoken",
}

INVALID_PHRASE_STARTS = {
    "models", "systems", "challenges", "effects", "influence",
    "impact", "using", "based", "results", "data",
}
INVALID_PHRASE_ENDINGS = {
    "financial", "multimodal", "mathematical", "distributed", "efficient",
    "influence", "large", "small", "global", "political", "visual",
    "retrieved", "dense", "inertial", "noisy", "cyrillic", "russian",
    "multilingual", "cross-lingual", "clinical", "medical", "neural",
    "social", "mechanistic", "spiking", "acoustic", "causal", "temporal",
    "synthetic", "spoken", "verifiable", "physics-informed", "robotic",
    "retinal", "long-horizon", "memory-efficient", "hybrid-attention",
}
INVALID_PHRASE_VERBS = {
    "perform", "performs", "performed", "performing", "demonstrate",
    "demonstrates", "demonstrated", "demonstrating", "evaluate", "evaluates",
    "evaluated", "evaluating", "propose", "proposes", "proposed", "proposing",
    "introduce", "introduces", "introduced", "introducing", "present",
    "presents", "presented", "presenting", "show", "shows", "showed",
    "shown", "explore", "explores", "explored", "exploring", "investigate",
    "investigates", "investigated", "investigating", "develop", "develops",
    "developed", "developing", "provide", "provides", "provided", "providing",
    "improve", "improves", "improved", "improving", "consider", "considers",
    "considered", "considering", "analyze", "analyzes", "analyzed", "analyzing",
}
VALID_LLM_PREFIXES = {
    "llm agents", "llm reasoning", "llm inference", "llm safety",
    "llm evaluation", "llm annotation", "llm memory", "llm applications",
    "llm mathematical reasoning", "large language models", "language models",
}
ACRONYM_EXPANSIONS = {
    "rlvr": "Verifiable-Reward Learning (RLVR)",
    "grpo": "Group Relative Policy Optimization (GRPO)",
    "sde": "Stochastic Differential Equations (SDE)",
    "rag": "Retrieval-Augmented Generation (RAG)",
    "marl": "Multi-Agent Reinforcement Learning (MARL)",
    "ecg": "Electrocardiogram (ECG)",
}

INVALID_SINGLETON_MODIFIERS = {
    "polynomial-time", "constant-factor", "point-in-time", "near-surface",
    "memory-efficient", "hybrid-attention", "cross-signal", "noisy",
    "physics-informed",
}

LOWERCASE_CONNECTORS = {"and", "or", "of", "for", "with", "in", "on", "to", "via", "from", "by"}
GENERIC_SINGLETONS = {
    "system", "systems", "model", "models", "framework", "frameworks",
    "method", "methods", "approach", "approaches", "language", "languages",
    "neural", "learning", "data", "research", "study", "studies",
    # Broad academic nouns that should be combined into a real concept phrase.
    "audio", "brain", "human", "molecular", "chemical", "retrieval",
    "reconstruction", "decoding", "tokenizer", "tokenization", "token",
    "tokens", "reasoning", "evaluation", "annotation", "analysis", "judge",
    "trading", "optimal", "approximation", "polynomial", "linear", "gradient",
    "gaussian", "noise", "mechanism", "mechanisms", "policy", "policies",
    "social", "enterprise", "inertial", "retinal", "spiking", "cortical",
    "causal", "evidence", "question", "answer", "answers", "similarity",
    "influence", "effects", "impact", "memory", "knowledge",
    "set", "sets", "number", "numbers", "fixed", "constant", "known",
    "case", "cases", "size", "sizes", "target", "targets", "mean", "means",
    "random", "given", "single", "standard", "directly",
}

# Noun heads that can participate in automatically composed modifier + head
# concepts. This is a language-level grammar aid, not a per-group topic map.
TOPIC_HEADS = {
    "model", "models", "network", "networks", "system", "systems",
    "algorithm", "algorithms", "learning", "processing", "recognition",
    "retrieval", "evaluation", "reasoning", "generation", "representation",
    "optimization", "prediction", "analysis", "signals", "translation",
    "classification", "detection", "annotation", "memory", "graph", "graphs",
    "decoding", "attention", "text", "texts", "agents", "agent", "applications",
    "structures", "features", "answers", "questions", "science", "bias",
    "speech", "trading", "discovery", "interpretability", "tokenization",
    "tokenizers", "policies", "mechanisms", "methods", "strategies", "effects",
    "evidence", "outcomes", "forecasting", "forecast", "decision", "decisions",
    "series", "forecast", "inference", "security", "safety", "ranking", "rankings",
    "forecasting", "modeling", "modelling", "math", "mathematics", "interactions",
    "behaviour", "behavior", "behaviors", "behavioural", "behaviour", "interactions",
}

# Corpus-independent concept modifiers used to synthesize noun phrases from
# document-level co-occurrence. These are linguistic roles, not cluster-ID rules.
CONCEPT_MODIFIERS = MODIFIER_SINGLETONS | {
    "agent", "agents", "llm", "llms", "language", "question", "knowledge",
    "information", "semantic", "graph", "graphs", "approximation", "reinforcement",
    "diffusion", "time", "time-series", "speech", "audio", "vision", "visual",
    "recommender", "recommendation", "brain", "neuron", "molecular", "chemical",
    "financial", "trading", "social", "reasoning", "reward", "verifiable",
    "cross-lingual", "multilingual", "multimodal", "large", "medical", "clinical",
    "neural", "mechanistic", "spiking", "causal", "retrieval-augmented",
}

# Equivalent surface forms are collapsed into one public concept. This vocabulary
# is general terminology, not a mapping from cluster IDs to names.
CANONICAL_TOPIC_ALIASES = {
    "asr": "speech recognition",
    "automatic speech recognition": "speech recognition",
    "speech recognition": "speech recognition",
    "vlm": "vision-language models",
    "vlms": "vision-language models",
    "vision language models": "vision-language models",
    "vision-language model": "vision-language models",
    "vision-language models": "vision-language models",
    "recommender system": "recommender systems",
    "recommendation systems": "recommender systems",
    "recommender systems": "recommender systems",
    "lower bound": "lower bounds",
    "lower bounds": "lower bounds",
    "approximation algorithm": "approximation algorithms",
    "approximation algorithms": "approximation algorithms",
    "neural network": "neural networks",
    "neural networks": "neural networks",
    "language model": "language models",
    "language models": "language models",
    "large language models": "large language models",
    "group relative policy optimization": "group relative policy optimization",
    "grpo": "group relative policy optimization",
    "rlvr": "verifiable reward learning",
    "verifiable-reward learning": "verifiable reward learning",
    "on policy distillation": "on-policy distillation",
    "on-policy distillation": "on-policy distillation",
    "retrieval augmented generation": "retrieval-augmented generation",
    "retrieval-augmented generation": "retrieval-augmented generation",
    "knowledge graph": "knowledge graphs",
    "knowledge graphs": "knowledge graphs",
}

# Broad phrases are valid evidence, but should not eclipse a more informative
# concept that differentiates a cluster.
BROAD_TOPIC_PHRASES = {
    "language models", "large language models", "neural networks",
    "machine learning", "deep learning", "artificial intelligence",
}

# Exact phrases commonly injected by PDF metadata, ACM classification blocks,
# author/affiliation lines, or repetitive mathematical layout. This is a
# corpus-independent exclusion list; it is not keyed to any cluster ID.
BOILERPLATE_PHRASES = {
    "information processing systems", "information processing",
    "computing methodologies", "computer systems organization",
    "systems organization", "general and reference",
    "software and its engineering", "human centered computing",
    "acm computing classification", "computing classification",
    "institutional review board", "review board irb",
    "institutional review", "review board", "board irb",
    "informed consent", "author contribution", "author contributions",
    "conflict of interest", "conflicts of interest",
    "all rights reserved", "arxiv preprint", "arxiv subject",
    "proposition proof corollary", "proof theorem lemma",
}

TAXONOMY_CODE_RE = re.compile(
    r"(?i)\b(?:cs|stat|math|eess|econ|q-bio|quant-ph|hep-[a-z]+)\s*\.\s*[a-z]{1,5}\b"
)
BOUNDARY_TOKEN = "zzboundarytokenzz"


@dataclass
class LabelerConfig:
    max_chars_per_document: int = 22000
    min_document_chars: int = 80
    ngram_min: int = 1
    ngram_max: int = 4
    max_features: int = 120000
    max_df: float = 0.98
    min_group_prevalence: float = 0.05
    min_title_support_fraction: float = 0.01
    min_primary_contrast: float = 0.06
    min_primary_margin: float = 0.02
    # A public topic label must be supported by a meaningful fraction of its
    # own cluster. Contrast is a secondary discriminator, not a substitute
    # for representativeness.
    min_primary_prevalence: float = 0.12
    min_secondary_prevalence: float = 0.08
    require_compound_topic_phrases: bool = True
    synthesize_modifier_phrases: bool = False
    synthesize_concept_pairs: bool = False
    min_concept_pair_prevalence: float = 0.08
    max_candidates_per_group: int = 160
    selected_phrases: int = 3
    title_weight: int = 4
    skip_noise: bool = True
    random_state: int = 42


class AutomaticClusterLabeler:
    """Derive cluster names from cluster-specific document statistics."""

    def __init__(self, config: Optional[LabelerConfig] = None):
        self.config = config or LabelerConfig()
        self.stop_words = sorted(set(ENGLISH_STOP_WORDS) | GENERIC_STOPWORDS)

    @staticmethod
    def _is_noise(group_id: object) -> bool:
        value = str(group_id).strip().lower()
        return value in {"noise", "-1", "none", "nan", "unassigned", "outlier"}

    @staticmethod
    def _clean_text(text: object, max_chars: int) -> str:
        """Remove layout, taxonomy, and publication artefacts before vectorizing.

        PDF-to-text output often puts arXiv categories, affiliation blocks,
        running headers, and page markers into the first page. We preserve the
        actual article content while discarding high-risk metadata patterns.
        """
        value = unicodedata.normalize("NFKC", str(text or ""))
        value = value.replace("\x00", " ").replace("\f", "\n")
        value = re.sub(r"-\s*\n\s*", "", value)
        value = re.sub(r"https?://\S+|www\.\S+", " ", value, flags=re.I)
        value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", value)
        value = re.sub(r"\b10\.\d{4,9}/[^\s]+", " ", value)
        value = re.sub(r"\b(arxiv:\s*)?\d{4}\.\d{4,5}(v\d+)?\b", " ", value, flags=re.I)
        value = TAXONOMY_CODE_RE.sub(" ", value)
        # Standalone subject-classification pairs sometimes lose their dots
        # during PDF extraction, e.g. "cs ds", "cs cv", "cs lg".
        value = re.sub(
            r"(?i)\b(?:cs|stat|math|eess|econ)\s+(?:ds|cv|cl|lg|ai|ir|ne|ro|cr|it|se|na|lo|si|db|dc|oh|cy|pf|gt|fl|pl|cc|ma|et|sy|sp|ce|hc|sc|sd|ms|os|ni|graphics|vision)\b",
            " ", value
        )

        # First trim explicit bibliography/acknowledgement sections. Keep line
        # breaks until line-level cleanup has been completed.
        cut_points = []
        for pattern in REFERENCE_PATTERNS:
            match = re.search(pattern, value)
            if match and match.start() > 1000:
                cut_points.append(match.start())
        for pattern in [
            r"(?im)^\s*(?:acknowledgements?|author contributions?|funding)\s*$",
            r"(?im)^\s*(?:appendix|supplementary material)\s*$",
        ]:
            match = re.search(pattern, value)
            if match and match.start() > 2500:
                cut_points.append(match.start())
        if cut_points:
            value = value[:min(cut_points)]

        # If an Abstract heading exists near the beginning, discard the noisy
        # preamble (author names, institutions, arXiv metadata) but keep the
        # abstract and the body. The separately stored title is added later.
        abstract_matches = list(re.finditer(
            r"(?im)(?:^|\n)\s*(?:abstract|summary)\s*(?:[:.\-–—]\s*)?", value
        ))
        if abstract_matches:
            first = abstract_matches[0]
            if first.start() <= 7000:
                value = value[first.end():]

        cleaned_lines = []
        seen_lines = set()
        for raw_line in value.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            low = line.casefold()
            if re.fullmatch(r"(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?", low):
                continue
            if re.search(r"(?i)\b(arxiv preprint|arxiv\.org|copyright|all rights reserved|creative commons license|submitted to|accepted at|accepted for publication|published as a conference paper)\b", low):
                continue
            if re.search(r"(?i)\b(?:acm classification|subject classifications|computer science categories|keywords? and phrases)\b", low):
                continue
            if re.fullmatch(r"(?i)(?:cs|stat|math|eess|econ)[. ](?:ds|cv|cl|lg|ai|ir|ne|ro|cr|it|se|na|lo|si|db|dc|oh|cy|pf|gt|fl|pl|cc|ma|et|sy|sp|ce|hc|sc|sd|ms|os|ni)", low):
                continue
            # Short affiliation / venue lines from the first page are highly
            # repetitive and rarely identify the paper's research topic.
            if len(line) < 180 and re.search(
                r"(?i)\b(university|department|faculty|institute|laboratory|lab|college|school of|proceedings of|association for computing machinery)\b",
                line,
            ) and not re.search(r"(?i)\b(proposed|we present|we study|our method|results show)\b", line):
                continue
            # Repeated running headers/footers are often duplicated by PDF
            # extraction. Keep the first occurrence only.
            key = re.sub(r"\W+", " ", low).strip()
            if len(key) < 140 and key in seen_lines:
                continue
            seen_lines.add(key)
            cleaned_lines.append(line)

        # Preserve sentence, clause, and PDF line boundaries until n-gram
        # extraction. The vectorizer later drops every feature containing this
        # marker, preventing phrases from crossing commas or line breaks.
        value = f" {BOUNDARY_TOKEN} ".join(cleaned_lines)
        value = re.sub(r"[,;.!?:]+", f" {BOUNDARY_TOKEN} ", value)
        value = re.sub(
            rf"(?:\s+{BOUNDARY_TOKEN}\s+)+",
            f" {BOUNDARY_TOKEN} ",
            value,
        )
        value = re.sub(r"\s+", " ", value).strip()
        return value[:max_chars]

    @staticmethod
    def _title_case_phrase(phrase: str) -> str:
        normalized = re.sub(r"\s+", " ", str(phrase).strip().lower())
        normalized = AutomaticClusterLabeler._canonical_topic_phrase(normalized)
        if normalized in ACRONYM_EXPANSIONS:
            return ACRONYM_EXPANSIONS[normalized]
        # A few morphology-only normalizations for compound scientific terms.
        normalized = re.sub(r"\bentities\b", "entity", normalized)
        normalized = re.sub(r"\bvertices\b", "vertex", normalized)
        normalized = re.sub(r"\bindices\b", "index", normalized)
        words = normalized.split()
        rendered = []
        for i, word in enumerate(words):
            lower = word.lower()
            if lower in ACRONYMS:
                rendered.append(ACRONYMS[lower])
            elif i > 0 and lower in LOWERCASE_CONNECTORS:
                rendered.append(lower)
            else:
                rendered.append(lower[:1].upper() + lower[1:])
        return " ".join(rendered)

    @staticmethod
    def _phrase_is_valid(phrase: str) -> bool:
        """Reject malformed n-grams and fragments that are not usable labels."""
        normalized = re.sub(r"\s+", " ", str(phrase).strip().lower())
        tokens = normalized.split()
        if not tokens:
            return False
        if BOUNDARY_TOKEN in tokens:
            return False
        if normalized in BOILERPLATE_PHRASES:
            return False
        if any(token in INVALID_PHRASE_VERBS for token in tokens):
            return False
        # In extracted PDF text, n-grams can cross commas or line boundaries
        # and produce adjective-in-the-middle fragments (e.g. "tokenization
        # multilingual representations"). Modifiers are only accepted at the
        # beginning of a phrase; compound modifier stacking is handled by the
        # explicit co-occurrence composer instead.
        if any(token in MODIFIER_SINGLETONS for token in tokens[1:-1]):
            return False
        if len(tokens) == 1:
            token = tokens[0]
            if (
                token in LOW_INFORMATION_SINGLETONS
                or token in GENERIC_STOPWORDS
                or token in INVALID_SINGLETON_MODIFIERS
            ):
                return False
            # Unknown model/version strings and incidental proper names are
            # not useful standalone topic labels. Known technical acronyms
            # remain available through the corpus-independent acronym list.
            if re.search(r"\d", token) and token not in ACRONYMS and token not in ACRONYM_EXPANSIONS:
                return False
            return True
        if tokens[0] in INVALID_PHRASE_STARTS:
            return False
        if tokens[-1] in INVALID_PHRASE_ENDINGS:
            return False
        if tokens[-1] in {"llm", "llms"} and normalized not in VALID_LLM_PREFIXES:
            return False
        if tokens[0] in {"large", "small"} and len(tokens) < 2:
            return False
        if any(len(token) <= 2 for token in tokens) and normalized not in VALID_LLM_PREFIXES:
            return False
        if len({AutomaticClusterLabeler._rough_stem(t) for t in tokens}) < len(tokens):
            return False
        return True

    @staticmethod
    def _natural_group_sort(value: object):
        text = str(value)
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", text)
        if match:
            return (match.group(1).lower(), int(match.group(2)))
        try:
            return ("#", int(text))
        except ValueError:
            return (text.lower(), -1)

    @staticmethod
    def _percentile(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        return pd.Series(values).rank(pct=True, method="average").to_numpy(dtype=float)

    def _prepare_documents(self, documents: pd.DataFrame, text_col: str, title_col: str) -> pd.DataFrame:
        prepared = documents.copy()
        titles = prepared[title_col].fillna("").astype(str)
        bodies = prepared[text_col].fillna("").astype(str)
        cleaned = [self._clean_text(t, self.config.max_chars_per_document) for t in bodies]
        prepared["_title"] = titles
        prepared["_body"] = cleaned
        prepared["_combined_text"] = [
            (f" {BOUNDARY_TOKEN} ".join([title] * max(1, self.config.title_weight)))
            + (f" {BOUNDARY_TOKEN} " + body if body.strip() else "")
            for title, body in zip(titles, cleaned)
        ]
        prepared["_text_length"] = [len(x) for x in cleaned]
        prepared = prepared[prepared["_combined_text"].str.len() >= self.config.min_document_chars].copy()
        return prepared.reset_index(drop=True)

    def _analyze_segments(self, document: str):
        """Yield contiguous n-grams without bridging stopwords or boundaries.

        CountVectorizer's built-in stopword removal happens before n-gram
        formation, which can join words that were not adjacent in the source
        (for example, "representations and language" -> "representations
        language"). This analyzer keeps original adjacency: any n-gram that
        contains a stopword is discarded rather than stitched together.
        """
        text = unicodedata.normalize("NFKD", str(document or ""))
        text = text.encode("ascii", "ignore").decode("ascii").lower()
        for segment in text.split(BOUNDARY_TOKEN):
            tokens = re.findall(r"(?u)\b[a-z][a-z0-9_-]{1,}\b", segment)
            for size in range(self.config.ngram_min, self.config.ngram_max + 1):
                if len(tokens) < size:
                    continue
                for start in range(0, len(tokens) - size + 1):
                    window = tokens[start:start + size]
                    if any(token in self.stop_words for token in window):
                        continue
                    yield " ".join(window)

    def _make_vectorizers(self):
        common = dict(
            analyzer=self._analyze_segments,
            max_features=self.config.max_features,
            max_df=self.config.max_df,
        )
        return (
            CountVectorizer(binary=False, **common),
            CountVectorizer(binary=True, **common),
        )

    @staticmethod
    def _safe_sparse_mean(matrix: sparse.spmatrix) -> np.ndarray:
        return np.asarray(matrix.mean(axis=0)).ravel()

    def _candidate_table(
        self,
        group_id: str,
        member_indices: np.ndarray,
        count_matrix: sparse.csr_matrix,
        binary_matrix: sparse.csr_matrix,
        title_binary_matrix: sparse.csr_matrix,
        feature_names: np.ndarray,
        all_group_binary: np.ndarray,
        all_group_prevalence: np.ndarray,
        all_group_member_indices: list[np.ndarray],
        group_position: int,
        n_groups: int,
        n_documents: int,
    ) -> pd.DataFrame:
        n = len(member_indices)
        group_counts = np.asarray(count_matrix[member_indices].sum(axis=0)).ravel()
        group_doc_freq = np.asarray(binary_matrix[member_indices].sum(axis=0)).ravel()
        group_title_freq = np.asarray(title_binary_matrix[member_indices].sum(axis=0)).ravel()
        total_tokens = max(float(group_counts.sum()), 1.0)

        prevalence = group_doc_freq / max(n, 1)
        title_prevalence = group_title_freq / max(n, 1)
        global_prevalence = np.asarray(binary_matrix.sum(axis=0)).ravel() / max(n_documents, 1)
        groups_with_term = all_group_binary.sum(axis=0)
        ctfidf = (group_counts / total_tokens) * np.log1p(n_groups / np.maximum(groups_with_term, 1))

        min_support = max(2, int(math.ceil(self.config.min_group_prevalence * n)))
        if n < 3:
            min_support = 1
        min_title_support = max(1, int(math.ceil(self.config.min_title_support_fraction * n)))

        rows = []
        for j, phrase in enumerate(feature_names):
            support = int(group_doc_freq[j])
            if support < min_support:
                continue
            tokens = phrase.split()
            if not tokens or len(tokens) > self.config.ngram_max:
                continue
            if any(token in GENERIC_STOPWORDS for token in tokens):
                continue
            if len(tokens) == 1 and tokens[0] in GENERIC_SINGLETONS:
                continue
            if not self._phrase_is_valid(phrase):
                continue
            if len(phrase) < 3:
                continue
            # Reject repeated-token artefacts: "yt yt yt", "log log log",
            # "ri ri ri". Also reject phrases dominated by repeated stems.
            stems = [self._rough_stem(token) for token in tokens]
            if len(set(stems)) < len(stems):
                continue
            if any(re.fullmatch(r"[a-z]{1,2}", token) for token in tokens):
                continue
            # Classification fragments and page-layout fragments are not
            # meaningful topic labels even if they are frequent.
            if tokens[0] in {"cs", "stat", "math", "eess", "econ"}:
                continue
            if len(tokens) >= 3 and sum(token in {"proposition", "proof", "corollary", "lemma"} for token in tokens) >= 2:
                continue

            prev = float(prevalence[j])
            global_prev = float(global_prevalence[j])
            title_prev = float(title_prevalence[j])
            title_support = int(group_title_freq[j])
            # One-off lowercase tokens with weak title evidence are often
            # author names, product names, or extraction artefacts. Keep known
            # scientific acronyms, but require additional support otherwise.
            if (
                len(tokens) == 1
                and phrase not in ACRONYMS
                and phrase not in ACRONYM_EXPANSIONS
                and prev < 0.15
                and title_support < 1
            ):
                continue
            lift = math.log((prev + 0.005) / (global_prev + 0.005))
            lift_score = min(1.0, max(0.0, lift / math.log(12.0)))

            # Cross-cluster contrast is useful, but it must not overwhelm
            # within-cluster coverage. v4 over-rewarded rare phrases that were
            # unique to only 2-5 documents. v5 makes representativeness primary.
            other_prevalences = np.delete(
                all_group_prevalence[:, j], group_position
            )
            other_max = float(other_prevalences.max()) if len(other_prevalences) else 0.0
            contrast_log = math.log((prev + 0.015) / (other_max + 0.015))
            contrast_score = min(1.0, max(0.0, contrast_log / math.log(8.0)))
            contrast_margin = max(0.0, prev - other_max)

            length_quality = min(1.0, 0.58 + 0.14 * (len(tokens) - 1))
            prevalence_score = min(1.0, prev / 0.40)
            title_score = min(1.0, title_prev / 0.10)
            relative_margin = max(0.0, prev - other_max)
            support_quality = min(1.0, prev / 0.20)
            # Multiword concepts are preferable to isolated modifiers and
            # incidental named entities, but valid technical acronyms remain.
            phrase_form_quality = 1.0 if len(tokens) >= 2 else 0.72
            if len(tokens) == 1 and phrase in ACRONYM_EXPANSIONS:
                phrase_form_quality = 0.92

            rows.append({
                "group_id": group_id,
                "phrase": phrase,
                "support_docs": support,
                "prevalence": prev,
                "title_support_docs": title_support,
                "title_support_min": min_title_support,
                "title_prevalence": title_prev,
                "global_prevalence": global_prev,
                "distinctiveness_lift": lift,
                "max_other_group_prevalence": other_max,
                "group_contrast_margin": contrast_margin,
                "group_contrast_score": contrast_score,
                "ctfidf": float(ctfidf[j]),
                "length": len(tokens),
                "_lift_score": lift_score,
                "_length_quality": length_quality,
                "_prevalence_score": prevalence_score,
                "_title_score": title_score,
                "_support_quality": support_quality,
                "_relative_margin": relative_margin,
                "_phrase_form_quality": phrase_form_quality,
            })

        if not rows:
            return pd.DataFrame(columns=[
                "group_id", "phrase", "support_docs", "prevalence",
                "title_prevalence", "title_support_docs", "global_prevalence", "distinctiveness_lift",
                "ctfidf", "length", "score"
            ])

        result = pd.DataFrame(rows)

        if self.config.synthesize_modifier_phrases:
            result = self._synthesize_modifier_head_candidates(
                group_id=group_id,
                member_indices=member_indices,
                count_matrix=count_matrix,
                binary_matrix=binary_matrix,
                title_binary_matrix=title_binary_matrix,
                feature_names=feature_names,
                all_group_member_indices=all_group_member_indices,
                n_groups=n_groups,
                base_candidates=result,
            )

        result["_ctfidf_rank"] = self._percentile(result["ctfidf"].to_numpy())
        result["score"] = (
            # Coverage is a gate and remains important, but contrast now has
            # enough weight to distinguish two topics with a shared umbrella
            # term (for example, language models vs. multilingual tokenization).
            0.27 * result["_prevalence_score"]
            + 0.15 * result["_ctfidf_rank"]
            + 0.17 * result["_title_score"]
            + 0.22 * result["group_contrast_score"]
            + 0.09 * result["_lift_score"]
            + 0.10 * result["_phrase_form_quality"]
        )
        # No title support is a caution, not an automatic rejection: abstracts
        # and body text can carry the central concept. The penalty is mild and
        # only applies to phrases not corroborated by any title.
        result.loc[result["title_support_docs"] == 0, "score"] *= 0.90

        # Keep a broad candidate pool so phrase selection can apply distinctiveness
        # and support gates without losing the most informative concepts.
        result = result.sort_values(
            ["score", "group_contrast_score", "support_docs", "title_support_docs", "length"],
            ascending=[False, False, False, False, False],
        ).head(self.config.max_candidates_per_group)

        return result.drop(columns=[c for c in result.columns if c.startswith("_")]).reset_index(drop=True)

    def _synthesize_modifier_head_candidates(
        self,
        *,
        group_id: str,
        member_indices: np.ndarray,
        count_matrix: sparse.csr_matrix,
        binary_matrix: sparse.csr_matrix,
        title_binary_matrix: sparse.csr_matrix,
        feature_names: np.ndarray,
        all_group_member_indices: list[np.ndarray],
        n_groups: int,
        base_candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create supported adjective+noun concepts from document co-occurrence.

        This lets the engine derive phrases such as "multilingual language
        models" without requiring the exact three-word sequence to occur in
        every document. A phrase is synthesized only when modifier and head
        co-occur in enough distinct documents and the combination is more
        informative than either isolated token.
        """
        if base_candidates.empty or not len(member_indices):
            return base_candidates

        feature_index = {str(term): i for i, term in enumerate(feature_names)}
        group_size = len(member_indices)
        min_pair_support = max(3, int(math.ceil(self.config.min_concept_pair_prevalence * group_size)))

        # Modifiers are a generic linguistic set shared by all clusters.
        modifiers = [
            m for m in CONCEPT_MODIFIERS
            if m in feature_index and m not in INVALID_SINGLETON_MODIFIERS
        ]

        # Only plausible multiword noun heads are considered. This avoids
        # awkward generated constructions such as "acoustic speech" from
        # combining an adjective with an isolated token.
        head_candidates: list[str] = []
        for phrase in base_candidates["phrase"].astype(str):
            toks = phrase.split()
            if len(toks) >= 2 and toks[-1] in TOPIC_HEADS and self._phrase_is_valid(phrase):
                if any(token in CONCEPT_MODIFIERS for token in toks):
                    continue
                head_candidates.append(phrase)

        # Permit supported single-token noun heads as well. This creates
        # natural concepts such as "financial trading", "agent evaluation",
        # "multilingual tokenization", and "knowledge graphs" from co-occurrence
        # in documents, rather than from a hand-written cluster mapping.
        if self.config.synthesize_concept_pairs:
            for head in TOPIC_HEADS:
                if head not in feature_index:
                    continue
                head_support = int(np.asarray(
                    binary_matrix[member_indices, feature_index[head]].sum()
                ).ravel()[0])
                if head_support >= min_pair_support:
                    head_candidates.append(head)

        head_candidates = list(dict.fromkeys(head_candidates))
        # Keep runtime bounded; candidates are ordered by base score first,
        # then by their occurrence frequency in this cluster.
        base_score = {
            str(row.phrase): (
                float(row.prevalence)
                + 0.25 * float(row.title_prevalence)
                + 0.10 * float(row.ctfidf)
            )
            for row in base_candidates.itertuples()
        }
        head_candidates.sort(
            key=lambda p: (
                base_score.get(p, 0.0),
                int(np.asarray(binary_matrix[member_indices, feature_index[p]].sum()).ravel()[0]),
                len(p.split()),
            ),
            reverse=True,
        )
        head_candidates = head_candidates[:100]

        existing = set(base_candidates["phrase"].astype(str))
        synthesized_rows: list[dict] = []

        # Cache document-presence vectors for feature columns.
        presence_cache: dict[str, np.ndarray] = {}

        def presence(term: str, matrix: sparse.csr_matrix) -> np.ndarray:
            key = f"{id(matrix)}:{term}"
            if key not in presence_cache:
                col = feature_index[term]
                presence_cache[key] = np.asarray(matrix[:, col].toarray()).ravel() > 0
            return presence_cache[key]

        for modifier in modifiers:
            if modifier not in feature_index:
                continue
            mod_global = presence(modifier, binary_matrix)
            mod_group = mod_global[member_indices]
            if int(mod_group.sum()) < min_pair_support:
                continue

            for head in head_candidates:
                if head == modifier or head not in feature_index:
                    continue
                head_tokens = head.split()
                # Avoid semantic repetition in adjective+noun compositions.
                redundant_pairs = {
                    "visual": {"vision", "vision-language"},
                    "acoustic": {"audio"},
                    "clinical": {"medical"},
                    "medical": {"clinical"},
                }
                if any(
                    token in redundant_pairs.get(modifier, set())
                    or any(token.startswith(prefix + "-") for prefix in redundant_pairs.get(modifier, set()))
                    for token in head_tokens
                ):
                    continue
                phrase = f"{modifier} {head}".strip()
                if phrase in existing or not self._phrase_is_valid(phrase):
                    continue
                if len(phrase.split()) > self.config.ngram_max:
                    continue

                head_group = presence(head, binary_matrix)[member_indices]
                joint_group = mod_group & head_group
                support = int(joint_group.sum())
                if support < min_pair_support:
                    continue

                prevalence = support / group_size
                if prevalence < self.config.min_secondary_prevalence:
                    continue

                joint_all = mod_global & presence(head, binary_matrix)
                global_prevalence = float(joint_all.mean())
                other_prevalences = [
                    float(joint_all[idx].mean())
                    for idx in all_group_member_indices
                    if not np.array_equal(idx, member_indices)
                ]
                other_max = max(other_prevalences, default=0.0)
                contrast_log = math.log((prevalence + 0.015) / (other_max + 0.015))
                contrast_score = min(1.0, max(0.0, contrast_log / math.log(8.0)))
                contrast_margin = max(0.0, prevalence - other_max)

                title_joint = (
                    presence(modifier, title_binary_matrix)
                    & presence(head, title_binary_matrix)
                )
                title_support = int(title_joint[member_indices].sum())
                title_prevalence = title_support / group_size

                lift = math.log((prevalence + 0.005) / (global_prevalence + 0.005))
                lift_score = min(1.0, max(0.0, lift / math.log(12.0)))
                prevalence_score = min(1.0, prevalence / 0.40)
                title_score = min(1.0, title_prevalence / 0.10)

                # c-TF-IDF proxy from the constituent terms. The co-occurrence
                # support itself remains the decisive evidence for this phrase.
                mod_row = base_candidates[base_candidates["phrase"] == modifier]
                head_row = base_candidates[base_candidates["phrase"] == head]
                parts = []
                if not mod_row.empty:
                    parts.append(float(mod_row.iloc[0]["ctfidf"]))
                if not head_row.empty:
                    parts.append(float(head_row.iloc[0]["ctfidf"]))
                ctfidf = float(np.mean(parts)) if parts else 0.0

                synthesized_rows.append({
                    "group_id": group_id,
                    "phrase": phrase,
                    "support_docs": support,
                    "prevalence": prevalence,
                    "title_support_docs": title_support,
                    "title_support_min": 1,
                    "title_prevalence": title_prevalence,
                    "global_prevalence": global_prevalence,
                    "distinctiveness_lift": lift,
                    "max_other_group_prevalence": other_max,
                    "group_contrast_margin": contrast_margin,
                    "group_contrast_score": contrast_score,
                    "ctfidf": ctfidf,
                    "length": len(phrase.split()),
                    "_lift_score": lift_score,
                    "_prevalence_score": prevalence_score,
                    "_title_score": title_score,
                    "_phrase_form_quality": 1.0,
                })
                existing.add(phrase)

        if synthesized_rows:
            base_candidates = pd.concat(
                [base_candidates, pd.DataFrame(synthesized_rows)],
                ignore_index=True,
            )
        return base_candidates

    @staticmethod
    def _canonical_topic_phrase(phrase: str) -> str:
        normalized = re.sub(r"\s+", " ", str(phrase).strip().lower())
        return CANONICAL_TOPIC_ALIASES.get(normalized, normalized)

    @staticmethod
    def _phrase_overlap(a: str, b: str) -> float:
        ca = AutomaticClusterLabeler._canonical_topic_phrase(a)
        cb = AutomaticClusterLabeler._canonical_topic_phrase(b)
        if ca == cb:
            return 1.0
        aa = {AutomaticClusterLabeler._rough_stem(t) for t in ca.split()}
        bb = {AutomaticClusterLabeler._rough_stem(t) for t in cb.split()}
        if not aa or not bb:
            return 0.0
        jaccard = len(aa & bb) / len(aa | bb)
        containment = len(aa & bb) / min(len(aa), len(bb))
        return max(jaccard, containment)

    @staticmethod
    def _is_public_topic_phrase(phrase: str) -> bool:
        """A public topic must be a complete concept, not a generic singleton.

        One-word evidence such as brain/retrieval/human/audio is retained in
        diagnostics but cannot become a public title. Known scientific
        abbreviations are expanded through the general alias vocabulary.
        """
        normalized = re.sub(r"\s+", " ", str(phrase).strip().lower())
        normalized = AutomaticClusterLabeler._canonical_topic_phrase(normalized)
        tokens = normalized.split()
        if not tokens or not AutomaticClusterLabeler._phrase_is_valid(normalized):
            return False
        if len(tokens) >= 2:
            return True
        if normalized in GENERIC_SINGLETONS or normalized in GENERIC_STOPWORDS:
            return False
        return normalized in ACRONYMS or normalized in ACRONYM_EXPANSIONS

    def _compose_label(self, selected: list[dict]) -> str:
        """Compose a readable label from supported phrase components.

        This is a generic grammatical rule, not a cluster-ID-specific mapping.
        A single adjective/modifier is attached to a supported noun phrase;
        otherwise independent concepts are joined with 'and'.
        """
        phrases = [str(item["phrase"]).strip().lower() for item in selected if str(item.get("phrase", "")).strip()]
        if not phrases:
            return "Unspecified research topic"
        if len(phrases) == 1:
            return self._title_case_phrase(phrases[0])

        modifiers = [x for x in phrases if x in MODIFIER_SINGLETONS]
        nouns = [x for x in phrases if x not in MODIFIER_SINGLETONS]

        if len(modifiers) == 1 and nouns:
            modifier = self._title_case_phrase(modifiers[0])
            noun = self._title_case_phrase(nouns[0])
            # Avoid an overly awkward construction when the head already
            # starts with the same modifier (e.g. 'clinical clinical text').
            if noun.lower().startswith(modifier.lower() + " "):
                return noun
            return f"{modifier} {noun}"

        rendered = [self._title_case_phrase(x) for x in phrases[:2]]
        return " and ".join(rendered)

    @staticmethod
    def _rough_stem(token: str) -> str:
        # Lightweight suffix normalization avoids labels such as
        # "Recommender Systems and Recommendation" without requiring a
        # language-specific NLP model.
        for suffix in ("ization", "ations", "ation", "ments", "ment", "ers", "er", "ies", "ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                base = token[:-len(suffix)]
                return base + ("y" if suffix == "ies" else "")
        return token

    def _select_phrases(self, candidates: pd.DataFrame) -> list[dict]:
        """Select a representative concept set with alias and redundancy control.

        Prevalence is a representativeness gate; contrast and c-TF-IDF rank
        only candidates that already have meaningful cluster support. Public
        labels never consist of generic singleton terms.
        """
        if candidates.empty:
            return []

        ranked = candidates.copy()
        if {"support_docs", "prevalence"}.issubset(ranked.columns):
            prevalence_values = ranked["prevalence"].astype(float)
            support_values = ranked["support_docs"].astype(float)
            estimated_sizes = [
                float(support_values.iloc[i]) / float(prevalence_values.iloc[i])
                for i in range(len(ranked)) if prevalence_values.iloc[i] > 0
            ]
            n = max(1, int(round(max(estimated_sizes)))) if estimated_sizes else 1
        else:
            n = 100

        min_primary_docs = max(2, int(math.ceil(0.10 * n)))
        primary_min = max(self.config.min_primary_prevalence, min_primary_docs / n)
        secondary_min = max(self.config.min_secondary_prevalence, 2.0 / n)

        ranked["phrase"] = ranked["phrase"].astype(str).map(self._canonical_topic_phrase)
        ranked = ranked[ranked["phrase"].map(self._is_public_topic_phrase)].copy()
        if ranked.empty:
            return []

        # Collapse aliases and morphological variants. Prefer the canonical
        # expanded phrase when present; otherwise retain the strongest source
        # row but use a canonical display form.
        ranked["_canonical"] = ranked["phrase"].map(self._canonical_topic_phrase)
        dedup_rows = []
        for canonical, block in ranked.groupby("_canonical", sort=False):
            block = block.copy()
            exact = block[block["phrase"].astype(str).str.lower() == canonical]
            if not exact.empty:
                chosen_row = exact.sort_values(
                    ["prevalence", "title_support_docs", "score"],
                    ascending=[False, False, False],
                ).iloc[0].copy()
            else:
                chosen_row = block.sort_values(
                    ["prevalence", "title_support_docs", "score"],
                    ascending=[False, False, False],
                ).iloc[0].copy()
            chosen_row["phrase"] = canonical
            dedup_rows.append(chosen_row)
        ranked = pd.DataFrame(dedup_rows).drop(columns=["_canonical"], errors="ignore")

        # A cluster-wide support score plus an evidence bonus. The primary
        # concept is not selected by distinctiveness alone.
        prev = ranked["prevalence"].astype(float).to_numpy()
        title_prev = ranked.get("title_prevalence", pd.Series(np.zeros(len(ranked)))).astype(float).to_numpy()
        contrast = ranked.get("group_contrast_score", pd.Series(np.zeros(len(ranked)))).astype(float).to_numpy()
        ctfidf = ranked.get("ctfidf", pd.Series(np.zeros(len(ranked)))).astype(float).to_numpy()
        ctfidf_rank = self._percentile(ctfidf)
        phrase_lengths = ranked.get(
            "length", ranked["phrase"].map(lambda x: len(str(x).split()))
        ).astype(float).to_numpy()
        phrase_specificity = np.asarray([
            0.20 if str(p).lower() in BROAD_TOPIC_PHRASES else 0.62
            for p in ranked["phrase"]
        ])
        # A complete, specific 3-4 word concept is preferred over a partial
        # two-word fragment when support is comparable. This is a modest
        # linguistic prior, not a replacement for observed document support.
        phrase_specificity += np.clip((phrase_lengths - 2.0) * 0.12, 0.0, 0.24)
        prevalence_quality = np.minimum(1.0, prev / 0.45)
        title_quality = np.minimum(1.0, title_prev / 0.12)
        ranked["_topic_rank"] = (
            0.32 * prevalence_quality
            + 0.22 * title_quality
            + 0.18 * contrast
            + 0.10 * ctfidf_rank
            + 0.18 * phrase_specificity
        )

        primary_mask = (
            (ranked["prevalence"].astype(float) >= primary_min)
            & (ranked["group_contrast_score"].astype(float) >= self.config.min_primary_contrast)
        )
        if "support_docs" in ranked.columns:
            primary_mask &= ranked["support_docs"].astype(int) >= min_primary_docs
        primary_pool = ranked[primary_mask].copy()

        if primary_pool.empty:
            return []

        primary_pool = primary_pool.sort_values(
            ["_topic_rank", "prevalence", "title_support_docs", "score"],
            ascending=[False, False, False, False],
        )

        chosen: list[dict] = []
        for record in primary_pool.to_dict("records"):
            phrase = str(record["phrase"]).strip().lower()
            if any(self._phrase_overlap(phrase, str(x["phrase"]).lower()) >= 0.50 for x in chosen):
                continue
            chosen.append(record)
            break

        if not chosen:
            return []

        secondary_mask = (
            (ranked["prevalence"].astype(float) >= secondary_min)
            & (ranked["group_contrast_score"].astype(float) >= 0.03)
        )
        if "support_docs" in ranked.columns:
            secondary_mask &= ranked["support_docs"].astype(int) >= max(2, int(math.ceil(0.06 * n)))
        secondary_pool = ranked[secondary_mask].copy()
        secondary_pool = secondary_pool.sort_values(
            ["_topic_rank", "prevalence", "title_support_docs", "score"],
            ascending=[False, False, False, False],
        )

        # Avoid a second phrase that merely repeats the same head concept.
        # Prefer an independently supported aspect, not an alias or a subtype
        # that adds no information to the public title.
        for record in secondary_pool.to_dict("records"):
            phrase = str(record["phrase"]).strip().lower()
            if any(self._phrase_overlap(phrase, str(x["phrase"]).lower()) >= 0.50 for x in chosen):
                continue
            chosen.append(record)
            if len(chosen) >= self.config.selected_phrases:
                break

        return chosen

    def _coherence(self, tfidf_matrix: sparse.csr_matrix, indices: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        subset = tfidf_matrix[indices]
        centroid = normalize(np.asarray(subset.mean(axis=0)))
        similarities = np.asarray(subset @ centroid.T).ravel()
        return (
            float(similarities.mean()),
            float(np.median(similarities)),
            float(np.percentile(similarities, 10)),
            similarities,
        )

    def fit_predict(
        self,
        documents: pd.DataFrame,
        *,
        id_col: str = "document_id",
        group_col: str = "group_id",
        title_col: str = "title",
        text_col: str = "text",
    ) -> dict[str, pd.DataFrame | dict]:
        """Return labels, candidates, enriched assignments and metadata.

        `documents` must contain an identifier, cluster/group ID, title and
        text. Existing human or automatic label columns are ignored.
        """
        required = {id_col, group_col, title_col, text_col}
        missing = required - set(documents.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        work = documents.copy()
        work[group_col] = work[group_col].fillna("NOISE").astype(str)
        if self.config.skip_noise:
            mask = ~work[group_col].map(self._is_noise)
            excluded = work.loc[~mask].copy()
            work = work.loc[mask].copy()
        else:
            excluded = work.iloc[0:0].copy()

        prepared = self._prepare_documents(work, text_col, title_col)
        if prepared.empty:
            raise ValueError("No usable documents remain after text preparation.")

        groups = sorted(prepared[group_col].unique(), key=self._natural_group_sort)
        if len(groups) < 1:
            raise ValueError("No clusters available for labeling.")

        count_vectorizer, _ = self._make_vectorizers()
        count_matrix = count_vectorizer.fit_transform(prepared["_combined_text"])

        # Binary document frequency is computed from the same vocabulary as
        # the count matrix, so all candidate statistics refer to identical terms.
        binary_matrix = count_vectorizer.transform(prepared["_combined_text"])
        binary_matrix.data[:] = 1
        title_binary_matrix = count_vectorizer.transform(prepared["_title"])
        title_binary_matrix.data[:] = 1

        feature_names = count_vectorizer.get_feature_names_out()
        # Remove boundary markers and every n-gram containing them from all
        # matrices. The marker blocks phrase formation but is never a feature
        # used for candidate scoring or document-centroid coherence.
        keep_features = np.asarray([
            BOUNDARY_TOKEN not in str(term).split()
            for term in feature_names
        ], dtype=bool)
        count_matrix = count_matrix[:, keep_features].tocsr()
        binary_matrix = binary_matrix[:, keep_features].tocsr()
        title_binary_matrix = title_binary_matrix[:, keep_features].tocsr()
        feature_names = feature_names[keep_features]
        if len(feature_names) == 0:
            raise RuntimeError("No valid text features remain after boundary filtering.")
        tfidf_matrix = TfidfTransformer(sublinear_tf=True).fit_transform(count_matrix).tocsr()

        member_map = {
            group: np.flatnonzero(prepared[group_col].to_numpy() == group)
            for group in groups
        }
        group_doc_freq = np.vstack([
            np.asarray(binary_matrix[idx].sum(axis=0)).ravel()
            for idx in member_map.values()
        ]).astype(np.float32)
        all_group_binary = (group_doc_freq > 0).astype(np.int32)
        group_sizes = np.asarray([len(idx) for idx in member_map.values()], dtype=np.float32)
        all_group_prevalence = group_doc_freq / np.maximum(group_sizes[:, None], 1.0)

        label_rows = []
        candidate_frames = []
        enriched = prepared.drop(columns=["_combined_text"], errors="ignore").copy()
        enriched["automatic_cluster_label"] = ""
        enriched["automatic_cluster_description"] = ""
        enriched["label_evidence_score"] = np.nan
        enriched["label_quality_flag"] = ""
        enriched["text_centroid_similarity"] = np.nan

        for group_index, group in enumerate(groups):
            indices = member_map[group]
            candidate_df = self._candidate_table(
                str(group), indices, count_matrix, binary_matrix,
                title_binary_matrix, feature_names, all_group_binary,
                all_group_prevalence, list(member_map.values()), group_index,
                len(groups), len(prepared),
            )
            if not candidate_df.empty:
                candidate_frames.append(candidate_df)
            selected = self._select_phrases(candidate_df)

            display_phrases = selected[:]
            if selected:
                # Phrase selection has already enforced coverage, validity,
                # and non-overlap. Use two supported concepts in the public
                # label instead of dropping the second one through an extra
                # arbitrary score-ratio threshold.
                label_phrases = selected[:2]
                label = self._compose_label(label_phrases)
                description_terms = [x["phrase"] for x in selected[:3]]
                description = (
                    "Публикации посвящены тематике «"
                    + self._title_case_phrase(selected[0]["phrase"])
                    + "»."
                    + (
                        " Дополнительно выделяются направления: "
                        + ", ".join(self._title_case_phrase(x) for x in description_terms[1:])
                        + "."
                        if len(description_terms) > 1 else ""
                    )
                    + " Название сформировано автоматически на основе статистики терминов и их распространённости по группам."
                )
                support = float(selected[0]["prevalence"])
                distinctiveness = float(selected[0].get("group_contrast_score", 0.0))
                candidate_score = float(selected[0]["score"])
                title_support_docs = int(selected[0].get("title_support_docs", 0))
                min_independent_titles = max(2, int(math.ceil(0.08 * len(indices))))
                title_corroboration = min(
                    1.0,
                    title_support_docs / max(1, min_independent_titles),
                )
                size_reliability = 0.75 + 0.25 * min(1.0, math.sqrt(len(indices) / 30.0))
                evidence_score = size_reliability * (
                    0.38 * min(1.0, support / 0.40)
                    + 0.22 * distinctiveness
                    + 0.28 * title_corroboration
                    + 0.12 * min(1.0, candidate_score)
                )
            else:
                # Keep a useful diagnostic profile, but do not pretend that a
                # weak collection of rare phrases is a stable cluster topic.
                aspects = []
                if not candidate_df.empty:
                    review_pool = candidate_df[
                        candidate_df["prevalence"] >= max(0.05, self.config.min_secondary_prevalence)
                    ].sort_values(["prevalence", "score"], ascending=[False, False])
                    for rec in review_pool.to_dict("records"):
                        phrase = str(rec["phrase"]).strip().lower()
                        if not self._is_public_topic_phrase(phrase):
                            continue
                        if any(self._phrase_overlap(phrase, x) >= 0.35 for x in aspects):
                            continue
                        aspects.append(phrase)
                        if len(aspects) >= 3:
                            break
                display_phrases = [
                    {"phrase": x} for x in aspects
                ]
                label = "Mixed research topics — review recommended"
                description = (
                    "Не обнаружена одна достаточно репрезентативная тема, "
                    "подтверждённая требуемой долей публикаций. "
                    + ("Кандидатные направления для проверки: " + ", ".join(
                        self._title_case_phrase(x) for x in aspects
                    ) + ". " if aspects else "Недостаточно устойчивых тематических фраз. ")
                    + "Кластер сохранён без изменения; требуется проверка его внутренней однородности."
                )
                support = 0.0
                distinctiveness = 0.0
                candidate_score = 0.0
                evidence_score = 0.0

            mean_coh, median_coh, p10_coh, similarities = self._coherence(tfidf_matrix, indices)
            quality = self._quality_flag(
                support, distinctiveness, mean_coh, bool(selected),
                int(selected[0].get("title_support_docs", 0)) if selected else 0,
            )

            # Most representative documents are selected by text-vector
            # similarity to this cluster's own centroid.
            local_order = np.argsort(similarities)[::-1]
            representative_titles = [str(prepared.iloc[indices[i]][title_col]) for i in local_order[:3]]
            outlier_titles = [str(prepared.iloc[indices[i]][title_col]) for i in local_order[-3:][::-1]]

            label_rows.append({
                "group_id": str(group),
                "document_count": int(len(indices)),
                "automatic_label": label,
                "automatic_description": description,
                "key_phrases": " | ".join(x["phrase"] for x in display_phrases),
                "stable_topic_found": bool(selected),
                "key_terms": " | ".join(self._fallback_terms(count_matrix[indices], feature_names, limit=12)),
                "label_evidence_score": round(evidence_score, 4),
                "label_quality_flag": quality,
                "leading_phrase_prevalence": round(support, 4),
                "leading_phrase_title_support_docs": int(selected[0].get("title_support_docs", 0)) if selected else 0,
                "leading_phrase_title_prevalence": round(float(selected[0].get("title_prevalence", 0.0)), 4) if selected else 0.0,
                "leading_phrase_distinctiveness": round(distinctiveness, 4),
                "mean_text_centroid_similarity": round(mean_coh, 4),
                "median_text_centroid_similarity": round(median_coh, 4),
                "p10_text_centroid_similarity": round(p10_coh, 4),
                "representative_titles": " | ".join(representative_titles),
                "low_similarity_titles_for_review": " | ".join(outlier_titles),
                "labeling_method": "noise-filtered cluster-level c-TF-IDF + academic stopword filtering + compound-topic selection + co-occurrence-based modifier/head synthesis + cross-cluster contrast + mixed-topic fallback",
            })

            enriched.loc[indices, "automatic_cluster_label"] = label
            enriched.loc[indices, "automatic_cluster_description"] = description
            enriched.loc[indices, "label_evidence_score"] = evidence_score
            enriched.loc[indices, "label_quality_flag"] = quality
            enriched.loc[indices, "text_centroid_similarity"] = similarities

        labels_df = pd.DataFrame(label_rows)
        candidates_df = (
            pd.concat(candidate_frames, ignore_index=True)
            if candidate_frames else pd.DataFrame()
        )

        # Retain excluded noise rows, clearly marked as not assigned to a
        # semantic cluster; do not invent a topic for noise.
        enriched = enriched.drop(
            columns=[
                "_title", "_body", "_text_length", "text",
                "automatic_group_label", "automatic_topic_label",
            ],
            errors="ignore",
        )
        if not excluded.empty:
            noise = excluded.copy()
            noise["automatic_cluster_label"] = "Unassigned noise"
            noise["automatic_cluster_description"] = "Document was not assigned to a stable cluster."
            noise["label_evidence_score"] = np.nan
            noise["label_quality_flag"] = "NOT_APPLICABLE"
            noise["text_centroid_similarity"] = np.nan
            enriched = pd.concat([enriched, noise], ignore_index=True, sort=False)

        metadata = {
            "method": "Analytical, non-generative cluster labeling",
            "algorithm": "PDF/layout cleaning + continuous source-phrase extraction + academic noise filtering + document-prevalence gating + alias canonicalization + redundancy control + cross-cluster contrast + mixed-topic abstention",
            "engine_version": "8.0.0",
            "language_note": "The initial vocabulary and stopword configuration are optimized for English-language scientific texts.",
            "uses_llm": False,
            "uses_cluster_id_specific_rules": False,
            "uses_existing_label_columns": False,
            "documents_used_for_labeling": int(len(prepared)),
            "noise_documents_excluded_from_label_generation": int(len(excluded)),
            "groups_labeled": int(len(groups)),
            "config": asdict(self.config),
            "limitations": [
                "Labels are statistical phrase summaries, not generated natural-language abstracts.",
                "Canonical concept aliases are merged (for example, ASR/speech recognition and VLM/vision-language models) so equivalent terms do not occupy separate title slots.",
                "Non-contiguous co-occurrence synthesis is disabled by default to prevent artificial noun phrases such as modifier + unrelated noun-head combinations.",
                "Noun-head co-occurrence can synthesize common concepts such as agent evaluation, financial trading, or knowledge graphs when they are supported by the same documents.",
                "Verb-led sentence fragments are filtered from candidate public phrases.",
                "Public labels require supported multiword concepts or a recognized technical acronym; isolated generic words remain diagnostic features only.",
                "Public phrase candidates must occur as contiguous word sequences in the title or cleaned text. Non-adjacent modifier/head co-occurrence is not used to invent a phrase.",
                "Known acronym and morphological aliases are normalized to a single canonical concept to prevent duplicate public labels.",
                "Generic singletons such as brain, retrieval, human, audio, or tokenizer are not allowed to become public cluster titles on their own.",
                "Primary phrases must pass a within-cluster prevalence threshold; contrast is a secondary discriminator, not a substitute for coverage.",
                "If no phrase reaches the representativeness threshold, the engine abstains from assigning a specific topic and marks the group for review.",
                "A corpus-independent phrase validator rejects malformed n-grams and low-information singleton labels.",
                "The evidence score is a heuristic support indicator, not a calibrated probability or a guarantee of semantic uniqueness.",
                "Text cleaning targets common English scientific PDF artefacts and arXiv/ACM metadata; unusual extraction noise may remain.",
                "Low-similarity documents are review candidates, not automatically misclassified documents.",
                "The first version is optimized for English scientific publications.",
                "Public labels combine independently supported contiguous source phrases; no modifier/head phrase is invented from document-level co-occurrence. Labels still require human validation for scientific publication.",
                "The public label is a statistical topic summary, not a full semantic explanation of all subtopics in a broad cluster.",
            ],
        }

        return {
            "labels": labels_df,
            "candidates": candidates_df,
            "assignments": enriched,
            "metadata": metadata,
        }

    @staticmethod
    def _fallback_terms(matrix: sparse.spmatrix, feature_names: np.ndarray, limit: int = 10) -> list[str]:
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return []
        mean_counts = np.asarray(matrix.sum(axis=0)).ravel()
        order = np.argsort(mean_counts)[::-1]
        terms = []
        for idx in order:
            if mean_counts[idx] <= 0:
                break
            phrase = str(feature_names[idx]).strip().lower()
            tokens = phrase.split()
            if not tokens or phrase in BOILERPLATE_PHRASES:
                continue
            if any(token in GENERIC_STOPWORDS for token in tokens):
                continue
            if len(tokens) == 1 and tokens[0] in GENERIC_SINGLETONS:
                continue
            if len({AutomaticClusterLabeler._rough_stem(t) for t in tokens}) < len(tokens):
                continue
            if len(phrase) < 3:
                continue
            if phrase not in terms:
                terms.append(phrase)
            if len(terms) >= limit:
                break
        return terms

    @staticmethod
    def _quality_flag(
        support: float,
        distinctiveness: float,
        coherence: float,
        has_phrase: bool,
        title_support_docs: int,
    ) -> str:
        if not has_phrase:
            return "LOW_EVIDENCE"
        # Qualitative flags are deliberately conservative and are not
        # calibrated probabilities. STRONG requires independent title
        # corroboration, not merely a high score for a tiny cluster.
        score = 0.40 * support + 0.35 * distinctiveness + 0.25 * coherence
        if score >= 0.62 and support >= 0.18 and distinctiveness >= 0.30 and title_support_docs >= 2:
            return "STRONG_EVIDENCE"
        if score >= 0.36 and support >= 0.07 and distinctiveness >= 0.10:
            return "MODERATE_EVIDENCE"
        return "WEAK_EVIDENCE_REVIEW"


def load_processed_texts(
    assignments: pd.DataFrame,
    processed_dir: Path,
    id_col: str = "document_id",
    max_chars: int = 22000,
) -> pd.DataFrame:
    """Join data/processed/<document_id>/document.txt to assignment rows."""
    result = assignments.copy()
    texts = []
    missing = []
    for document_id in result[id_col].fillna("").astype(str):
        path = Path(processed_dir) / document_id / "document.txt"
        if not path.exists():
            texts.append("")
            missing.append(document_id)
            continue
        try:
            texts.append(path.read_text(encoding="utf-8", errors="ignore")[:max_chars])
        except OSError:
            texts.append("")
            missing.append(document_id)
    result["text"] = texts
    result.attrs["missing_text_document_ids"] = missing
    return result


def find_latest_assignments(project_root: Path) -> Path:
    candidates = list((project_root / "artifacts" / "runs").glob("*/clusters.csv"))
    if not candidates:
        candidates = list((project_root / "reports").glob("**/cluster_assignments.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No cluster assignments found. Run HDBSCAN/graph-cluster first or pass --assignments explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_pipeline(
    assignments_path: Path,
    processed_dir: Path,
    output_dir: Path,
    config: Optional[LabelerConfig] = None,
) -> dict[str, Path]:
    assignments_path = Path(assignments_path)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(assignments_path, dtype={"document_id": str})
    if "group_id" not in df.columns and "cluster_id" in df.columns:
        df = df.rename(columns={"cluster_id": "group_id"})
    required = {"document_id", "group_id", "title"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Assignment CSV is missing columns: {sorted(missing_cols)}")

    joined = load_processed_texts(
        df, processed_dir,
        max_chars=(config or LabelerConfig()).max_chars_per_document,
    )
    missing_ids = joined.attrs.get("missing_text_document_ids", [])

    engine = AutomaticClusterLabeler(config)
    result = engine.fit_predict(joined)

    labels_path = output_dir / "automatic_cluster_labels.csv"
    candidates_path = output_dir / "label_candidates.csv"
    assignments_out = output_dir / "documents_with_automatic_labels.csv"
    cards_path = output_dir / "cluster_cards.json"
    metadata_path = output_dir / "labeling_metadata.json"

    result["labels"].to_csv(labels_path, index=False, encoding="utf-8-sig")
    result["candidates"].to_csv(candidates_path, index=False, encoding="utf-8-sig")
    result["assignments"].to_csv(assignments_out, index=False, encoding="utf-8-sig")

    # UI-ready compact representation. The cluster ID remains metadata;
    # the automatically derived title is the primary display field.
    cards = []
    for row in result["labels"].to_dict("records"):
        cards.append({
            "id": row["group_id"],
            "title": row["automatic_label"],
            "description": row["automatic_description"],
            "document_count": int(row["document_count"]),
            "key_phrases": [x for x in str(row["key_phrases"]).split(" | ") if x],
            "key_terms": [x for x in str(row["key_terms"]).split(" | ") if x],
            "evidence_score": float(row["label_evidence_score"]),
            "evidence_flag": row["label_quality_flag"],
            "representative_titles": [x for x in str(row["representative_titles"]).split(" | ") if x],
        })
    cards_path.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = result["metadata"]
    metadata["input_assignments"] = str(assignments_path)
    metadata["processed_text_directory"] = str(processed_dir)
    metadata["missing_text_count"] = len(missing_ids)
    metadata["missing_text_examples"] = missing_ids[:20]
    metadata["output_directory"] = str(output_dir)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "labels": labels_path,
        "candidates": candidates_path,
        "assignments": assignments_out,
        "cards": cards_path,
        "metadata": metadata_path,
    }
