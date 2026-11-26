"""Deterministic, evidence-based bilingual naming for scientific communities.

The module does not call an LLM or an external translation service. English source
terms remain available for audit; Russian labels are produced from a curated
scientific terminology lexicon and transparent composition rules.
"""
from __future__ import annotations

import math
import re

import pandas as pd

# Canonical forms intentionally collapse spelling, number and common abbreviation
# variants. They do not infer a topic from a community ID.
ALIASES: dict[str, str] = {
    "robotic manipulation": "robot manipulation",
    "robot manipulation": "robot manipulation",
    "vision language models": "vision-language models",
    "vision-language model": "vision-language models",
    "vision-language models": "vision-language models",
    "large language model": "large language models",
    "language model": "language models",
    "retrieval-augmented generation": "retrieval augmented generation",
    "rag": "retrieval augmented generation",
    "llm agent": "llm agents",
    "ai agents": "llm agents",
    "agentic systems": "llm agents",
    "semantic id": "semantic ids",
    "recommendation systems": "recommender systems",
    "generative recommender systems": "generative recommendation",
    "integrated sensing and communications": "integrated sensing",
    "integrated sensing communication": "integrated sensing",
    "multiple input multiple output": "multiple-input multiple-output",
    "mimo": "multiple-input multiple-output",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "intelligence economics": "artificial intelligence economics",
    "intelligence adoption": "ai adoption",
    "artificial intelligence adoption": "ai adoption",
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "rl": "reinforcement learning",
    "reinforcement learning": "reinforcement learning",
    "world models": "world model",
    "world action models": "world action model",
    "world-action models": "world action model",
    "large language models": "large language models",
    "neural network": "neural networks",
    "diffusion model": "diffusion models",
    "protein representation": "protein representations",
    "molecular dynamic": "molecular dynamics",
    "generative recommender": "generative recommendation",
}

# Exact phrase translations take precedence over token-level fallback.
RU_TERMS: dict[str, str] = {
    "strongly convex": "сильно выпуклая оптимизация",
    "lower bound": "нижние оценки",
    "lower bounds": "нижние оценки",
    "optimal control": "оптимальное управление",
    "gradient descent": "градиентный спуск",
    "reinforcement learning": "обучение с подкреплением",
    "machine learning": "машинное обучение",
    "deep learning": "глубокое обучение",
    "artificial intelligence": "искусственный интеллект",
    "artificial intelligence economics": "экономика искусственного интеллекта",
    "economics of ai": "экономика искусственного интеллекта",
    "language models": "языковые модели",
    "large language models": "большие языковые модели",
    "retrieval augmented generation": "генерация с дополнением извлечённой информацией (RAG)",
    "question answering": "ответы на вопросы",
    "robot manipulation": "роботизированные манипуляции",
    "robotic manipulation": "роботизированные манипуляции",
    "world action": "модели действий и взаимодействия с миром",
    "world action model": "модели действий и взаимодействия с миром",
    "world model": "модели окружающего мира",
    "imitation learning": "обучение по демонстрациям",
    "llm agents": "интеллектуальные агенты на основе языковых моделей",
    "coding agents": "агенты для программирования",
    "frontier models": "передовые языковые модели",
    "vision-language models": "визуально-языковые модели",
    "video generation": "генерация видео",
    "visual quality": "качество визуальной генерации",
    "diffusion models": "диффузионные модели",
    "image generation": "генерация изображений",
    "visual evidence": "визуальные свидетельства",
    "neural networks": "нейронные сети",
    "foundation models": "фундаментальные модели",
    "multimodal reasoning": "мультимодальное рассуждение",
    "policy optimization": "оптимизация политик управления",
    "graph retrieval": "поиск по графам",
    "knowledge graphs": "графы знаний",
    "forecasting agents": "агенты прогнозирования",
    "autonomous systems": "автономные системы",
    "autonomous agents": "автономные агенты",
    "motion planning": "планирование движений",
    "robotics": "робототехника",
    "wireless communications": "беспроводная связь",
    "radar sensing": "радиолокационное зондирование",
    "computational biology": "вычислительная биология",
    "medical imaging": "медицинская визуализация",
    "molecular dynamics": "молекулярная динамика",
    "economic examination": "экономический анализ",
    "speculative decoding": "спекулятивное декодирование",
    "forward pass": "прямой проход нейронной сети",
    "channel estimation": "оценивание канала связи",
    "integrated sensing": "интегрированное зондирование",
    "channel model": "модель канала связи",
    "multiple-input multiple-output": "системы с множеством входов и выходов (MIMO)",
    "mean-square error": "среднеквадратичная ошибка",
    "frequency division": "частотное разделение",
    "spectral efficiency": "спектральная эффективность",
    "received signal": "принятый сигнал",
    "recommender systems": "рекомендательные системы",
    "generative recommendation": "генеративные рекомендательные методы",
    "semantic ids": "семантические идентификаторы",
    "generative retrieval": "генеративный поиск",
    "sequential recommendation": "последовательные рекомендации",
    "recommendation performance": "качество рекомендательных систем",
    "user preference": "пользовательские предпочтения",
    "user preferences": "пользовательские предпочтения",
    "item representations": "представления объектов рекомендаций",
    "computational biology": "вычислительная биология",
    "protein representations": "представления белков",
    "molecular dynamics": "молекулярная динамика",
    "proteomics": "протеомика",
    "transcriptomics": "транскриптомика",
    "medical artificial intelligence": "искусственный интеллект в медицине",
    "clinical data": "клинические данные",
    "breast cancer": "рак молочной железы",
    "algorithmic pricing": "алгоритмическое ценообразование",
    "labor market": "рынок труда",
    "labour market": "рынок труда",
    "ai adoption": "внедрение искусственного интеллекта",
    "artificial intelligence adoption": "внедрение искусственного интеллекта",
    "economic productivity": "экономическая производительность",
    "digital economy": "цифровая экономика",
    "economic impact": "экономические последствия",
    "economic effects": "экономические эффекты",
    "social welfare": "общественное благосостояние",
}

BROAD_PHRASES = {
    "machine learning", "deep learning", "artificial intelligence", "neural networks",
    "language models", "large language models", "success rate", "error rate",
    "percentage points", "united states", "lower bound", "lower bounds",
}
NON_TOPIC_PHRASES = {
    # Statistical/reporting boilerplate and geographic boilerplate are evidence
    # terms, but poor user-facing names for a scientific subject.
    "united states", "percentage points", "success rate", "success rates",
    "error rate", "error rates", "lower bound", "lower bounds",
    "state of the art", "confidence interval", "p value",
}

# Phrase combinations are semantic templates, not community-ID rules. A template
# fires only when its component concepts have independent corpus evidence.

def canonicalize(phrase: str) -> str:
    value = re.sub(r"\s+", " ", str(phrase or "").strip().casefold())
    value = value.replace("retrieval-augmented generation", "retrieval augmented generation")
    value = value.replace("vision language", "vision-language")
    return ALIASES.get(value, value)


def translate_term(phrase: str) -> str:
    canonical = canonicalize(phrase)
    if canonical in RU_TERMS:
        return RU_TERMS[canonical]
    # Do not translate unknown phrases word-by-word: Russian case, gender and
    # number agreement cannot be recovered reliably from isolated tokens.
    # Keeping an unfamiliar scientific term in English is safer than emitting
    # a fluent-looking but grammatically or semantically incorrect translation.
    return _english_display(canonical)


def _english_display(phrase: str) -> str:
    acronyms = {"llm": "LLM", "llms": "LLMs", "rag": "RAG", "vlm": "VLM", "gpu": "GPU", "mimo": "MIMO", "ai": "AI"}
    return " ".join(acronyms.get(x, x[:1].upper() + x[1:]) for x in canonicalize(phrase).split())


def _generic_penalty(phrase: str) -> float:
    p = canonicalize(phrase)
    if p in BROAD_PHRASES:
        return 0.12 if p in {"language models", "large language models", "neural networks"} else 0.28
    tokens = p.split()
    if len(tokens) == 1:
        return 0.18
    generic = {"system", "systems", "model", "models", "method", "methods", "data", "study", "research", "performance", "result", "results", "approach", "approaches", "application", "applications", "rate", "points", "states"}
    ratio = sum(token in generic for token in tokens) / max(1, len(tokens))
    return min(0.25, 0.25 * ratio)


def score_naming_candidates(candidates: pd.DataFrame, group_size: int) -> pd.DataFrame:
    """Add a naming-specific score that rewards title support and specificity.

    `score` from the phrase extractor is retained. `naming_score` is a separate
    ranking for selecting a readable label, not a clustering or quality metric.
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame() if candidates is None else candidates.copy()
    frame = candidates.copy()
    if "canonical_phrase" not in frame.columns:
        source = frame.get("phrase", pd.Series("", index=frame.index))
        frame["canonical_phrase"] = source.astype(str)
    frame["canonical_phrase"] = frame["canonical_phrase"].astype(str).map(canonicalize)
    defaults = {
        "score": 0.0, "prevalence": 0.0, "title_support_docs": 0,
        "contrast_score": 0.0, "ctfidf": 0.0, "support_docs": 0,
    }
    for col, default in defaults.items():
        if col not in frame.columns:
            frame[col] = default
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(default)
    title_prev = frame.get("title_prevalence")
    if title_prev is None:
        title_prev = frame["title_support_docs"] / max(int(group_size), 1)
    else:
        title_prev = pd.to_numeric(title_prev, errors="coerce").fillna(0.0)
    ctfidf = frame["ctfidf"]
    if ctfidf.nunique() > 1:
        ctfidf_rank = ctfidf.rank(pct=True)
    else:
        ctfidf_rank = pd.Series(0.5, index=frame.index)

    specificity, penalties = [], []
    for phrase in frame.get("canonical_phrase", frame.get("phrase", pd.Series("", index=frame.index))).astype(str):
        penalty = _generic_penalty(phrase)
        penalties.append(penalty)
        specificity.append(max(0.0, 1.0 - penalty - (0.08 if len(canonicalize(phrase).split()) == 1 else 0.0)))
    support_signal = (frame["prevalence"] / 0.20).clip(0, 1)
    title_signal = (title_prev / 0.10).clip(0, 1)
    frame["generic_penalty"] = penalties
    frame["specificity_score"] = specificity
    frame["naming_score"] = (
        0.22 * frame["score"].clip(0, 1)
        + 0.18 * support_signal
        + 0.24 * title_signal
        + 0.22 * frame["contrast_score"].clip(0, 1)
        + 0.14 * pd.Series(specificity, index=frame.index)
        - 0.18 * pd.Series(penalties, index=frame.index)
    ).clip(0, 1)
    # Canonical aliases are one concept, not separate topics. Keep the strongest
    # source row; do not add supports because aliases may overlap document-wise.
    frame = frame.sort_values(
        ["naming_score", "title_support_docs", "support_docs"], ascending=False
    ).drop_duplicates("canonical_phrase", keep="first")
    return frame.reset_index(drop=True)


def select_naming_terms(candidates: pd.DataFrame, group_size: int, max_terms: int = 3) -> list[dict]:
    ranked = score_naming_candidates(candidates, group_size)
    if ranked.empty:
        return []
    min_support = max(3, int(math.ceil(0.035 * max(group_size, 1))))
    eligible = ranked[
        (ranked["support_docs"] >= min_support)
        & ((ranked["title_support_docs"] >= 2) | (ranked["contrast_score"] >= 0.25))
    ]
    selected: list[dict] = []
    for row in eligible.to_dict("records"):
        term = canonicalize(row.get("canonical_phrase") or row.get("phrase") or "")
        if not term or term in NON_TOPIC_PHRASES:
            continue
        # Remove near-duplicates, including singular/plural variants after
        # canonicalization. Token overlap is intentionally conservative.
        token_set = set(term.split())
        duplicate = False
        for old in selected:
            old_set = set(old["canonical_phrase"].split())
            overlap = len(token_set & old_set) / max(1, len(token_set | old_set))
            if term == old["canonical_phrase"] or overlap >= 0.55:
                duplicate = True
                break
        if duplicate:
            continue
        if selected and float(row["naming_score"]) < float(selected[0]["naming_score"]) * 0.46:
            continue
        row["canonical_phrase"] = term
        selected.append(row)
        if len(selected) >= max_terms:
            break
    return selected


def _compose_label(terms: list[str]) -> str:
    term_set = set(terms)
    # More natural Russian domain titles for common, independently supported
    # combinations. The rules operate on term evidence, never on cluster IDs.
    if "recommender systems" in term_set and "generative recommendation" in term_set:
        suffix = " и семантические идентификаторы" if "semantic ids" in term_set else ""
        return "Генеративные рекомендательные системы" + suffix
    if "language models" in term_set or "large language models" in term_set:
        if term_set & {"retrieval augmented generation", "question answering"}:
            return "Большие языковые модели и интеллектуальный поиск"
        if term_set & {"speculative decoding", "neural networks", "diffusion models"}:
            return "Архитектуры языковых моделей и нейросетевые методы"
    if "robot manipulation" in term_set and term_set & {"world action", "world action model", "world model"}:
        return "Робототехника и модели взаимодействия с окружающим миром"
    if "vision-language models" in term_set:
        if term_set & {"diffusion models", "video generation", "image generation"}:
            return "Мультимодальные модели и генерация визуального контента"
        return "Визуально-языковые модели"
    if "integrated sensing" in term_set and term_set & {"channel estimation", "channel model", "multiple-input multiple-output"}:
        return "Беспроводная связь и интегрированные системы зондирования"
    if "llm agents" in term_set or "coding agents" in term_set:
        return "Интеллектуальные агенты и автономные системы"
    if "artificial intelligence economics" in term_set or term_set & {"algorithmic pricing", "labor market", "ai adoption", "digital economy"}:
        return "Экономика искусственного интеллекта и цифровая трансформация"
    if "computational biology" in term_set or term_set & {"protein representations", "molecular dynamics", "proteomics", "transcriptomics"}:
        if "medical artificial intelligence" in term_set:
            return "Вычислительная биология и медицинский искусственный интеллект"
        return "Вычислительная биология и анализ биомедицинских данных"
    if term_set & {"optimal control", "gradient descent", "strongly convex"}:
        if "reinforcement learning" in term_set:
            return "Методы оптимизации, управления и обучения с подкреплением"
        return "Методы оптимизации и оптимального управления"

    ru_terms = [translate_term(term) for term in terms]
    ru_terms = list(dict.fromkeys(x for x in ru_terms if x))
    if not ru_terms:
        return "Тематическое направление не определено"
    if len(ru_terms) == 1:
        return ru_terms[0][:1].upper() + ru_terms[0][1:]
    if len(ru_terms) == 2:
        return f"{ru_terms[0][:1].upper() + ru_terms[0][1:]} и {ru_terms[1]}"
    return f"{ru_terms[0][:1].upper() + ru_terms[0][1:]}, {ru_terms[1]} и {ru_terms[2]}"


def build_topic_name(candidates: pd.DataFrame, group_size: int, status: str = "REVIEW_PROFILE") -> dict:
    """Build English evidence terms, a Russian name, and a cautious confidence tag."""
    ranked = score_naming_candidates(candidates, group_size)
    selected = select_naming_terms(ranked, group_size)
    terms = [x["canonical_phrase"] for x in selected]
    if selected:
        label_en = " and ".join(_english_display(x) for x in terms[:2])
        label_ru = _compose_label(terms)
    else:
        label_en = "Mixed research topics — review recommended"
        label_ru = "Разнородные исследования — требуется тематическая проверка"

    confidence_map = {
        "STRONG_PROFILE": "Высокая",
        "MODERATE_PROFILE": "Умеренная",
        "REVIEW_PROFILE": "Низкая",
        "MIXED_TOPIC_REVIEW": "Не определена",
        "INTERNAL_SUBTOPICS_DETECTED": "Требуется проверка подтем",
    }
    confidence = confidence_map.get(status, "Низкая")
    evidence = []
    for row in selected:
        evidence.append({
            "term_en": row.get("canonical_phrase"),
            "term_ru": translate_term(row.get("canonical_phrase", "")),
            "support_docs": int(row.get("support_docs", 0)),
            "prevalence": round(float(row.get("prevalence", 0.0)), 4),
            "title_support_docs": int(row.get("title_support_docs", 0)),
            "contrast_score": round(float(row.get("contrast_score", 0.0)), 4),
            "naming_score": round(float(row.get("naming_score", 0.0)), 4),
        })
    aspects_en = [_english_display(x) for x in terms]
    aspects_ru = [translate_term(x) for x in terms]
    description = (
        f"Основная тема: {label_ru}. "
        + ("Подтверждающие направления: " + "; ".join(aspects_ru) + ". " if aspects_ru else "")
        + "Название сформировано по повторяющимся понятиям заголовков и аннотаций; исходные назначения графовой кластеризации не изменялись."
    )
    return {
        "automatic_label_en": label_en,
        "automatic_label_ru": label_ru,
        "automatic_description_ru": description,
        "topic_aspects_en": " | ".join(aspects_en),
        "topic_aspects_ru": " | ".join(aspects_ru),
        "naming_confidence": confidence,
        "naming_confidence_score": round(float(selected[0]["naming_score"]), 4) if selected else 0.0,
        "naming_evidence": evidence,
        "selected_topics": selected,
        "ranked_candidates": ranked,
    }


def detect_ontology_concepts(title: str, abstract: str) -> list[str]:
    """Add a few corpus-independent normalized concepts missed by contiguous n-grams.

    This is particularly useful for acronym/stop-word constructions such as
    'AI and the Economy'. It only adds evidence when a documented text pattern
    is present and never changes a document's clustering assignment.
    """
    text = f"{title or ''} {abstract or ''}".casefold()
    concepts: list[str] = []
    ai = r"(?:\bartificial intelligence\b|\bai\b|\bgenerative ai\b|\blarge language models?\b)"
    economics = r"(?:\beconom(?:y|ic|ics)\b|\blabou?r\b|\bworkforce\b|\bproductivity\b|\bpric(?:e|es|ing)\b|\bwelfare\b|\bfirms?\b|\badoption\b)"
    if re.search(ai + r"[\s\S]{0,120}" + economics, text) or re.search(economics + r"[\s\S]{0,120}" + ai, text):
        concepts.append("artificial intelligence economics")

    biology = r"\b(?:protein|proteome|proteomic|transcriptom\w*|genom\w*|molecular dynamics|sequence-based protein)\b"
    if re.search(biology, text):
        concepts.append("computational biology")
    medical = r"\b(?:medical|clinical|healthcare|patient|diagnos\w*|cancer|radiolog\w*)\b"
    ai_methods = r"\b(?:artificial intelligence|\bai\b|machine learning|deep learning|neural network|language model|multimodal)\b"
    if re.search(medical, text) and re.search(ai_methods, text):
        concepts.append("medical artificial intelligence")

    if re.search(r"\b(?:agentic|llm agents?|ai agents?|coding agents?|autonomous agents?)\b", text):
        concepts.append("llm agents")
    return list(dict.fromkeys(concepts))
