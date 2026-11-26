#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic semantic audit pipeline for SPECTER document clusters.

The pipeline does not retrain embeddings, alter HDBSCAN labels, or use an LLM.
It audits existing labels, identifies boundary documents, proposes exploratory
within-cluster partitions, extracts contiguous lexical phrases, and exports
reproducible tables and visualizations.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("Не установлен numpy. Выполните: pip install -r requirements-semantic-audit.txt") from exc

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import silhouette_score
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("Не установлен scikit-learn. Выполните: pip install -r requirements-semantic-audit.txt") from exc

VERSION = "1.0.0"
DEFAULT_EMBEDDING_REL = Path("artifacts/embeddings/sentence-transformers_allenai-specter")
DEFAULT_LABEL_FILENAME = "clusters.csv"

# Stop words are removed before contiguous n-grams are generated. This avoids
# creating phrases by joining words that were not adjacent in the source text.
STOP_WORDS = set("""
a about above across after again against all am an and any are as at be because been before being below between both but by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just may me might more most must my myself no nor not of off on once only or other ought our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself yourselves
also use using used based via paper papers study studies approach approaches method methods model models result results show shown demonstrate demonstrated propose proposed proposedly framework frameworks work works research experiment experiments experimental performance task tasks problem problems data dataset datasets new novel existing different several various one two first second third respectively significant effectively efficiently application applications methodologically
""".split())

STOP_WORDS.update({
    "introduction", "conclusion", "conclusions", "abstract", "summary", "background",
    "related", "discussion", "appendix", "references", "figure", "table", "section",
    "equation", "et", "al", "ie", "eg", "etc", "fig", "tab", "arxiv", "copyright",
    "preprint", "anonymous", "author", "authors", "university", "conference", "proceedings",
    "conferencepaper", "supplementary", "material", "materials", "methodology", "results",
})

SECTION_STOP_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:"
    r"introduction|background|related\s+work|preliminaries|method(?:ology)?|"
    r"materials\s+and\s+methods|experiments?|results?|discussion|conclusions?|"
    r"references|bibliography|acknowledg(?:e)?ments?|keywords?|index\s+terms|"
    r"supplementary\s+material|appendix)\b",
    re.IGNORECASE,
)
ABSTRACT_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:abstract|summary)\s*[:.]?\s*$",
    re.IGNORECASE,
)
INTRO_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?introduction\s*[:.]?\s*$",
    re.IGNORECASE,
)
INLINE_ABSTRACT_RE = re.compile(
    r"(?is)\b(?:abstract|summary)\s*[:—–-]\s*(.+?)(?=\n\s*(?:\d+\s*\.?\s*)?(?:introduction|keywords?|index\s+terms|1\s+introduction)\b|\Z)"
)
INLINE_INTRO_RE = re.compile(
    r"(?is)\bintroduction\s*[:—–-]?\s*(.+?)(?=\n\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:related\s+work|background|method(?:ology)?|materials\s+and\s+methods|experiments?|results?)\b|\Z)"
)
WORD_RE = re.compile(r"(?u)[a-z][a-z0-9-]{2,}")


def log(message: str) -> None:
    print(message, flush=True)


def natural_group_key(value: str) -> tuple:
    if value == "NOISE":
        return (2, 0)
    m = re.match(r"^([A-Za-z]+)(\d+)$", value)
    if m:
        prefix = m.group(1)
        prefix_order = {"P": 0, "S": 1}.get(prefix, 1)
        return (prefix_order, int(m.group(2)))
    return (1, value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV без заголовка: {path}")
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (np.floating,)):
                    value = float(value)
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.bool_,)):
                    value = bool(value)
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    value = ""
                clean[key] = value
            writer.writerow(clean)


def discover_embedding_assets(root: Path, explicit_dir: str | None) -> tuple[Path, Path, dict[str, Any], np.ndarray]:
    candidates: list[Path] = []
    if explicit_dir:
        directory = (root / explicit_dir).resolve() if not Path(explicit_dir).is_absolute() else Path(explicit_dir)
        candidates = [directory]
    else:
        preferred = root / DEFAULT_EMBEDDING_REL
        if preferred.is_dir():
            candidates.append(preferred)
        for manifest in root.glob("artifacts/embeddings/**/manifest.json"):
            if manifest.parent not in candidates:
                candidates.append(manifest.parent)

    failures: list[str] = []
    for directory in candidates:
        manifest_path = directory / "manifest.json"
        vectors_path = directory / "vectors.npy"
        if not manifest_path.is_file() or not vectors_path.is_file():
            failures.append(f"{directory}: нет manifest.json и/или vectors.npy")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ids = manifest.get("ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError("manifest.ids отсутствует или пуст")
            if len(set(ids)) != len(ids):
                raise ValueError("manifest.ids содержит дубликаты")
            vectors = np.load(vectors_path, allow_pickle=False)
            expected_dim = int(manifest.get("dimension", vectors.shape[1] if vectors.ndim == 2 else -1))
            if vectors.ndim != 2 or vectors.shape != (len(ids), expected_dim):
                raise ValueError(f"несогласованные размеры: vectors={vectors.shape}, ids={len(ids)}, dim={expected_dim}")
            if not np.issubdtype(vectors.dtype, np.number):
                raise ValueError(f"недопустимый dtype: {vectors.dtype}")
            vectors = np.asarray(vectors, dtype=np.float32)
            if not np.isfinite(vectors).all():
                raise ValueError("матрица содержит NaN или бесконечные значения")
            if np.any(np.linalg.norm(vectors, axis=1) == 0):
                raise ValueError("матрица содержит нулевые векторы")
            return directory, manifest_path, manifest, vectors
        except Exception as exc:
            failures.append(f"{directory}: {exc}")
    detail = "\n".join(failures) if failures else "Не найдены каталоги embeddings с manifest.json и vectors.npy"
    raise FileNotFoundError(detail)


def discover_labels_csv(root: Path, embedding_ids: list[str], explicit_csv: str | None) -> tuple[Path, list[dict[str, str]], list[str]]:
    expected = set(embedding_ids)
    if explicit_csv:
        paths = [(root / explicit_csv).resolve() if not Path(explicit_csv).is_absolute() else Path(explicit_csv)]
    else:
        paths = sorted(root.glob(f"artifacts/runs/*/{DEFAULT_LABEL_FILENAME}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not paths:
            paths = sorted(root.glob("reports/**/cluster_assignments.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    valid: list[tuple[Path, list[dict[str, str]], list[str]]] = []
    rejected: list[str] = []
    for path in paths:
        if not path.is_file():
            rejected.append(f"{path}: файл не найден")
            continue
        try:
            rows = read_csv_rows(path)
            if rows and "group_id" not in rows[0] and "cluster_id" in rows[0]:
                for row in rows:
                    row["group_id"] = str(row.get("cluster_id", ""))
            if not rows or "document_id" not in rows[0] or "group_id" not in rows[0]:
                raise ValueError("требуются столбцы document_id и group_id (или cluster_id)")
            ids = [str(row.get("document_id", "")).strip() for row in rows]
            if any(not x for x in ids):
                raise ValueError("есть пустые document_id")
            if len(set(ids)) != len(ids):
                raise ValueError("есть повторяющиеся document_id")
            missing = sorted(expected - set(ids))
            extra = sorted(set(ids) - expected)
            if missing or extra:
                raise ValueError(f"ID не совпадают с манифестом (missing={len(missing)}, extra={len(extra)})")
            valid.append((path, rows, ids))
        except Exception as exc:
            rejected.append(f"{path}: {exc}")

    if not valid:
        details = "\n".join(rejected) if rejected else "CSV не найден"
        raise FileNotFoundError("Не найден подходящий CSV с полным совпадением document_id.\n" + details)

    # Prefer the latest valid file by modification time. The selection is printed
    # and recorded in metadata; --labels-csv can pin a specific snapshot.
    valid.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    selected = valid[0]
    notes = [f"Кандидат: {p} | mtime={dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds')} | строк={len(r)}" for p, r, _ in valid]
    notes.extend([f"Отклонён: {x}" for x in rejected])
    return selected[0], selected[1], notes


def locate_document_text(root: Path, doc_id: str) -> Path | None:
    direct = root / "data" / "processed" / doc_id / "document.txt"
    if direct.is_file():
        return direct
    # Fallback for projects where processed documents have an extra subfolder.
    matches = list((root / "data" / "processed").glob(f"**/{doc_id}/document.txt")) if (root / "data" / "processed").exists() else []
    if matches:
        return matches[0]
    return None


def _extract_lines_section(text: str, heading_re: re.Pattern[str], limit: int = 5000) -> str:
    lines = text.replace("\r", "\n").split("\n")
    start = None
    collected: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip().strip("#* ")
        if start is None:
            if heading_re.match(stripped):
                start = i + 1
            # headings with text on the same line
            elif re.match(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:abstract|summary)\s*[:—–-]\s*\S", line, re.I) and heading_re is ABSTRACT_HEADING_RE:
                collected.append(re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:abstract|summary)\s*[:—–-]\s*", "", line, flags=re.I))
                start = i
            elif re.match(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?introduction\s*[:—–-]\s*\S", line, re.I) and heading_re is INTRO_HEADING_RE:
                collected.append(re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?introduction\s*[:—–-]\s*", "", line, flags=re.I))
                start = i
        else:
            if SECTION_STOP_RE.match(stripped):
                break
            if stripped:
                collected.append(stripped)
            if sum(len(x) for x in collected) >= limit:
                break
    if not collected and start is not None:
        for line in lines[start:]:
            stripped = line.strip()
            if SECTION_STOP_RE.match(stripped):
                break
            if stripped:
                collected.append(stripped)
            if sum(len(x) for x in collected) >= limit:
                break
    return re.sub(r"\s+", " ", " ".join(collected)).strip()[:limit]


def extract_research_text(raw: str) -> tuple[str, str]:
    raw = raw.replace("\x00", " ")
    abstract = _extract_lines_section(raw, ABSTRACT_HEADING_RE)
    if len(abstract) >= 80:
        return abstract, "abstract_heading"
    inline = INLINE_ABSTRACT_RE.search(raw)
    if inline:
        value = re.sub(r"\s+", " ", inline.group(1)).strip()[:5000]
        if len(value) >= 80:
            return value, "abstract_inline"
    intro = _extract_lines_section(raw, INTRO_HEADING_RE, limit=2200)
    if len(intro) >= 100:
        return intro, "intro_fallback"
    inline_intro = INLINE_INTRO_RE.search(raw)
    if inline_intro:
        value = re.sub(r"\s+", " ", inline_intro.group(1)).strip()[:2200]
        if len(value) >= 100:
            return value, "intro_fallback"
    # Last-resort front-matter excerpt is explicitly marked; it is not called an abstract.
    front = re.sub(r"\s+", " ", raw).strip()[:1800]
    if len(front) >= 100:
        return front, "front_matter_fallback"
    return "", "title_only_or_missing"


def contiguous_ngrams(text: str) -> Iterable[str]:
    tokens = [t.lower() for t in WORD_RE.findall(text.lower())]
    for i in range(len(tokens)):
        if tokens[i] in STOP_WORDS:
            continue
        for n in (1, 2, 3):
            part = tokens[i:i+n]
            if len(part) != n or any(token in STOP_WORDS for token in part):
                continue
            # discard near-empty fragments and punctuation artifacts
            if n == 1 and len(part[0]) < 3:
                continue
            yield " ".join(part)


def build_keyword_matrix(documents: list[dict[str, Any]]):
    texts = [d["analysis_text"] for d in documents]
    vectorizer = CountVectorizer(analyzer=contiguous_ngrams, min_df=2, max_df=0.92, binary=False)
    try:
        matrix = vectorizer.fit_transform(texts).tocsr()
        terms = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
        return matrix, terms, None
    except ValueError as exc:
        return None, np.asarray([], dtype=object), str(exc)


def rank_terms_for_indices(matrix, terms: np.ndarray, indices: list[int], max_terms: int = 12) -> list[tuple[str, float, int, float]]:
    if matrix is None or not indices:
        return []
    sub = matrix[indices]
    tf = np.asarray(sub.sum(axis=0)).ravel().astype(float)
    df = np.asarray((sub > 0).sum(axis=0)).ravel().astype(int)
    n_docs = len(indices)
    # Document prevalence and normalized frequency make the ranking less dominated
    # by long documents. IDF is based on document prevalence within this corpus slice.
    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    score = (tf / max(float(tf.sum()), 1.0)) * idf
    valid = np.where(df >= min(2, n_docs))[0]
    if not len(valid):
        valid = np.arange(len(terms))
    order = valid[np.argsort(score[valid])[::-1]]
    results: list[tuple[str, float, int, float]] = []
    for j in order:
        term = str(terms[j])
        if len(term) < 3:
            continue
        results.append((term, float(score[j]), int(df[j]), float(df[j] / n_docs)))
        if len(results) >= max_terms:
            break
    return results


def save_group_keywords(documents: list[dict[str, Any]], groups: np.ndarray, matrix, terms: np.ndarray, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_groups = sorted(set(groups.tolist()), key=natural_group_key)
    if matrix is None or len(terms) == 0:
        write_csv(out_dir / "group_keywords.csv", rows, ["group_id", "rank", "phrase", "score", "documents_with_phrase", "document_prevalence", "groups_with_phrase"])
        return rows
    # Class-based TF-IDF: term prevalence inside a group is weighted by how
    # many distinct groups contain the phrase. This suppresses corpus-generic
    # vocabulary and favours contiguous phrases characteristic of one group.
    group_df = np.zeros(len(terms), dtype=np.int32)
    for other in all_groups:
        other_idx = np.where(groups == other)[0]
        group_df += (np.asarray((matrix[other_idx] > 0).sum(axis=0)).ravel() > 0).astype(np.int32)
    global_idf = np.log((len(all_groups) + 1.0) / (group_df + 1.0)) + 1.0
    for group in all_groups:
        idx = np.where(groups == group)[0].tolist()
        sub = matrix[idx]
        tf = np.asarray(sub.sum(axis=0)).ravel().astype(float)
        df = np.asarray((sub > 0).sum(axis=0)).ravel().astype(int)
        n_docs = len(idx)
        score = (tf / max(float(tf.sum()), 1.0)) * global_idf
        valid = np.where(df >= min(2, n_docs))[0]
        order = valid[np.argsort(score[valid])[::-1]]
        rank = 0
        for j in order:
            phrase = str(terms[j])
            if len(phrase) < 3:
                continue
            rank += 1
            rows.append({
                "group_id": group, "rank": rank, "phrase": phrase,
                "score": float(score[j]), "documents_with_phrase": int(df[j]),
                "document_prevalence": float(df[j] / n_docs),
                "groups_with_phrase": int(group_df[j]),
            })
            if rank >= 15:
                break
    write_csv(out_dir / "group_keywords.csv", rows, ["group_id", "rank", "phrase", "score", "documents_with_phrase", "document_prevalence", "groups_with_phrase"])
    return rows


def make_subtopic_diagnostics(vectors_norm: np.ndarray, groups: np.ndarray, documents: list[dict[str, Any]], matrix, terms: np.ndarray, out_dir: Path, max_k: int, min_group_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_groups = sorted(set(groups.tolist()), key=natural_group_key)
    candidate_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for group in all_groups:
        if group == "NOISE":
            continue
        idx = np.where(groups == group)[0]
        n = len(idx)
        if n < min_group_size:
            continue
        local = vectors_norm[idx]
        sim = np.clip(local @ local.T, -1.0, 1.0)
        dist = np.maximum(0.0, 1.0 - sim)
        np.fill_diagonal(dist, 0.0)
        best: dict[str, Any] | None = None
        max_allowed_k = min(max_k, n // 3)
        for k in range(2, max_allowed_k + 1):
            try:
                try:
                    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
                except TypeError:  # compatibility with older scikit-learn
                    model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
                labels = model.fit_predict(dist)
                counts = np.bincount(labels)
                min_size = int(counts.min())
                if min_size < 3:
                    continue
                score = float(silhouette_score(dist, labels, metric="precomputed"))
                sizes = sorted([int(x) for x in counts], reverse=True)
                row = {
                    "group_id": group,
                    "documents": n,
                    "k": k,
                    "silhouette_cosine": score,
                    "cluster_sizes": ";".join(map(str, sizes)),
                    "min_cluster_size": min_size,
                    "max_cluster_size": int(counts.max()),
                    "status": "POTENTIAL_STRUCTURE" if score >= 0.10 else ("WEAK_STRUCTURE" if score >= 0.05 else "NO_CLEAR_STRUCTURE"),
                }
                candidate_rows.append(row)
                if best is None or score > best["score"]:
                    best = {"score": score, "k": k, "labels": labels.copy(), "counts": counts.copy(), "status": row["status"]}
            except Exception as exc:
                candidate_rows.append({"group_id": group, "documents": n, "k": k, "silhouette_cosine": "", "cluster_sizes": "", "min_cluster_size": "", "max_cluster_size": "", "status": f"SKIPPED: {type(exc).__name__}: {exc}"})
        if best is None:
            continue
        local_labels = best["labels"]
        for local_idx, raw_cluster in enumerate(local_labels):
            doc_index = int(idx[local_idx])
            doc = documents[doc_index]
            assignment_rows.append({
                "document_id": doc["document_id"],
                "title": doc["title"],
                "parent_group": group,
                "candidate_subtopic": f"{group}_T{int(raw_cluster):02d}",
                "selected_k": best["k"],
                "best_silhouette": best["score"],
                "partition_status": best["status"],
                "text_source": doc["text_source"],
            })
        # attach terms for the selected exploratory partition
        for cluster_id in sorted(set(local_labels.tolist())):
            selected_global = [int(idx[i]) for i in np.where(local_labels == cluster_id)[0]]
            ranked = rank_terms_for_indices(matrix, terms, selected_global, max_terms=6)
            term_string = " | ".join(t[0] for t in ranked[:6])
            for row in candidate_rows:
                if row["group_id"] == group and row.get("k") == best["k"]:
                    row.setdefault("candidate_subtopics", [])
                    row["candidate_subtopics"].append(f"T{int(cluster_id):02d}: {term_string}")
    for row in candidate_rows:
        if isinstance(row.get("candidate_subtopics"), list):
            row["candidate_subtopics"] = " || ".join(row["candidate_subtopics"])
    write_csv(out_dir / "internal_subtopic_candidates.csv", candidate_rows)
    write_csv(out_dir / "internal_subtopic_assignments.csv", assignment_rows)
    return candidate_rows, assignment_rows


def compute_diagnostics(X: np.ndarray, groups: np.ndarray, ids: list[str], documents: list[dict[str, Any]], top_neighbors: int):
    n = len(ids)
    similarities = np.clip(X @ X.T, -1.0, 1.0)
    np.fill_diagonal(similarities, -np.inf)
    group_names = sorted(set(groups.tolist()), key=natural_group_key)
    group_indices = {g: np.where(groups == g)[0] for g in group_names}
    k = min(max(1, top_neighbors), n - 1)
    # Sorting is deterministic on equal similarities.
    neighbor_indices = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    neighbor_purity = {}
    for kk in sorted(set([min(5, k), k])):
        vals = np.asarray([np.mean(groups[neighbor_indices[i, :kk]] == groups[i]) for i in range(n)])
        neighbor_purity[kk] = vals

    centroids: dict[str, np.ndarray] = {}
    centroid_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for group in group_names:
        c = X[group_indices[group]].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[group] = c / norm if norm else c
    centroid_names = group_names
    centroid_sim = np.stack([centroids[g] for g in centroid_names])
    centroid_sim = np.clip(centroid_sim @ centroid_sim.T, -1.0, 1.0)
    for i, a in enumerate(group_names):
        for j, b in enumerate(group_names):
            centroid_matrix[a][b] = float(centroid_sim[i, j])

    # Mean pairwise cosine within each group and cross-group means.
    group_metrics: list[dict[str, Any]] = []
    document_metrics: list[dict[str, Any]] = []
    for group in group_names:
        idx = group_indices[group]
        sub = similarities[np.ix_(idx, idx)]
        if len(idx) > 1:
            pair_vals = sub[np.triu_indices(len(idx), k=1)]
            pair_vals = pair_vals[np.isfinite(pair_vals)]
            pair_mean = float(np.mean(pair_vals)) if len(pair_vals) else float("nan")
            pair_median = float(np.median(pair_vals)) if len(pair_vals) else float("nan")
            pair_p10 = float(np.percentile(pair_vals, 10)) if len(pair_vals) else float("nan")
            pair_p90 = float(np.percentile(pair_vals, 90)) if len(pair_vals) else float("nan")
        else:
            pair_mean = pair_median = pair_p10 = pair_p90 = float("nan")
        local_docs = X[idx]
        centroid_scores = local_docs @ centroids[group]
        own_vs_other = []
        best_other_groups = []
        for doc_index in idx:
            scores = {other: float(X[doc_index] @ centroids[other]) for other in group_names if other != group}
            best_other = max(scores, key=scores.get) if scores else ""
            best_score = scores[best_other] if best_other else float("nan")
            own_score = float(X[doc_index] @ centroids[group])
            margin = own_score - best_score if best_other else float("nan")
            own_vs_other.append(margin)
            best_other_groups.append(best_other)
            kn = neighbor_indices[doc_index]
            same5 = float(np.mean(groups[kn[:min(5, len(kn))]] == group))
            same10 = float(np.mean(groups[kn[:min(10, len(kn))]] == group))
            title = documents[doc_index]["title"]
            document_metrics.append({
                "document_id": ids[doc_index], "group_id": group, "title": title,
                "own_centroid_cosine": own_score,
                "best_other_group": best_other,
                "best_other_centroid_cosine": best_score,
                "centroid_margin": margin,
                "top5_same_group": same5,
                "top10_same_group": same10,
                "text_source": documents[doc_index]["text_source"],
            })
        p5 = min(5, k)
        p10 = min(10, k)
        expected10 = max(0, (len(idx) - 1) / max(1, n - 1))
        margin_arr = np.asarray(own_vs_other, dtype=float)
        negative_margin_share = float(np.mean(margin_arr < 0)) if len(margin_arr) else float("nan")
        avg_margin = float(np.nanmean(margin_arr)) if len(margin_arr) else float("nan")
        avg_p5 = float(np.mean(neighbor_purity[p5][idx])) if p5 in neighbor_purity else float("nan")
        avg_p10 = float(np.mean(neighbor_purity[p10][idx])) if p10 in neighbor_purity else float("nan")
        # Flags are review indicators, not quality grades or automatic reassignment.
        reasons = []
        if negative_margin_share >= 0.20:
            reasons.append("many_documents_closer_to_other_centroid")
        if avg_p10 < expected10 + 0.12:
            reasons.append("weak_local_neighbor_enrichment")
        if avg_margin < 0.015:
            reasons.append("small_mean_centroid_margin")
        group_metrics.append({
            "group_id": group,
            "documents": len(idx),
            "mean_pairwise_cosine": pair_mean,
            "median_pairwise_cosine": pair_median,
            "pairwise_p10": pair_p10,
            "pairwise_p90": pair_p90,
            "mean_centroid_cosine": float(np.mean(centroid_scores)),
            "centroid_p10": float(np.percentile(centroid_scores, 10)),
            "mean_centroid_margin": avg_margin,
            "negative_centroid_margin_share": negative_margin_share,
            "mean_top5_same_group": avg_p5,
            "mean_top10_same_group": avg_p10,
            "expected_top10_same_group_random": expected10,
            "top10_neighbor_enrichment": (avg_p10 / expected10 if expected10 > 0 else ""),
            "review_signals": ";".join(reasons) if reasons else "none_detected_by_current_rules",
        })

    cross_rows: list[dict[str, Any]] = []
    cross_matrix = np.zeros((len(group_names), len(group_names)), dtype=float)
    for i, a in enumerate(group_names):
        ia = group_indices[a]
        for j, b in enumerate(group_names):
            ib = group_indices[b]
            if i == j:
                cross_matrix[i, j] = float(np.mean([r for r in similarities[np.ix_(ia, ib)][np.triu_indices(len(ia), k=1)] if np.isfinite(r)])) if len(ia) > 1 else float("nan")
                continue
            vals = similarities[np.ix_(ia, ib)]
            mean = float(np.mean(vals[np.isfinite(vals)]))
            cross_matrix[i, j] = mean
            if i < j:
                cross_rows.append({"group_a": a, "group_b": b, "documents_a": len(ia), "documents_b": len(ib), "mean_pairwise_cosine": mean, "centroid_cosine": float(centroid_sim[i, j])})

    # Neighbor detail for all documents; only first top_neighbors are exported.
    all_neighbor_rows = []
    for i in range(n):
        for rank, j in enumerate(neighbor_indices[i], 1):
            all_neighbor_rows.append({
                "document_id": ids[i], "group_id": groups[i], "title": documents[i]["title"],
                "neighbor_rank": rank, "neighbor_id": ids[j], "neighbor_group": groups[j],
                "neighbor_title": documents[j]["title"], "cosine_similarity": float(similarities[i, j]),
            })
    return similarities, group_names, group_indices, centroids, centroid_sim, group_metrics, document_metrics, cross_rows, cross_matrix, neighbor_indices, all_neighbor_rows


def select_boundary_documents(document_metrics: list[dict[str, Any]], group_metrics: list[dict[str, Any]], groups: np.ndarray, documents: list[dict[str, Any]], per_group: int):
    rows: list[dict[str, Any]] = []
    metrics_by_group = {m["group_id"]: m for m in group_metrics}
    for group in sorted(set(groups.tolist()), key=natural_group_key):
        candidates = [r for r in document_metrics if r["group_id"] == group]
        candidates.sort(key=lambda r: (float(r["centroid_margin"]), float(r["own_centroid_cosine"]), r["document_id"]))
        group_indices = [i for i, x in enumerate(groups) if x == group]
        for rank, item in enumerate(candidates[:per_group], 1):
            row = dict(item)
            row["boundary_rank_in_group"] = rank
            row["group_review_signals"] = metrics_by_group[group]["review_signals"]
            row["abstract_source"] = documents[next(i for i in group_indices if documents[i]["document_id"] == item["document_id"])]["text_source"]
            row["boundary_reason"] = ";".join([
                reason for condition, reason in [
                    (float(item["centroid_margin"]) < 0, "closer_to_other_group_centroid"),
                    (float(item["top10_same_group"]) < 0.5, "less_than_half_of_top10_neighbors_in_own_group"),
                    (float(item["own_centroid_cosine"]) < 0.75, "low_own_centroid_similarity_relative_to_this_run"),
                ] if condition
            ]) or "relative_rank_among_group_documents"
            rows.append(row)
    return rows


def create_visualizations(out_dir: Path, X: np.ndarray, groups: np.ndarray, documents: list[dict[str, Any]], group_names: list[str], cross_matrix: np.ndarray, cross_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], neighbor_indices: np.ndarray, similarities: np.ndarray, ids: list[str], centroid_sim: np.ndarray) -> dict[str, str]:
    result: dict[str, str] = {}
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        # Semantic PCA map.
        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(X)
        write_csv(out_dir / "pca_coordinates.csv", [
            {"document_id": ids[i], "group_id": str(groups[i]), "title": documents[i]["title"],
             "pca_1": float(coordinates[i, 0]), "pca_2": float(coordinates[i, 1])}
            for i in range(len(ids))
        ])
        result["pca_coordinates"] = str(out_dir / "pca_coordinates.csv")
        fig, ax = plt.subplots(figsize=(14, 10))
        cmap = plt.get_cmap("tab20", max(20, len(group_names)))
        for gi, group in enumerate(group_names):
            idx = np.where(groups == group)[0]
            ax.scatter(coordinates[idx, 0], coordinates[idx, 1], s=19 if group != "NOISE" else 25,
                       alpha=0.68, label=f"{group} (n={len(idx)})", color=cmap(gi), marker="x" if group == "NOISE" else "o")
            cx, cy = coordinates[idx].mean(axis=0)
            ax.text(cx, cy, group, fontsize=8, weight="bold", ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        ax.set_title("SPECTER document space — PCA projection (exploratory view)")
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1, frameon=False)
        ax.grid(alpha=0.18)
        fig.tight_layout()
        path = fig_dir / "semantic_map_pca.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        result["semantic_map_pca"] = str(path)

        # Similarity heatmap, using average cross-document cosine.
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cross_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_xticks(range(len(group_names)), group_names, rotation=60, ha="right")
        ax.set_yticks(range(len(group_names)), group_names)
        ax.set_title("Mean cross-document cosine similarity between groups")
        fig.colorbar(im, ax=ax, label="Mean cosine similarity")
        fig.tight_layout()
        path = fig_dir / "group_similarity_heatmap.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        result["group_similarity_heatmap"] = str(path)

        # Group-level similarity graph: retain top 2 edges per group (union).
        try:
            import networkx as nx
            G = nx.Graph()
            sizes = {g: int(np.sum(groups == g)) for g in group_names}
            for g in group_names:
                G.add_node(g, documents=sizes[g])
            pair_lookup = {(r["group_a"], r["group_b"]): r["mean_pairwise_cosine"] for r in cross_rows}
            pair_lookup.update({(b, a): v for (a, b), v in list(pair_lookup.items())})
            selected_edges = set()
            for g in group_names:
                others = sorted([h for h in group_names if h != g], key=lambda h: pair_lookup.get((g, h), -1), reverse=True)[:2]
                for h in others:
                    selected_edges.add(tuple(sorted((g, h), key=natural_group_key)))
            for a, b in selected_edges:
                w = float(pair_lookup[(a, b)])
                G.add_edge(a, b, weight=w)
            pos = nx.spring_layout(G, seed=42, weight="weight", k=1.1 / math.sqrt(max(len(G), 1)))
            for node, (px, py) in pos.items():
                G.nodes[node]["x"] = float(px)
                G.nodes[node]["y"] = float(py)
            nx.write_graphml(G, out_dir / "group_similarity_network.graphml")
            fig, ax = plt.subplots(figsize=(13, 10))
            node_sizes = [220 + 23 * math.sqrt(sizes[g]) for g in G.nodes]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax, alpha=0.88, cmap=plt.get_cmap("tab20"), node_color=list(range(len(G.nodes))))
            weights = [G[u][v]["weight"] for u, v in G.edges]
            if weights:
                low, high = min(weights), max(weights)
                widths = [1.0 + 4.0 * (w - low) / (high - low + 1e-9) for w in weights]
                nx.draw_networkx_edges(G, pos, width=widths, alpha=0.55, ax=ax)
            labels = {g: f"{g}\n(n={sizes[g]})" for g in G.nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
            ax.set_title("Group similarity network — each group connected to its two strongest peers")
            ax.axis("off")
            fig.tight_layout()
            path = fig_dir / "group_similarity_network.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            result["group_similarity_network"] = str(path)
        except ImportError:
            result["group_similarity_network"] = "SKIPPED: networkx is not installed"

        # Boundary-document neighborhood graph, drawn in the same PCA coordinates.
        try:
            import networkx as nx
            G = nx.Graph()
            index_by_id = {doc_id: i for i, doc_id in enumerate(ids)}
            candidate_ids = {row["document_id"] for row in boundary_rows}
            for row in boundary_rows:
                i = index_by_id[row["document_id"]]
                G.add_node(ids[i], group=str(groups[i]), title=documents[i]["title"], candidate=True)
                for j in neighbor_indices[i, :3]:
                    neighbor_id = ids[int(j)]
                    G.add_node(neighbor_id, group=str(groups[int(j)]), title=documents[int(j)]["title"], candidate=neighbor_id in candidate_ids)
                    G.add_edge(ids[i], neighbor_id, weight=float(similarities[i, int(j)]))
            fig, ax = plt.subplots(figsize=(16, 13))
            coords = {node: (float(coordinates[index_by_id[node], 0]), float(coordinates[index_by_id[node], 1])) for node in G.nodes}
            for node, (px, py) in coords.items():
                G.nodes[node]["x"] = float(px)
                G.nodes[node]["y"] = float(py)
            nx.write_graphml(G, out_dir / "boundary_document_network.graphml")
            # Use PCA positions for semantic context, with slight deterministic jitter for exact overlaps.
            groups_for_nodes = sorted(set(G.nodes[n].get("group", "NOISE") for n in G.nodes), key=natural_group_key)
            color_index = {g: i for i, g in enumerate(groups_for_nodes)}
            colors = [color_index[G.nodes[node].get("group", "NOISE")] for node in G.nodes]
            sizes = [65 if G.nodes[node].get("candidate") else 22 for node in G.nodes]
            nx.draw_networkx_edges(G, coords, alpha=0.20, width=0.65, ax=ax)
            nx.draw_networkx_nodes(G, coords, node_color=colors, cmap=plt.get_cmap("tab20"), node_size=sizes, alpha=0.88, ax=ax)
            labels = {node: node[:5] for node in G.nodes if G.nodes[node].get("candidate")}
            nx.draw_networkx_labels(G, coords, labels=labels, font_size=6, ax=ax)
            ax.set_title("Boundary papers and their 3 nearest SPECTER neighbors (labels: first 5 ID chars)")
            ax.axis("off")
            fig.tight_layout()
            path = fig_dir / "boundary_document_network.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            result["boundary_document_network"] = str(path)
        except ImportError:
            result["boundary_document_network"] = "SKIPPED: networkx is not installed"

        # Interactive Plotly map and graph when available.
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            plot_df = pd.DataFrame({
                "x": coordinates[:, 0], "y": coordinates[:, 1], "group": groups,
                "document_id": ids, "title": [d["title"] for d in documents],
            })
            fig = px.scatter(plot_df, x="x", y="y", color="group", hover_data=["document_id", "title"],
                             title="Interactive SPECTER map — PCA projection")
            fig.update_traces(marker={"size": 6, "opacity": 0.72})
            html_path = out_dir / "semantic_map_interactive.html"
            fig.write_html(html_path, include_plotlyjs="cdn")
            result["semantic_map_interactive"] = str(html_path)

            # Interactive boundary graph.
            index_by_id = {doc_id: i for i, doc_id in enumerate(ids)}
            candidate_ids = {row["document_id"] for row in boundary_rows}
            node_ids: set[str] = set()
            edge_pairs: list[tuple[str, str, float]] = []
            for row in boundary_rows:
                source_id = row["document_id"]
                i = index_by_id[source_id]
                node_ids.add(source_id)
                for j in neighbor_indices[i, :3]:
                    target_id = ids[int(j)]
                    node_ids.add(target_id)
                    edge_pairs.append((source_id, target_id, float(similarities[i, int(j)])))
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            for a, b, _ in edge_pairs:
                ia, ib = index_by_id[a], index_by_id[b]
                edge_x.extend([coordinates[ia, 0], coordinates[ib, 0], None])
                edge_y.extend([coordinates[ia, 1], coordinates[ib, 1], None])
            ids_sorted = sorted(node_ids)
            node_x = [coordinates[index_by_id[x], 0] for x in ids_sorted]
            node_y = [coordinates[index_by_id[x], 1] for x in ids_sorted]
            node_group = [str(groups[index_by_id[x]]) for x in ids_sorted]
            node_title = [documents[index_by_id[x]]["title"] for x in ids_sorted]
            node_size = [11 if x in candidate_ids else 6 for x in ids_sorted]
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
            group_color_map = {g: palette[i % len(palette)] for i, g in enumerate(sorted(set(node_group), key=natural_group_key))}
            node_colors = [group_color_map[g] for g in node_group]
            graph_fig = go.Figure()
            graph_fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 0.7, "color": "rgba(120,120,120,0.25)"}, hoverinfo="skip", name="nearest-neighbor edges"))
            graph_fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker={"size": node_size, "color": node_colors, "showscale": False},
                                           text=[f"{g} | {t} | {doc_id}" for g, t, doc_id in zip(node_group, node_title, ids_sorted)],
                                           hovertemplate="%{text}<extra></extra>", name="documents"))
            graph_fig.update_layout(title="Boundary papers and nearest neighbors", xaxis_title="PCA 1", yaxis_title="PCA 2")
            graph_path = out_dir / "boundary_neighbor_graph_interactive.html"
            graph_fig.write_html(graph_path, include_plotlyjs="cdn")
            result["boundary_neighbor_graph_interactive"] = str(graph_path)
        except ImportError:
            result["interactive_visualizations"] = "SKIPPED: plotly/pandas is not installed"
    except ImportError as exc:
        result["static_visualizations"] = f"SKIPPED: matplotlib is not installed ({exc})"

    return result


def write_boundary_review(out_dir: Path, boundary_rows: list[dict[str, Any]], neighbor_rows: list[dict[str, Any]], documents_by_id: dict[str, dict[str, Any]], top_n_snippet: int = 1800) -> None:
    neighbor_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in neighbor_rows:
        neighbor_map[row["document_id"]].append(row)
    lines = [
        "# Пограничные публикации: содержательная проверка",
        "",
        "Материал сформирован автоматически по SPECTER-сходству. Это список для экспертной проверки, а не рекомендации по автоматическому переносу документов.",
        "",
    ]
    for row in boundary_rows:
        doc_id = row["document_id"]
        doc = documents_by_id[doc_id]
        lines += [
            f"## {row['group_id']} — {row['title']}",
            "",
            f"- ID: `{doc_id}`",
            f"- Ранг нетипичности внутри группы: {row['boundary_rank_in_group']}",
            f"- Сходство со своим центроидом: {float(row['own_centroid_cosine']):.4f}",
            f"- Ближайший альтернативный центроид: {row['best_other_group']} ({float(row['best_other_centroid_cosine']):.4f})",
            f"- Разница own − other: {float(row['centroid_margin']):+.4f}",
            f"- Доля соседей своей группы в top-10: {float(row['top10_same_group']):.2f}",
            f"- Источник текста: `{doc['text_source']}`",
            "",
            "**Ближайшие публикации:**",
            "",
        ]
        for near in neighbor_map.get(doc_id, [])[:5]:
            lines.append(f"- [{near['neighbor_group']}] {near['neighbor_title']} (cos={float(near['cosine_similarity']):.4f}; `{near['neighbor_id']}`)")
        snippet = doc.get("research_text", "")
        if snippet:
            lines += ["", f"**Фрагмент ({doc['text_source']}):**", "", snippet[:top_n_snippet].replace("\n", " "), ""]
        else:
            lines += ["", "Текстовый файл не найден или пригодный фрагмент не извлечён.", ""]
        lines.append("---")
        lines.append("")
    (out_dir / "boundary_review.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(out_dir: Path, metadata: dict[str, Any], group_metrics: list[dict[str, Any]], cross_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], subtopic_rows: list[dict[str, Any]], keyword_rows: list[dict[str, Any]], viz: dict[str, str]) -> None:
    group_keywords: dict[str, list[str]] = defaultdict(list)
    for row in keyword_rows:
        if row["rank"] <= 8:
            group_keywords[row["group_id"]].append(row["phrase"])
    lines = [
        "# Отчёт семантического аудита тематических групп",
        "",
        f"Версия пайплайна: `{VERSION}`",
        f"Дата запуска: `{metadata['created_at']}`",
        "",
        "## 1. Входные данные и проверка",
        "",
        f"- Документов: **{metadata['documents']}**",
        f"- Размерность SPECTER: **{metadata['embedding_dimension']}**",
        f"- Модель: `{metadata.get('model', 'unknown')}`",
        f"- Групп с NOISE: **{metadata['groups']}**",
        f"- ID векторов и CSV совпали: **{metadata['id_matching']}**",
        f"- CSV меток: `{metadata['labels_csv']}`",
        f"- Каталог эмбеддингов: `{metadata['embedding_dir']}`",
        f"- Текстовые документы найдены: **{metadata['text_files_found']} / {metadata['documents']}**",
        f"- Источник текста: {metadata['text_source_counts']}",
        "",
        "## 2. Метрики групп",
        "",
        "`top10_enrichment` сравнивает долю соседей своей группы с ожидаемой долей при случайном выборе из корпуса. Метрики — индикаторы структуры, не абсолютные оценки качества.",
        "",
        "| Группа | n | Попарное cos | Центроид | Margin | Top-10 same | Random baseline | Enrichment | Сигналы |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(group_metrics, key=lambda r: natural_group_key(r["group_id"])):
        def fmt(value):
            return "—" if value == "" or value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.3f}"
        lines.append(f"| {row['group_id']} | {row['documents']} | {fmt(row['mean_pairwise_cosine'])} | {fmt(row['mean_centroid_cosine'])} | {fmt(row['mean_centroid_margin'])} | {fmt(row['mean_top10_same_group'])} | {fmt(row['expected_top10_same_group_random'])} | {fmt(row['top10_neighbor_enrichment'])} | {row['review_signals']} |")
    lines += ["", "## 3. Наиболее близкие пары разных групп", "", "| Группы | Среднее межгрупповое cos | Cos центроидов |", "|---|---:|---:|"]
    for row in sorted(cross_rows, key=lambda r: r["mean_pairwise_cosine"], reverse=True)[:20]:
        lines.append(f"| {row['group_a']} — {row['group_b']} | {row['mean_pairwise_cosine']:.4f} | {row['centroid_cosine']:.4f} |")
    lines += ["", "## 4. Кандидаты на внутреннюю тематическую структуру", "", "Результаты агломеративного разбиения приведены как исследовательские кандидаты. Разбиение не изменяет исходные группы.", "", "| Группа | k | Silhouette (cosine) | Размеры подгрупп | Статус |", "|---|---:|---:|---|---|"]
    best_sub: dict[str, dict[str, Any]] = {}
    for row in subtopic_rows:
        if not isinstance(row.get("silhouette_cosine"), (int, float)):
            continue
        g = row["group_id"]
        if g not in best_sub or row["silhouette_cosine"] > best_sub[g]["silhouette_cosine"]:
            best_sub[g] = row
    for group in sorted(best_sub, key=natural_group_key):
        row = best_sub[group]
        lines.append(f"| {group} | {row['k']} | {row['silhouette_cosine']:.4f} | {row['cluster_sizes']} | {row['status']} |")
    if not best_sub:
        lines.append("| — | — | — | — | Не удалось получить допустимые разбиения |")
    lines += ["", "## 5. Пограничные публикации", "", f"Сформировано **{len(boundary_rows)}** кандидатов (до заданного количества на каждую группу). Подробности и текстовые фрагменты находятся в `boundary_review.md`.", ""]
    for row in boundary_rows:
        if row["boundary_rank_in_group"] <= 2:
            lines.append(f"- **{row['group_id']}** — {row['title']} (margin={float(row['centroid_margin']):+.3f}; best other={row['best_other_group']})")
    lines += ["", "## 6. Ключевые фразы", ""]
    for group in sorted(group_keywords, key=natural_group_key):
        lines.append(f"- **{group}:** " + "; ".join(group_keywords[group]))
    lines += ["", "## 7. Визуализации", ""]
    for name, path in viz.items():
        lines.append(f"- `{name}`: `{path}`")
    lines += [
        "",
        "## 8. Методические ограничения",
        "",
        "1. SPECTER-сходство отражает геометрию обученного пространства эмбеддингов; косинус 0.8 не означает 80% тематического совпадения.",
        "2. PCA-карта является двумерной проекцией и искажает часть расстояний. Она предназначена для визуального осмотра, а не для повторной кластеризации.",
        "3. Низкая доля соседей собственной группы, малый центроидный margin и близость к другой группе — сигналы проверки, но не основания автоматически менять метку.",
        "4. Agglomerative/Silhouette-подгруппы — кандидаты для экспертного анализа. Низкий silhouette не доказывает однородность группы.",
        "5. Ключевые фразы построены только из смежных n-грамм исходного текста; искусственное склеивание несоседних слов не выполняется.",
        "6. Для части документов аннотация может отсутствовать. В таблицах указан фактический источник фрагмента: abstract, introduction, fallback или title-only.",
        "7. NOISE — остаточная категория, а не обычный тематический кластер; её центроидные метрики следует трактовать только как подсказки при разборе шума.",
        "8. Исходные HDBSCAN-метки, векторы и файлы корпуса не изменяются.",
        "",
    ]
    (out_dir / "semantic_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Корень проекта не найден: {root}")
    emb_dir, manifest_path, manifest, vectors = discover_embedding_assets(root, args.embedding_dir)
    ids = [str(x) for x in manifest["ids"]]
    labels_path, rows, candidates_log = discover_labels_csv(root, ids, args.labels_csv)
    rows_by_id = {str(row["document_id"]).strip(): row for row in rows}
    if len(rows_by_id) != len(ids):
        raise ValueError("Число уникальных строк CSV не совпадает с числом ID эмбеддингов")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "reports" / "semantic_audit" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    text_source_counts: Counter[str] = Counter()
    text_files_found = 0
    for doc_id in ids:
        row = rows_by_id[doc_id]
        title = (row.get("title") or row.get("paper_title") or "").strip()
        group = (row.get("group_id") or "NOISE").strip() or "NOISE"
        text_path = locate_document_text(root, doc_id)
        raw = ""
        text_source = "title_only_or_missing"
        research_text = ""
        if text_path and text_path.is_file():
            text_files_found += 1
            try:
                raw = text_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                raw = text_path.read_text(encoding="latin-1", errors="replace")
            research_text, text_source = extract_research_text(raw)
        text_source_counts[text_source] += 1
        analysis_text = (title + " " + research_text).strip() if research_text else title
        documents.append({
            "document_id": doc_id,
            "title": title or "[title missing]",
            "group_id": group,
            "domain": row.get("domain", ""),
            "text_path": str(text_path) if text_path else "",
            "text_source": text_source,
            "research_text": research_text,
            "analysis_text": analysis_text,
        })

    groups = np.asarray([d["group_id"] for d in documents], dtype=object)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    X = vectors / norms
    group_names = sorted(set(groups.tolist()), key=natural_group_key)
    log("\nSEMANTIC AUDIT PIPELINE")
    log(f"Version: {VERSION}")
    log(f"Root: {root}")
    log(f"Embedding directory: {emb_dir}")
    log(f"Manifest: {manifest_path}")
    log(f"Vector matrix: {vectors.shape} {vectors.dtype}")
    log(f"Selected labels CSV: {labels_path}")
    log(f"ID matching: OK ({len(ids)} matched; 0 missing; 0 extra)")
    log(f"Documents: {len(ids)} | groups: {len(group_names)}")
    log(f"Processed text files: {text_files_found}/{len(ids)}")
    log("CSV candidates:")
    for item in candidates_log[:12]:
        log("  " + item)
    if len(candidates_log) > 12:
        log(f"  ... еще {len(candidates_log) - 12}")

    similarities, group_names, group_indices, centroids, centroid_sim, group_metrics, document_metrics, cross_rows, cross_matrix, neighbor_indices, all_neighbor_rows = compute_diagnostics(
        X, groups, ids, documents, max(10, args.neighbors)
    )

    # Lexical audit uses only document titles plus detected abstracts/intro excerpts.
    keyword_matrix, terms, keyword_error = build_keyword_matrix(documents)
    keyword_rows = save_group_keywords(documents, groups, keyword_matrix, terms, out_dir)
    if keyword_error:
        log(f"Keyword extraction warning: {keyword_error}")

    boundary_rows = select_boundary_documents(document_metrics, group_metrics, groups, documents, args.boundary_per_group)
    boundary_ids = {r["document_id"] for r in boundary_rows}
    index_by_id = {doc_id: i for i, doc_id in enumerate(ids)}
    boundary_neighbor_rows: list[dict[str, Any]] = []
    for row in boundary_rows:
        i = index_by_id[row["document_id"]]
        for rank, j in enumerate(neighbor_indices[i, :args.neighbors], 1):
            boundary_neighbor_rows.append({
                "document_id": row["document_id"], "group_id": row["group_id"], "title": row["title"],
                "neighbor_rank": rank, "neighbor_id": ids[int(j)], "neighbor_group": groups[int(j)],
                "neighbor_title": documents[int(j)]["title"], "cosine_similarity": float(similarities[i, int(j)]),
            })

    subtopic_rows, subtopic_assignments = make_subtopic_diagnostics(
        X, groups, documents, keyword_matrix, terms, out_dir, args.max_k, args.min_group_size
    )

    # Per-document complete audit table.
    write_csv(out_dir / "document_semantic_metrics.csv", document_metrics)
    write_csv(out_dir / "group_metrics.csv", group_metrics)
    write_csv(out_dir / "between_group_similarity.csv", sorted(cross_rows, key=lambda r: r["mean_pairwise_cosine"], reverse=True))
    write_csv(out_dir / "all_document_neighbors.csv", all_neighbor_rows)
    write_csv(out_dir / "boundary_documents.csv", boundary_rows)
    write_csv(out_dir / "boundary_neighbors.csv", boundary_neighbor_rows)

    text_audit_rows = [{
        "document_id": d["document_id"], "group_id": d["group_id"], "title": d["title"],
        "text_found": bool(d["text_path"]), "text_path": d["text_path"], "text_source": d["text_source"],
        "research_text_characters": len(d["research_text"]),
    } for d in documents]
    write_csv(out_dir / "document_text_audit.csv", text_audit_rows)

    documents_by_id = {d["document_id"]: d for d in documents}
    write_boundary_review(out_dir, boundary_rows, boundary_neighbor_rows, documents_by_id)

    viz = create_visualizations(out_dir, X, groups, documents, group_names, cross_matrix, cross_rows, boundary_rows, neighbor_indices, similarities, ids, centroid_sim)

    metadata = {
        "pipeline_version": VERSION,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "embedding_dir": str(emb_dir),
        "manifest_path": str(manifest_path),
        "vectors_path": str(emb_dir / "vectors.npy"),
        "model": manifest.get("model", "unknown"),
        "embedding_dimension": int(vectors.shape[1]),
        "documents": len(ids),
        "groups": len(group_names),
        "group_sizes": {g: int(np.sum(groups == g)) for g in group_names},
        "labels_csv": str(labels_path),
        "id_matching": "OK: exact equality of unique document_id sets",
        "text_files_found": text_files_found,
        "text_source_counts": dict(text_source_counts),
        "boundary_per_group": args.boundary_per_group,
        "neighbors_per_document": args.neighbors,
        "subtopic_max_k": args.max_k,
        "subtopic_min_group_size": args.min_group_size,
        "keyword_extraction_error": keyword_error,
        "visualizations": viz,
        "notes": [
            "No embedding model inference was performed.",
            "No source files, embeddings, or original HDBSCAN labels were modified.",
            "All candidate subtopics and boundary flags are diagnostic only.",
        ],
    }
    (out_dir / "semantic_audit_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, metadata, group_metrics, cross_rows, boundary_rows, subtopic_rows, keyword_rows, viz)

    log("\n=== GROUP OVERVIEW ===")
    for row in group_metrics:
        log(f"{row['group_id']:6} n={row['documents']:3} pair={row['mean_pairwise_cosine']:.4f} centroid={row['mean_centroid_cosine']:.4f} margin={row['mean_centroid_margin']:+.4f} top10={row['mean_top10_same_group']:.3f} baseline={row['expected_top10_same_group_random']:.3f} signals={row['review_signals']}")
    log("\n=== TOP INTER-GROUP PAIRS ===")
    for row in sorted(cross_rows, key=lambda r: r["mean_pairwise_cosine"], reverse=True)[:15]:
        log(f"{row['group_a']:6} - {row['group_b']:6} mean_cosine={row['mean_pairwise_cosine']:.4f} centroid={row['centroid_cosine']:.4f}")
    log("\n=== BEST INTERNAL PARTITION CANDIDATE PER GROUP ===")
    best: dict[str, dict[str, Any]] = {}
    for row in subtopic_rows:
        if isinstance(row.get("silhouette_cosine"), (float, int)) and (row["group_id"] not in best or row["silhouette_cosine"] > best[row["group_id"]]["silhouette_cosine"]):
            best[row["group_id"]] = row
    for group in sorted(best, key=natural_group_key):
        row = best[group]
        log(f"{group:6} k={row['k']} silhouette={row['silhouette_cosine']:.4f} sizes={row['cluster_sizes']} status={row['status']}")
    log(f"\nBoundary documents: {len(boundary_rows)}")
    log(f"Output directory: {out_dir}")
    log("Report: " + str(out_dir / "semantic_audit_summary.md"))
    log("Boundary review: " + str(out_dir / "boundary_review.md"))
    log("Готово. Исходные HDBSCAN-метки и матрица SPECTER не изменялись.")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Полный детерминированный аудит SPECTER-кластеров")
    parser.add_argument("--root", default=".", help="Корень проекта (по умолчанию текущий каталог)")
    parser.add_argument("--labels-csv", default=None, help="Явный путь к clusters.csv / cluster_assignments.csv")
    parser.add_argument("--embedding-dir", default=None, help="Явный каталог с vectors.npy и manifest.json")
    parser.add_argument("--output-dir", default=None, help="Каталог результатов; по умолчанию reports/semantic_audit/<timestamp>")
    parser.add_argument("--boundary-per-group", type=int, default=5, help="Сколько пограничных документов брать из каждой группы (default: 5)")
    parser.add_argument("--neighbors", type=int, default=5, help="Сколько ближайших соседей экспортировать (default: 5)")
    parser.add_argument("--max-k", type=int, default=5, help="Максимум подгрупп для исследовательского разбиения (default: 5)")
    parser.add_argument("--min-group-size", type=int, default=12, help="Минимальный размер родительской группы для подгруппового анализа")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.boundary_per_group < 1 or args.neighbors < 1 or args.max_k < 2 or args.min_group_size < 4:
        parser.error("boundary-per-group и neighbors должны быть >=1, max-k >=2, min-group-size >=4")
    try:
        run_pipeline(args)
        return 0
    except Exception as exc:
        log(f"\nОШИБКА ПАЙПЛАЙНА: {type(exc).__name__}: {exc}")
        if args.root:
            log("Проверьте корень проекта, наличие SPECTER manifest/vector и thematic CSV.")
        if "--debug" in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
