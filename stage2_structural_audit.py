#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 structural/semantic audit for a saved Louvain vs Leiden comparison.

Reads only saved Stage 1 assignments and the existing SPECTER embedding cache.
It does not rebuild or modify the graph, embeddings, or Stage 1 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ID_KEYS = {
    "document_id", "documentid", "doc_id", "docid", "paper_id", "paperid",
    "openalex_id", "id"
}
ID_LIST_KEYS = {
    "document_ids", "documentids", "doc_ids", "paper_ids", "ordered_document_ids",
    "ordered_ids", "row_ids", "ids", "documents"
}
INDEX_MAP_KEYS = {"document_id_to_index", "id_to_index", "document_to_index", "row_by_id"}
STOP_WORDS = {
    "using", "based", "via", "toward", "towards", "approach", "approaches",
    "method", "methods", "study", "analysis", "new", "novel", "framework",
    "frameworks", "design", "overview", "review", "case", "cases", "paper",
    "efficient", "effective", "learning", "model", "models"
}


def norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(value).lower())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def extract_manifest_order(manifest: Any, expected_ids: set[str], n_vectors: int) -> tuple[list[str], str]:
    """Find a trustworthy row-order ID list in a potentially variant manifest schema."""
    candidates: list[tuple[float, list[str], str]] = []

    def add_candidate(values: Any, path: str) -> None:
        if not isinstance(values, (list, tuple)) or not values:
            return
        ids: list[str] = []
        for item in values:
            if isinstance(item, (str, int)):
                ids.append(str(item))
            elif isinstance(item, dict):
                found = None
                for k, v in item.items():
                    if norm_key(k) in ID_KEYS and isinstance(v, (str, int)):
                        found = str(v)
                        break
                if found is None:
                    return
                ids.append(found)
            else:
                return
        if len(ids) != n_vectors or len(set(ids)) != len(ids):
            return
        overlap = len(set(ids) & expected_ids)
        # Prefer exact corpus coverage; then the strongest overlap.
        exact = set(ids) == expected_ids
        score = (10_000 if exact else 0) + overlap
        candidates.append((score, ids, path))

    def walk(node: Any, path: str = "manifest") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                nk = norm_key(key)
                child_path = f"{path}.{key}"
                if nk in ID_LIST_KEYS:
                    add_candidate(value, child_path)
                if nk in INDEX_MAP_KEYS and isinstance(value, dict):
                    try:
                        ordered = sorted(((int(i), str(k)) for k, i in value.items()), key=lambda x: x[0])
                        if ordered and [i for i, _ in ordered] == list(range(len(ordered))):
                            add_candidate([item_id for _, item_id in ordered], child_path)
                    except (TypeError, ValueError):
                        pass
                # Also recognize record lists regardless of their parent key.
                if isinstance(value, list):
                    add_candidate(value, child_path)
                walk(value, child_path)
        elif isinstance(node, list):
            # A list of records may be nested without a recognized parent key.
            add_candidate(node, path)
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(manifest)
    if not candidates:
        raise RuntimeError(
            "Не удалось восстановить порядок document_id из manifest.json. "
            "Скрипт не будет предполагать, что порядок CSV совпадает с порядком vectors.npy. "
            "Нужна структура manifest.json (верхние ключи и пример одной записи)."
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_ids, best_path = candidates[0]
    if set(best_ids) != expected_ids:
        missing = sorted(expected_ids - set(best_ids))[:5]
        extra = sorted(set(best_ids) - expected_ids)[:5]
        raise RuntimeError(
            f"Manifest candidate {best_path} does not exactly match assignment IDs "
            f"({len(set(best_ids) & expected_ids)}/{len(expected_ids)} matched). "
            f"Missing examples={missing}; extra examples={extra}."
        )
    return best_ids, best_path


def safe_int_series(series: pd.Series, column: str) -> pd.Series:
    try:
        converted = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"Столбец {column} содержит нечисловые cluster ID: {exc}") from exc
    if converted.isna().any():
        raise RuntimeError(f"Столбец {column} содержит пустые cluster ID")
    return converted.astype(int)


def average_within_cosine(unit_vectors: np.ndarray) -> float:
    n = len(unit_vectors)
    if n < 2:
        return float("nan")
    s = unit_vectors.sum(axis=0, dtype=np.float64)
    return float((np.dot(s, s) - n) / (n * (n - 1)))


def top_title_terms(titles: list[str], labels: np.ndarray) -> dict[int, str]:
    clean_titles = [t if isinstance(t, str) else "" for t in titles]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=sorted(STOP_WORDS),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(clean_titles)
        terms = np.asarray(vectorizer.get_feature_names_out())
        result: dict[int, str] = {}
        for cid in sorted(set(labels.tolist())):
            idx = np.flatnonzero(labels == cid)
            scores = np.asarray(matrix[idx].mean(axis=0)).ravel()
            order = np.argsort(scores)[::-1]
            result[int(cid)] = "; ".join(terms[j] for j in order[:12] if scores[j] > 0)
        return result
    except ValueError:
        return {int(cid): "" for cid in sorted(set(labels.tolist()))}


def community_stats(
    partition: str,
    labels: np.ndarray,
    unit_vectors: np.ndarray,
    titles: list[str],
    domains: list[str],
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, str]]:
    terms = top_title_terms(titles, labels)
    sums: dict[int, np.ndarray] = {}
    rows = []
    for cid in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == cid)
        members = unit_vectors[idx]
        s = members.sum(axis=0, dtype=np.float64)
        sums[int(cid)] = s
        centroid = s / max(np.linalg.norm(s), 1e-12)
        centroid_sims = members @ centroid
        top_idx = idx[np.argsort(centroid_sims)[::-1][:5]]
        domain_counts = Counter(domains[i] for i in idx if domains[i])
        rows.append({
            "partition": partition,
            "community_id": int(cid),
            "documents": int(len(idx)),
            "share_percent": float(100 * len(idx) / len(labels)),
            "mean_cosine_to_centroid": float(np.mean(centroid_sims)),
            "mean_pairwise_cosine_within": average_within_cosine(members),
            "dominant_domains": "; ".join(f"{k}:{v}" for k, v in domain_counts.most_common(5)),
            "title_terms_tfidf": terms.get(int(cid), ""),
            "representative_titles": " || ".join(str(titles[i])[:240] for i in top_idx),
        })
    return pd.DataFrame(rows), sums, terms


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 audit for saved Louvain/Leiden outputs")
    parser.add_argument("--root", required=True, help="Project root")
    parser.add_argument("--run-id", default="20261120_015622_graph_b63e68")
    parser.add_argument("--max-boundary", type=int, default=150, help="Number of most ambiguous documents to export")
    parser.add_argument("--silhouette-sample", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    stage1 = root / "reports" / "community_detection_stage1" / args.run_id
    output_dir = root / "reports" / "community_detection_stage2" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = root / "artifacts" / "embeddings" / "sentence-transformers_allenai-specter"
    assignments_path = stage1 / "leiden_assignments.csv"
    manifest_path = emb_dir / "manifest.json"
    vectors_path = emb_dir / "vectors.npy"
    for p in (assignments_path, manifest_path, vectors_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required input not found: {p}")

    print(f"[1/7] Reading Stage 1 assignments: {assignments_path}")
    df = pd.read_csv(assignments_path, dtype={"document_id": str}, encoding="utf-8-sig")
    required = {"document_id", "title", "louvain_cluster_id", "leiden_cluster_id"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"В assignments отсутствуют столбцы: {sorted(missing_cols)}")
    if df["document_id"].isna().any() or df["document_id"].duplicated().any():
        raise RuntimeError("document_id в assignments содержит пропуски или дубликаты")
    df["louvain_cluster_id"] = safe_int_series(df["louvain_cluster_id"], "louvain_cluster_id")
    df["leiden_cluster_id"] = safe_int_series(df["leiden_cluster_id"], "leiden_cluster_id")
    if "domain" not in df.columns:
        df["domain"] = ""
    df["domain"] = df["domain"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    ids = df["document_id"].astype(str).tolist()
    expected_ids = set(ids)
    n = len(df)

    print("[2/7] Loading the existing SPECTER matrix and resolving manifest row order")
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != n:
        raise RuntimeError(f"vectors.npy shape {vectors.shape} is incompatible with {n} assignment rows")
    manifest = load_json(manifest_path)
    manifest_ids, manifest_path_used = extract_manifest_order(manifest, expected_ids, vectors.shape[0])
    row_by_id = {doc_id: i for i, doc_id in enumerate(manifest_ids)}
    order = np.asarray([row_by_id[x] for x in ids], dtype=np.int64)
    X = np.asarray(vectors[order], dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms.ravel() <= 1e-12)[:10].tolist()
        raise RuntimeError(f"Найдены нулевые embedding-векторы; примеры индексов: {bad}")
    U = X / norms

    lou = df["louvain_cluster_id"].to_numpy(dtype=int)
    lei = df["leiden_cluster_id"].to_numpy(dtype=int)
    lou_ids = sorted(np.unique(lou).tolist())
    lei_ids = sorted(np.unique(lei).tolist())
    if -1 in lou_ids or -1 in lei_ids:
        print("WARNING: noise label -1 found; it will be treated as a separate label in overlap tables.")

    print("[3/7] Computing label-invariant correspondence")
    overlap = pd.crosstab(pd.Series(lou, name="louvain_community"), pd.Series(lei, name="leiden_community"))
    overlap = overlap.reindex(index=lou_ids, columns=lei_ids, fill_value=0)
    r_idx, c_idx = linear_sum_assignment(-overlap.to_numpy(dtype=np.int64))
    mapping = {int(lei_ids[c]): int(lou_ids[r]) for r, c in zip(r_idx, c_idx)}
    lei_aligned = np.asarray([mapping.get(int(x), -999999) for x in lei], dtype=int)
    changed_raw = lou != lei
    changed_aligned = lou != lei_aligned
    n_raw = int(changed_raw.sum())
    n_aligned = int(changed_aligned.sum())

    correspondence_rows = []
    for li in lou_ids:
        for le in lei_ids:
            inter = int(overlap.loc[li, le])
            if inter == 0:
                continue
            n_lou = int(np.sum(lou == li))
            n_lei = int(np.sum(lei == le))
            correspondence_rows.append({
                "louvain_community": li,
                "leiden_community": le,
                "intersection": inter,
                "jaccard": inter / (n_lou + n_lei - inter),
                "louvain_coverage": inter / n_lou,
                "leiden_coverage": inter / n_lei,
                "share_of_all_documents_percent": 100 * inter / n,
                "hungarian_matched": bool(mapping.get(le) == li),
            })
    correspondence = pd.DataFrame(correspondence_rows).sort_values("intersection", ascending=False)
    write_csv(correspondence, output_dir / "louvain_leiden_overlap.csv")

    transition_rows = []
    for li in lou_ids:
        for le in lei_ids:
            count = int(overlap.loc[li, le])
            if not count:
                continue
            transition_rows.append({
                "louvain_community": li,
                "leiden_community": le,
                "documents": count,
                "share_of_louvain_percent": 100 * count / int(np.sum(lou == li)),
                "share_of_leiden_percent": 100 * count / int(np.sum(lei == le)),
                "hungarian_matched": bool(mapping.get(le) == li),
            })
    transitions = pd.DataFrame(transition_rows).sort_values("documents", ascending=False)
    write_csv(transitions, output_dir / "document_transitions.csv")

    split_rows = []
    for li in lou_ids:
        counts = overlap.loc[li].sort_values(ascending=False)
        total = int(counts.sum())
        nonzero = counts[counts > 0]
        split_rows.append({
            "direction": "Louvain_to_Leiden",
            "community_id": li,
            "documents": total,
            "counterpart_communities": int(len(nonzero)),
            "counterparts_over_5pct": int((nonzero / total >= 0.05).sum()),
            "largest_counterpart_id": int(nonzero.index[0]),
            "largest_counterpart_documents": int(nonzero.iloc[0]),
            "largest_counterpart_share_percent": 100 * float(nonzero.iloc[0]) / total,
            "counterpart_distribution": "; ".join(f"{int(k)}:{int(v)} ({100*v/total:.1f}%)" for k, v in nonzero.items()),
        })
    for le in lei_ids:
        counts = overlap[le].sort_values(ascending=False)
        total = int(counts.sum())
        nonzero = counts[counts > 0]
        split_rows.append({
            "direction": "Leiden_from_Louvain",
            "community_id": le,
            "documents": total,
            "counterpart_communities": int(len(nonzero)),
            "counterparts_over_5pct": int((nonzero / total >= 0.05).sum()),
            "largest_counterpart_id": int(nonzero.index[0]),
            "largest_counterpart_documents": int(nonzero.iloc[0]),
            "largest_counterpart_share_percent": 100 * float(nonzero.iloc[0]) / total,
            "counterpart_distribution": "; ".join(f"{int(k)}:{int(v)} ({100*v/total:.1f}%)" for k, v in nonzero.items()),
        })
    split_merge = pd.DataFrame(split_rows)
    write_csv(split_merge, output_dir / "split_merge_analysis.csv")

    # Per-document transition details after the global best one-to-one label alignment.
    doc_out = df[["document_id", "title", "domain", "louvain_cluster_id", "leiden_cluster_id"]].copy()
    doc_out["leiden_aligned_to_louvain"] = lei_aligned
    doc_out["label_changed_before_alignment"] = changed_raw
    doc_out["community_changed_after_alignment"] = changed_aligned
    doc_out["hungarian_matched"] = [mapping.get(int(x)) == int(y) for x, y in zip(lei, lou)]

    print("[4/7] Computing embedding-space cohesion and boundary candidates")
    stats_lou, sums_lou, _ = community_stats("Louvain", lou, U, df["title"].tolist(), df["domain"].tolist())
    stats_lei, sums_lei, _ = community_stats("Leiden", lei, U, df["title"].tolist(), df["domain"].tolist())
    profiles = pd.concat([stats_lou, stats_lei], ignore_index=True)
    write_csv(profiles, output_dir / "community_semantic_profiles.csv")

    # Cosine similarity to own and nearest other community centroids, for each partition.
    def boundary_for(labels: np.ndarray, sums: dict[int, np.ndarray], partition: str) -> pd.DataFrame:
        cids = sorted(sums)
        centroids = []
        for cid in cids:
            c = sums[cid]
            centroids.append(c / max(np.linalg.norm(c), 1e-12))
        C = np.vstack(centroids).astype(np.float32)
        sims = U @ C.T
        cid_to_col = {cid: j for j, cid in enumerate(cids)}
        own = np.asarray([sims[i, cid_to_col[int(labels[i])]] for i in range(n)])
        other = sims.copy()
        for i, cid in enumerate(labels):
            other[i, cid_to_col[int(cid)]] = -np.inf
        best_col = np.argmax(other, axis=1)
        best_sim = other[np.arange(n), best_col]
        best_id = np.asarray(cids)[best_col]
        margin = own - best_sim
        return pd.DataFrame({
            "partition": partition,
            "document_id": ids,
            "title": df["title"].tolist(),
            "assigned_community": labels,
            "own_centroid_cosine": own,
            "nearest_other_community": best_id,
            "nearest_other_centroid_cosine": best_sim,
            "centroid_margin": margin,
            "nearest_centroid_would_disagree": margin < 0,
        })

    boundaries = pd.concat([
        boundary_for(lou, sums_lou, "Louvain"),
        boundary_for(lei, sums_lei, "Leiden"),
    ], ignore_index=True)
    # Export the most ambiguous rows for each partition, not just one partition.
    boundary_top = boundaries.sort_values("centroid_margin", ascending=True).groupby("partition", sort=False).head(args.max_boundary)
    write_csv(boundary_top, output_dir / "boundary_documents.csv")
    write_csv(doc_out, output_dir / "document_label_alignment.csv")

    # Exact average cosine across all pairs of distinct communities.
    inter_rows = []
    for i, a in enumerate(lou_ids):
        for b in lou_ids[i + 1:]:
            na, nb = int(np.sum(lou == a)), int(np.sum(lou == b))
            inter_rows.append({"partition": "Louvain", "community_a": a, "community_b": b,
                               "mean_cross_cosine": float(np.dot(sums_lou[a], sums_lou[b]) / (na * nb))})
    for i, a in enumerate(lei_ids):
        for b in lei_ids[i + 1:]:
            na, nb = int(np.sum(lei == a)), int(np.sum(lei == b))
            inter_rows.append({"partition": "Leiden", "community_a": a, "community_b": b,
                               "mean_cross_cosine": float(np.dot(sums_lei[a], sums_lei[b]) / (na * nb))})
    write_csv(pd.DataFrame(inter_rows), output_dir / "intercommunity_cosine.csv")

    print("[5/7] Computing same-sample silhouette and partition metrics")
    rng = np.random.default_rng(args.seed)
    sample_n = min(args.silhouette_sample, n)
    sample_idx = np.sort(rng.choice(n, size=sample_n, replace=False)) if sample_n < n else np.arange(n)
    silhouette_lou = float(silhouette_score(U[sample_idx], lou[sample_idx], metric="cosine"))
    silhouette_lei = float(silhouette_score(U[sample_idx], lei[sample_idx], metric="cosine"))
    ari = float(adjusted_rand_score(lou, lei))
    nmi = float(normalized_mutual_info_score(lou, lei))
    sizes_lou = pd.Series(lou).value_counts().sort_index()
    sizes_lei = pd.Series(lei).value_counts().sort_index()
    sizes_rows = []
    for name, series in (("Louvain", sizes_lou), ("Leiden", sizes_lei)):
        for cid, count in series.items():
            sizes_rows.append({"partition": name, "community_id": int(cid), "documents": int(count),
                               "share_percent": 100 * int(count) / n})
    write_csv(pd.DataFrame(sizes_rows), output_dir / "community_sizes_stage2.csv")

    print("[6/7] Writing plots")
    plot_dir = output_dir / "visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_status = "matplotlib unavailable; plots skipped"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(overlap.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(lei_ids)), [str(x) for x in lei_ids])
        ax.set_yticks(range(len(lou_ids)), [str(x) for x in lou_ids])
        ax.set_xlabel("Leiden community")
        ax.set_ylabel("Louvain community")
        ax.set_title("Louvain–Leiden overlap (document counts)")
        fig.colorbar(im, ax=ax, label="Documents")
        fig.tight_layout()
        fig.savefig(plot_dir / "overlap_heatmap.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for name, series in (("Louvain", sizes_lou), ("Leiden", sizes_lei)):
            ax.plot(range(1, len(series) + 1), sorted(series.values, reverse=True), marker="o", label=name)
        ax.set_xlabel("Community rank (largest first)")
        ax.set_ylabel("Documents")
        ax.set_title("Community size distribution")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "community_size_distribution.png", dpi=180)
        plt.close(fig)

        top_flows = transitions.head(25).copy()
        fig, ax = plt.subplots(figsize=(11, 7))
        y = np.arange(len(top_flows))
        labels_flow = [f"L{r.louvain_community} → Le{r.leiden_community}" for r in top_flows.itertuples()]
        ax.barh(y, top_flows["documents"].to_numpy())
        ax.set_yticks(y, labels_flow)
        ax.invert_yaxis()
        ax.set_xlabel("Documents")
        ax.set_title("Largest Louvain → Leiden overlaps")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "largest_transitions.png", dpi=180)
        plt.close(fig)
        plot_status = "3 plots created"
    except Exception as exc:
        plot_status = f"plot generation failed: {type(exc).__name__}: {exc}"

    print("[7/7] Writing summary and report")
    summary = {
        "stage": 2,
        "run_id": args.run_id,
        "documents": n,
        "embedding_shape": list(vectors.shape),
        "embedding_dim": int(vectors.shape[1]),
        "manifest_order_source": manifest_path_used,
        "louvain_communities": len(lou_ids),
        "leiden_communities": len(lei_ids),
        "louvain_ids": lou_ids,
        "leiden_ids": lei_ids,
        "raw_numeric_label_mismatches": n_raw,
        "raw_numeric_label_mismatch_percent": 100 * n_raw / n,
        "mismatches_after_hungarian_label_alignment": n_aligned,
        "mismatches_after_hungarian_label_alignment_percent": 100 * n_aligned / n,
        "documents_reassigned_after_label_alignment": n_aligned,
        "hungarian_leiden_to_louvain_mapping": {str(k): int(v) for k, v in mapping.items()},
        "matched_one_to_one_overlap_percent": 100 * sum(int(overlap.loc[mapping[le], le]) for le in mapping) / n,
        "ari_louvain_leiden": ari,
        "nmi_louvain_leiden": nmi,
        "silhouette_louvain_same_sample_cosine": silhouette_lou,
        "silhouette_leiden_same_sample_cosine": silhouette_lei,
        "silhouette_sample_size": sample_n,
        "silhouette_sample_seed": args.seed,
        "louvain_sizes": {str(k): int(v) for k, v in sizes_lou.items()},
        "leiden_sizes": {str(k): int(v) for k, v in sizes_lei.items()},
        "plots": plot_status,
        "inputs": {
            "assignments": str(assignments_path),
            "manifest": str(manifest_path),
            "vectors": str(vectors_path),
        },
        "note": "Title terms are TF-IDF descriptors from titles only; they are not YAKE/RAKE output or validated semantic labels.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def fmt(x: float) -> str:
        return f"{x:.4f}"

    lines = [
        "# Этап 2. Структурный аудит Louvain vs Leiden",
        "",
        f"Контрольный запуск: `{args.run_id}`",
        "",
        "## Главное",
        "",
        f"- Публикаций: **{n}**; сообществ Louvain: **{len(lou_ids)}**; сообществ Leiden: **{len(lei_ids)}**.",
        f"- Различаются числовые ID без выравнивания: **{n_raw} ({100*n_raw/n:.1f}%)**.",
        f"- Различаются назначения после оптимального взаимно-однозначного сопоставления ID (алгоритм Венгера): **{n_aligned} ({100*n_aligned/n:.1f}%)**.",
        f"- Доля документов, попавших в сопоставленные пары сообществ: **{summary['matched_one_to_one_overlap_percent']:.1f}%**.",
        f"- ARI/NMI между разбиениями: **{fmt(ari)} / {fmt(nmi)}**.",
        f"- Silhouette в исходном SPECTER-пространстве на одном и том же наборе из {sample_n} документов: Louvain **{fmt(silhouette_lou)}**, Leiden **{fmt(silhouette_lei)}**.",
        "",
        "## Сопоставление меток",
        "",
        "Числовые ID кластеров произвольны. Поэтому прямое сравнение `louvain_cluster_id != leiden_cluster_id` завышает число изменений, если алгоритмы просто присвоили сопоставимым сообществам разные номера. Основная оценка реальных изменений в этом отчёте использует оптимальное взаимно-однозначное сопоставление сообществ по максимальному суммарному пересечению.",
        "",
        "### Сопоставление Leiden → Louvain",
        "",
        "| Leiden ID | Соответствующий Louvain ID | Пересечение |",
        "|---:|---:|---:|",
    ]
    for le, lo in sorted(mapping.items()):
        lines.append(f"| {le} | {lo} | {int(overlap.loc[lo, le])} |")
    lines += [
        "",
        "## Семантические профили",
        "",
        "Термины рассчитаны TF-IDF по заголовкам публикаций. Это предварительные дескрипторы: для полноценной семантической интерпретации нужно дополнить их ключевыми фразами из аннотаций/полных текстов и существующими YAKE/RAKE-артефактами.",
        "",
        "| Разбиение | Группа | Документов | Среднее сходство с центроидом | Среднее попарное сходство внутри | Термины из заголовков |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in profiles.sort_values(["partition", "community_id"]).itertuples():
        lines.append(f"| {r.partition} | {r.community_id} | {r.documents} | {r.mean_cosine_to_centroid:.4f} | {r.mean_pairwise_cosine_within:.4f} | {r.title_terms_tfidf} |")
    lines += [
        "",
        "## Файлы",
        "",
        "- `louvain_leiden_overlap.csv` — все ненулевые пересечения и метрики покрытия.",
        "- `document_transitions.csv` — потоки публикаций между исходными ID групп.",
        "- `split_merge_analysis.csv` — распределение каждой группы по группам другой модели.",
        "- `document_label_alignment.csv` — назначения до и после выравнивания ID.",
        "- `boundary_documents.csv` — документы с наименьшим отрывом от центроида своей группы; это кандидаты для ручной проверки, а не гарантированные ошибки.",
        "- `community_semantic_profiles.csv` — размеры, cohesion и предварительные TF-IDF термины по заголовкам.",
        "- `intercommunity_cosine.csv` — средняя попарная косинусная близость между группами.",
        "- `community_sizes_stage2.csv` — размеры групп.",
        "- `visualizations_stage2/` — графики, если matplotlib установлен.",
        "",
        "## Ограничения",
        "",
        "- Внутригрупповое попарное сходство рассчитано точно по нормированным векторам; среднее сходство с центроидом — отдельная метрика и не заменяет silhouette.",
        "- TF-IDF по заголовкам не является заменой YAKE/RAKE и не доказывает тематическую однородность.",
        "- Близость к центроиду — эвристика для поиска пограничных публикаций; для density-based границ нужны веса рёбер/локальная структура графа.",
        "- Граф и embedding-кэш не перестраивались и не изменялись.",
        "",
    ]
    (output_dir / "REPORT_STAGE2_RU.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nDONE")
    print(f"Output: {output_dir}")
    print(f"Raw numeric label mismatches: {n_raw}/{n} ({100*n_raw/n:.2f}%)")
    print(f"After label alignment: {n_aligned}/{n} ({100*n_aligned/n:.2f}%)")
    print(f"Silhouette Louvain={silhouette_lou:.4f}; Leiden={silhouette_lei:.4f}")
    print(f"Plots: {plot_status}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
