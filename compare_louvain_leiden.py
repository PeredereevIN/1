from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

DEFAULT_RUN_ID = "20261120_015622_graph_b63e68"
DEFAULT_MODEL_DIR = Path("artifacts/embeddings/sentence-transformers_allenai-specter")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_membership(membership: list[int]) -> list[int]:
    """Assign stable display IDs by descending community size, then first index."""
    groups: dict[int, list[int]] = defaultdict(list)
    for node, label in enumerate(membership):
        groups[int(label)].append(node)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), min(item[1])))
    remap = {old: new for new, (old, _) in enumerate(ordered)}
    return [remap[int(label)] for label in membership]


def sizes_for_labels(labels: list[int] | np.ndarray) -> dict[str, int]:
    counts = Counter(int(x) for x in labels if int(x) != -1)
    return {str(k): int(v) for k, v in sorted(counts.items())}


def apply_minimum_size(labels: list[int], minimum: int) -> list[int]:
    counts = Counter(labels)
    return [label if counts[label] >= minimum else -1 for label in labels]


def connectivity_diagnostics(graph: nx.Graph, labels: list[int]) -> dict[str, Any]:
    groups: dict[int, list[str]] = defaultdict(list)
    nodes = list(graph.nodes())
    for node, label in zip(nodes, labels, strict=True):
        if int(label) != -1:
            groups[int(label)].append(node)

    disconnected_groups = 0
    disconnected_group_nodes = 0
    extra_components = 0
    per_group = []
    for label, members in sorted(groups.items()):
        component_count = nx.number_connected_components(graph.subgraph(members))
        per_group.append({"community_id": label, "size": len(members), "components": component_count})
        if component_count > 1:
            disconnected_groups += 1
            disconnected_group_nodes += len(members)
            extra_components += component_count - 1

    return {
        "community_count": len(groups),
        "disconnected_community_count": disconnected_groups,
        "nodes_in_disconnected_communities": disconnected_group_nodes,
        "extra_components_inside_communities": extra_components,
        "per_community": per_group,
    }


def partition_modularity(graph: nx.Graph, labels: list[int], resolution: float) -> float | None:
    groups: dict[int, set[str]] = defaultdict(set)
    for node, label in zip(graph.nodes(), labels, strict=True):
        if int(label) != -1:
            groups[int(label)].add(node)
    retained = set().union(*groups.values()) if groups else set()
    subgraph = graph.subgraph(retained)
    communities = list(groups.values())
    if subgraph.number_of_edges() == 0 or len(communities) < 2:
        return None
    return float(nx.community.modularity(subgraph, communities, weight="weight", resolution=resolution))


def compute_partition_metrics(
    graph: nx.Graph,
    labels: list[int],
    domains: list[str],
    resolution: float,
) -> dict[str, Any]:
    unique = sorted(set(int(x) for x in labels) - {-1})
    non_noise = np.asarray(labels, dtype=np.int64) != -1
    return {
        "documents": len(labels),
        "communities": len(unique),
        "noise_count": int((~non_noise).sum()),
        "noise_fraction": float((~non_noise).mean()) if len(labels) else 0.0,
        "community_sizes": sizes_for_labels(labels),
        "modularity_retained_nodes": partition_modularity(graph, labels, resolution),
        "connectivity": connectivity_diagnostics(graph, labels),
        "ari_domains_weak_external": float(adjusted_rand_score(domains, labels)) if len(set(domains)) > 1 else None,
        "nmi_domains_weak_external": float(normalized_mutual_info_score(domains, labels)) if len(set(domains)) > 1 else None,
    }


def load_run(root: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
    run_dir = root / "artifacts" / "runs" / run_id
    clusters_path = run_dir / "clusters.csv"
    edges_path = run_dir / "graph_edges.csv"
    metrics_path = run_dir / "metrics.json"
    missing = [str(p) for p in (clusters_path, edges_path, metrics_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Для сравнения нужен полный артефакт контрольного запуска. Не найдены: " + "; ".join(missing)
        )

    clusters = pd.read_csv(clusters_path, encoding="utf-8-sig")
    edges = pd.read_csv(edges_path, encoding="utf-8-sig")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    required_clusters = {"document_id", "cluster_id", "title", "domain"}
    required_edges = {"source_id", "target_id", "similarity"}
    if not required_clusters.issubset(clusters.columns):
        raise ValueError(f"clusters.csv должен содержать столбцы: {sorted(required_clusters)}")
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"graph_edges.csv должен содержать столбцы: {sorted(required_edges)}")
    if clusters["document_id"].isna().any() or clusters["document_id"].duplicated().any():
        raise ValueError("clusters.csv содержит пустые или повторяющиеся document_id")
    if edges[list(required_edges)].isna().any().any():
        raise ValueError("graph_edges.csv содержит пустые идентификаторы или веса")
    if not str(metrics.get("algorithm", "")).lower().startswith("louvain"):
        raise ValueError(f"Ожидался контрольный запуск Louvain, получено: {metrics.get('algorithm')!r}")

    clusters["document_id"] = clusters["document_id"].astype(str)
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    edges["similarity"] = pd.to_numeric(edges["similarity"], errors="raise")
    if not np.isfinite(edges["similarity"].to_numpy(dtype=float)).all():
        raise ValueError("В graph_edges.csv обнаружены нечисловые/бесконечные веса")
    if (edges["similarity"] <= 0).any():
        raise ValueError("Для Leiden modularity все веса должны быть положительными")
    if (edges["source_id"] == edges["target_id"]).any():
        raise ValueError("В graph_edges.csv обнаружены self-loops; контрольный граф должен быть без них")

    node_ids = clusters["document_id"].tolist()
    node_set = set(node_ids)
    unknown = (set(edges["source_id"]) | set(edges["target_id"])) - node_set
    if unknown:
        raise ValueError(f"Рёбра ссылаются на {len(unknown)} неизвестных документов")
    normalized_pairs = [tuple(sorted((a, b))) for a, b in zip(edges["source_id"], edges["target_id"], strict=True)]
    if len(normalized_pairs) != len(set(normalized_pairs)):
        raise ValueError("В graph_edges.csv есть повторные неориентированные пары вершин")

    # Preserve the node order of the graph run, not CSV sort order from another export.
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_weighted_edges_from(
        (str(row.source_id), str(row.target_id), float(row.similarity))
        for row in edges.itertuples(index=False)
    )
    if graph.number_of_nodes() != int(metrics.get("graph_nodes", graph.number_of_nodes())):
        raise ValueError("Число вершин не совпадает с metrics.json")
    if graph.number_of_edges() != int(metrics.get("graph_edges", graph.number_of_edges())):
        raise ValueError("Число рёбер не совпадает с metrics.json")

    return clusters, edges, metrics, run_dir


def load_embeddings(root: Path, ids: list[str]) -> tuple[np.ndarray, dict[str, Any], Path, Path]:
    model_dir = root / DEFAULT_MODEL_DIR
    manifest_path = model_dir / "manifest.json"
    vectors_path = model_dir / "vectors.npy"
    if not manifest_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(
            f"Не найдены актуальные SPECTER-эмбеддинги в {model_dir}. Silhouette не будет подменён координатами x/y."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_ids = [str(x) for x in manifest.get("ids", [])]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("В manifest.json обнаружены повторные ID")
    if set(manifest_ids) != set(ids):
        missing = len(set(ids) - set(manifest_ids))
        extra = len(set(manifest_ids) - set(ids))
        raise ValueError(f"Manifest/graph ID mismatch: missing={missing}, extra={extra}. Не смешивайте разные корпуса.")
    matrix = np.load(vectors_path, mmap_mode="r")
    if matrix.ndim != 2 or len(matrix) != len(manifest_ids):
        raise ValueError("Форма vectors.npy не соответствует manifest.json")
    row_map = {doc_id: i for i, doc_id in enumerate(manifest_ids)}
    ordered = np.stack([np.asarray(matrix[row_map[doc_id]], dtype=np.float32) for doc_id in ids])
    norms = np.linalg.norm(ordered, axis=1)
    if not np.isfinite(ordered).all() or not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("Эмбеддинги имеют некорректные значения или не нормализованы")
    return ordered, manifest, manifest_path, vectors_path


def fair_silhouettes(vectors: np.ndarray, labels_a: list[int], labels_b: list[int], seed: int) -> dict[str, Any]:
    """Compare both partitions on the identical document subset in original cosine space."""
    core = (np.asarray(labels_a) != -1) & (np.asarray(labels_b) != -1)
    idx = np.flatnonzero(core)
    if len(idx) > 1500:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=1500, replace=False))
    if len(idx) < 3 or len(set(np.asarray(labels_a)[idx])) < 2 or len(set(np.asarray(labels_b)[idx])) < 2:
        return {"documents_evaluated": int(len(idx)), "louvain": None, "leiden": None, "reason": "insufficient_common_non_noise_partition"}
    selected = vectors[idx]
    distances = np.clip(1.0 - selected @ selected.T, 0.0, 2.0).astype(np.float32, copy=False)

    np.fill_diagonal(distances, 0.0)

    a = float(silhouette_score(distances, np.asarray(labels_a)[idx], metric="precomputed"))
    b = float(silhouette_score(distances, np.asarray(labels_b)[idx], metric="precomputed"))
    return {"documents_evaluated": int(len(idx)), "louvain": a, "leiden": b,
            "space": "original normalized SPECTER cosine distance", "same_document_sample": True}


def community_correspondence(
    ids: list[str], labels_a: list[int], labels_b: list[int],
    name_a: str = "louvain", name_b: str = "leiden",
) -> pd.DataFrame:
    a_ids = sorted(set(labels_a) - {-1})
    b_ids = sorted(set(labels_b) - {-1})
    rows = []
    for a in a_ids:
        for b in b_ids:
            left = {ids[i] for i, label in enumerate(labels_a) if label == a}
            right = {ids[i] for i, label in enumerate(labels_b) if label == b}
            overlap = len(left & right)
            union = len(left | right)
            rows.append({
                f"{name_a}_community": int(a), f"{name_b}_community": int(b),
                "intersection": overlap, "jaccard": overlap / union if union else 0.0,
                f"{name_a}_coverage": overlap / len(left) if left else 0.0,
                f"{name_b}_coverage": overlap / len(right) if right else 0.0,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Greedy one-to-one matching is sufficient for a readable correspondence table;
    # the complete pairwise table remains available for non-bijective split/merge analysis.
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches = []
    for row in frame.sort_values("jaccard", ascending=False).itertuples(index=False):
        a = int(getattr(row, f"{name_a}_community"))
        b = int(getattr(row, f"{name_b}_community"))
        if a not in used_a and b not in used_b:
            used_a.add(a); used_b.add(b)
            matches.append((a, b))
    match_map = {pair: rank + 1 for rank, pair in enumerate(matches)}
    frame["one_to_one_match_rank"] = [match_map.get((int(a), int(b))) for a, b in zip(frame[f"{name_a}_community"], frame[f"{name_b}_community"], strict=True)]
    return frame.sort_values(["one_to_one_match_rank", "jaccard"], ascending=[True, False], na_position="last")


def run_comparison(
    root: Path,
    run_id: str,
    resolution: float = 1.0,
    min_community_size: int = 5,
    seed: int = 42,
    n_iterations: int = -1,
) -> Path:
    try:
        import igraph as ig
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен igraph. Выполните: .\\.venv\\Scripts\\python.exe -m pip install igraph==1.0.0"
        ) from exc

    clusters, edges, baseline_metrics, run_dir = load_run(root, run_id)
    ids = clusters["document_id"].astype(str).tolist()
    base_labels = clusters["cluster_id"].astype(int).tolist()
    domains = clusters["domain"].fillna("unknown").astype(str).tolist()
    titles = clusters["title"].fillna("").astype(str).tolist()

    # Recreate exactly the saved undirected, weighted edge set. No kNN rebuild occurs here.
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(ids)}
    edge_pairs = [
        (id_to_index[str(row.source_id)], id_to_index[str(row.target_id)])
        for row in edges.itertuples(index=False)
    ]
    weights = edges["similarity"].astype(float).tolist()
    ig_graph = ig.Graph(n=len(ids), edges=edge_pairs, directed=False)
    ig_graph.es["weight"] = weights

    # Use Python's random.Random interface because igraph documents it as supported.
    ig.set_random_number_generator(random.Random(seed))
    started = time.perf_counter()
    leiden_result = ig_graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=float(resolution),
        beta=0.01,
        n_iterations=int(n_iterations),
    )
    leiden_elapsed = time.perf_counter() - started
    raw_leiden = canonicalize_membership([int(x) for x in leiden_result.membership])
    filtered_leiden = apply_minimum_size(raw_leiden, min_community_size)

    # Keep the saved Louvain labels untouched; their filtering already occurred in the source run.
    base_graph = nx.Graph()
    base_graph.add_nodes_from(ids)
    base_graph.add_weighted_edges_from(
        (str(row.source_id), str(row.target_id), float(row.similarity))
        for row in edges.itertuples(index=False)
    )
    graph_diag = {
        "nodes": base_graph.number_of_nodes(),
        "edges": base_graph.number_of_edges(),
        "components": nx.number_connected_components(base_graph),
        "isolates": len(list(nx.isolates(base_graph))),
        "density": float(nx.density(base_graph)),
        "average_degree": float(np.mean([degree for _, degree in base_graph.degree()])),
        "mean_weight": float(np.mean(weights)),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "edge_set_reused_from_run": run_id,
    }

    base_metrics = compute_partition_metrics(base_graph, base_labels, domains, resolution)
    raw_leiden_metrics = compute_partition_metrics(base_graph, raw_leiden, domains, resolution)
    filtered_leiden_metrics = compute_partition_metrics(base_graph, filtered_leiden, domains, resolution)

    # A fair assignment comparison includes the small-community-to-noise post-filter.
    agreement = {
        "ari_louvain_vs_leiden_filtered": float(adjusted_rand_score(base_labels, filtered_leiden)),
        "nmi_louvain_vs_leiden_filtered": float(normalized_mutual_info_score(base_labels, filtered_leiden)),
        "changed_documents": int(sum(a != b for a, b in zip(base_labels, filtered_leiden, strict=True))),
        "changed_fraction": float(np.mean(np.asarray(base_labels) != np.asarray(filtered_leiden))),
        "common_non_noise_documents": int(sum(a != -1 and b != -1 for a, b in zip(base_labels, filtered_leiden, strict=True))),
    }

    vectors, manifest, manifest_path, vectors_path = load_embeddings(root, ids)
    silhouettes = fair_silhouettes(vectors, base_labels, filtered_leiden, seed)

    output = root / "reports" / "community_detection_stage1" / run_id
    output.mkdir(parents=True, exist_ok=True)

    assignment_frame = clusters[["document_id", "title", "domain"]].copy()
    assignment_frame["louvain_cluster_id"] = base_labels
    assignment_frame["leiden_cluster_id_raw"] = raw_leiden
    assignment_frame["leiden_cluster_id"] = filtered_leiden
    assignment_frame.to_csv(output / "leiden_assignments.csv", index=False, encoding="utf-8-sig")

    changed = assignment_frame[assignment_frame["louvain_cluster_id"] != assignment_frame["leiden_cluster_id"]].copy()
    changed.to_csv(output / "changed_documents.csv", index=False, encoding="utf-8-sig")

    corr = community_correspondence(ids, base_labels, filtered_leiden)
    corr.to_csv(output / "community_correspondence.csv", index=False, encoding="utf-8-sig")

    size_rows = []
    for algorithm, metrics in (
        ("Louvain", base_metrics),
        ("Leiden_raw", raw_leiden_metrics),
        ("Leiden_min_size_filtered", filtered_leiden_metrics),
    ):
        for cluster_id, size in metrics["community_sizes"].items():
            size_rows.append({"partition": algorithm, "community_id": int(cluster_id), "documents": int(size)})
    pd.DataFrame(size_rows, columns=["partition", "community_id", "documents"]).to_csv(
        output / "community_sizes.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": "Stage 1 — Louvain vs Leiden on the identical saved weighted kNN graph",
        "status": "completed",
        "control_run_id": run_id,
        "parameters": {
            "graph_rebuilt": False,
            "objective_function": "modularity",
            "resolution": float(resolution),
            "beta": 0.01,
            "n_iterations": int(n_iterations),
            "minimum_community_size": int(min_community_size),
            "random_seed": int(seed),
            "rng": "random.Random(seed) passed to igraph.set_random_number_generator",
            "edge_weight_field": "similarity",
        },
        "software": {
            "igraph": getattr(ig, "__version__", "unknown"),
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "graph": graph_diag,
        "louvain_baseline_metrics_recomputed": base_metrics,
        "leiden_raw_metrics": raw_leiden_metrics,
        "leiden_min_size_filtered_metrics": filtered_leiden_metrics,
        "partition_agreement": agreement,
        "silhouette_original_specter_space_same_sample": silhouettes,
        "runtime_seconds": {"leiden": round(leiden_elapsed, 6)},
        "source_artifacts": {
            "clusters_csv_sha256": sha256_file(run_dir / "clusters.csv"),
            "graph_edges_csv_sha256": sha256_file(run_dir / "graph_edges.csv"),
            "metrics_json_sha256": sha256_file(run_dir / "metrics.json"),
            "embedding_manifest_sha256": sha256_file(manifest_path),
            "embedding_vectors_sha256": sha256_file(vectors_path),
            "embedding_model": manifest.get("model"),
            "embedding_dimension": manifest.get("dimension"),
        },
        "baseline_metrics_file": baseline_metrics,
        "interpretation_notes": [
            "Louvain assignments are read from the original run; Louvain is not rerun.",
            "Leiden receives the exact saved graph_edges.csv edge set and similarity weights.",
            "The primary comparison uses the same minimum-community-size post-filter as the source graph pipeline.",
            "The raw Leiden partition is also exported, so the effect of the small-community filter remains visible.",
            "Silhouette is computed in the original normalized SPECTER cosine space, never on x/y display coordinates.",
            "ARI/NMI versus domain labels are weak external diagnostics, not absolute topic ground truth.",
            "This is one fixed-seed run; multi-seed stability belongs to a later stage.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Этап 1. Louvain vs Leiden на одном графе",
        "",
        f"Контрольный запуск: `{run_id}`",
        "",
        "## Методика",
        "Leiden запущен на точном наборе рёбер и весов из `graph_edges.csv`. Граф не перестраивался. Исходные назначения Louvain взяты из `clusters.csv`; алгоритм Louvain повторно не запускался.",
        "",
        f"- Objective: modularity; resolution = {resolution}",
        f"- Leiden beta = 0.01; n_iterations = {n_iterations}; seed = {seed}",
        f"- Минимальный размер сообщества после обнаружения: {min_community_size}",
        f"- Вершины: {graph_diag['nodes']}; рёбра: {graph_diag['edges']}; компоненты: {graph_diag['components']}; изолированные: {graph_diag['isolates']}",
        "",
        "## Сравнение",
        "",
        "| Показатель | Louvain | Leiden после фильтра |",
        "|---|---:|---:|",
        f"| Сообщества | {base_metrics['communities']} | {filtered_leiden_metrics['communities']} |",
        f"| Шум | {base_metrics['noise_count']} | {filtered_leiden_metrics['noise_count']} |",
        f"| Модулярность на оставшихся вершинах | {fmt(base_metrics['modularity_retained_nodes'])} | {fmt(filtered_leiden_metrics['modularity_retained_nodes'])} |",
        f"| Связные сообщества | {base_metrics['connectivity']['community_count'] - base_metrics['connectivity']['disconnected_community_count']} / {base_metrics['communities']} | {filtered_leiden_metrics['connectivity']['community_count'] - filtered_leiden_metrics['connectivity']['disconnected_community_count']} / {filtered_leiden_metrics['communities']} |",
        f"| ARI относительно domain | {fmt(base_metrics['ari_domains_weak_external'])} | {fmt(filtered_leiden_metrics['ari_domains_weak_external'])} |",
        f"| NMI относительно domain | {fmt(base_metrics['nmi_domains_weak_external'])} | {fmt(filtered_leiden_metrics['nmi_domains_weak_external'])} |",
        "",
        f"ARI Louvain–Leiden: **{agreement['ari_louvain_vs_leiden_filtered']:.4f}**; NMI: **{agreement['nmi_louvain_vs_leiden_filtered']:.4f}**; изменили группу: **{agreement['changed_documents']}** документов ({agreement['changed_fraction']:.1%}).",
        "",
        f"Silhouette на одном и том же наборе документов в исходном SPECTER-пространстве: Louvain = **{fmt(silhouettes['louvain'])}**, Leiden = **{fmt(silhouettes['leiden'])}** (n = {silhouettes['documents_evaluated']}).",
        "",
        "## Файлы",
        "- `summary.json` — полный отчёт с параметрами, версиями и SHA-256 входных артефактов.",
        "- `leiden_assignments.csv` — исходные и новые назначения по каждой публикации.",
        "- `changed_documents.csv` — публикации, изменившие сообщество или ставшие шумом.",
        "- `community_correspondence.csv` — попарные пересечения сообществ для анализа split/merge.",
        "- `community_sizes.csv` — размеры исходных и новых групп.",
        "",
        "## Ограничения",
        "Это один запуск Leiden с фиксированным seed, а не оценка устойчивости. Категории `domain` являются слабыми внешними метками. Более высокая модульность или ARI сами по себе не доказывают лучшую тематическую интерпретируемость.",
    ]
    (output / "REPORT_STAGE1_RU.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output


def fmt(value: Any) -> str:
    return "н/д" if value is None else f"{float(value):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: compare saved Louvain partition with Leiden on the same graph")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-community-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iterations", type=int, default=-1, help="-1 means iterate until a stable quality iteration")
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / "artifacts").exists():
        parser.error(f"Не найдена папка artifacts в ProjectRoot: {root}")
    if args.resolution <= 0 or args.min_community_size < 1:
        parser.error("resolution должен быть > 0, min-community-size должен быть >= 1")

    try:
        out = run_comparison(root, args.run_id, args.resolution, args.min_community_size, args.seed, args.n_iterations)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Stage 1 comparison completed: {out}")
    print(f"Report: {out / 'REPORT_STAGE1_RU.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
