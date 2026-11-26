#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-label an existing Leiden result using a local Ollama model, without reclustering."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auto_topic_discovery import (
    DEFAULT_EMBEDDING_DIR,
    OllamaTopicLabeler,
    build_profiles,
    discover_texts,
    manifest_ids,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"Ожидался JSON-объект: {path}")
    return value


def resolve_source_dir(root: Path, run_id: str, source_result: str) -> Path:
    base = root / "reports" / "auto_topic_discovery" / run_id
    if source_result.lower() == "latest":
        latest_path = base / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"Не найден указатель последнего запуска: {latest_path}")
        latest = read_json(latest_path)
        folder = str(latest.get("output_dir", "")).strip()
        if not folder:
            raise RuntimeError(f"В {latest_path} отсутствует поле output_dir.")
    else:
        folder = source_result
    source = (base / folder).resolve()
    if not source.is_dir() or base.resolve() not in source.parents:
        raise FileNotFoundError(f"Каталог результата не найден или находится вне run-папки: {source}")
    for required in ("topic_profiles.csv", "document_assignments.csv", "projection_2d.csv", "summary.json"):
        if not (source / required).is_file():
            raise FileNotFoundError(f"В исходном результате отсутствует {required}: {source}")
    return source


def update_named_columns(frame: pd.DataFrame, name_by_cluster: dict[int, dict[str, str]]) -> pd.DataFrame:
    result = frame.copy()
    if "cluster_id" not in result.columns:
        return result
    ids = pd.to_numeric(result["cluster_id"], errors="raise").astype(int)
    mappings = {
        "semantic_name": "name_en",
        "semantic_name_en": "name_en",
        "semantic_name_ru": "name_ru",
        "semantic_description_ru": "description_ru",
        "semantic_scope_note_ru": "scope_note_ru",
        "semantic_label_method": "label_method",
        "semantic_label_status": "label_status",
    }
    for column, key in mappings.items():
        result[column] = [name_by_cluster.get(int(cid), {}).get(key, "") for cid in ids]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate LLM names/descriptions for an existing Leiden result without rerunning Leiden or UMAP."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", default="20261120_015622_graph_b63e68")
    parser.add_argument("--source-result", default="latest")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    root = Path(args.root).resolve()
    run_dir = root / "artifacts" / "runs" / args.run_id
    clusters_path = run_dir / "clusters.csv"
    emb_dir = root / DEFAULT_EMBEDDING_DIR
    manifest_path = emb_dir / "manifest.json"
    vectors_path = emb_dir / "vectors.npy"
    for path in (clusters_path, manifest_path, vectors_path):
        if not path.is_file():
            raise FileNotFoundError(f"Не найден обязательный входной файл: {path}")

    source_dir = resolve_source_dir(root, args.run_id, args.source_result)
    source_profiles = pd.read_csv(source_dir / "topic_profiles.csv", encoding="utf-8-sig")
    source_assignments = pd.read_csv(
        source_dir / "document_assignments.csv", encoding="utf-8-sig", dtype={"document_id": str}
    )
    if not {"cluster_id", "document_id"}.issubset(source_assignments.columns):
        raise RuntimeError("document_assignments.csv должен содержать cluster_id и document_id.")
    if source_assignments.document_id.isna().any() or source_assignments.document_id.duplicated().any():
        raise RuntimeError("В document_assignments.csv есть пустые или повторные document_id.")

    print(f"Исходный результат: {source_dir}")
    print("Проверяю локальную языковую модель...")
    labeler = OllamaTopicLabeler(args.ollama_url, args.ollama_model, args.ollama_timeout, args.seed)
    labeler_metadata = labeler.verify()
    labeler_metadata["output_language"] = "ru"
    print(f"Модель доступна: {args.ollama_model}")

    print("Загружаю исходный корпус и SPECTER-векторы...")
    docs = pd.read_csv(clusters_path, encoding="utf-8-sig", dtype={"document_id": str})
    docs["document_id"] = docs["document_id"].astype(str)
    if docs.document_id.isna().any() or docs.document_id.duplicated().any():
        raise RuntimeError("clusters.csv содержит пустые или повторные document_id.")
    ids = docs.document_id.tolist()
    assignment_ids = source_assignments.document_id.astype(str).tolist()
    if len(ids) != len(assignment_ids) or set(ids) != set(assignment_ids):
        raise RuntimeError(
            "Набор публикаций в исходных назначениях не совпадает с clusters.csv: "
            f"corpus={len(ids)}, assignments={len(assignment_ids)}."
        )
    labels_by_id = source_assignments.set_index("document_id")["cluster_id"]
    labels = pd.to_numeric(labels_by_id.loc[ids], errors="raise").to_numpy(dtype=np.int32)

    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    vector_ids = manifest_ids(manifest)
    if len(vector_ids) != len(set(vector_ids)) or set(vector_ids) != set(ids):
        raise RuntimeError("Manifest и clusters.csv не содержат одинаковый набор уникальных document_id.")
    matrix = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(vector_ids):
        raise RuntimeError(f"Неверная форма vectors.npy: {matrix.shape}")
    row_map = {doc_id: index for index, doc_id in enumerate(vector_ids)}
    X = np.asarray(matrix[[row_map[doc_id] for doc_id in ids]], dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.isfinite(X).all() or np.any(norms <= 1e-12):
        raise RuntimeError("SPECTER vectors содержат NaN/Inf или нулевые векторы.")
    U = X / norms

    docs["title"] = docs["title"].fillna("").astype(str)
    docs["domain"] = docs["domain"].fillna("").astype(str) if "domain" in docs.columns else ""
    titles = docs["title"].tolist()
    domains = docs["domain"].tolist() if "domain" in docs.columns else [""] * len(docs)
    texts, text_sources, used_text_files = discover_texts(root, docs, ids)
    if len(texts) != len(ids) or len(text_sources) != len(ids):
        raise RuntimeError("Нарушено выравнивание текстов и идентификаторов публикаций.")

    source_cluster_ids = set(pd.to_numeric(source_profiles["cluster_id"], errors="raise").astype(int))
    assignment_cluster_ids = set(np.unique(labels).astype(int).tolist())
    if source_cluster_ids != assignment_cluster_ids:
        raise RuntimeError(
            f"Набор сообществ в профилях и назначениях различается: "
            f"profiles={sorted(source_cluster_ids)}, assignments={sorted(assignment_cluster_ids)}"
        )

    print(f"Групп для именования: {len(source_cluster_ids)}. Перекластеризация не выполняется.")
    new_profiles, _centroids, _representatives, label_audit = build_profiles(
        labels, U, titles, domains, texts, ids, "en",
        labeler=labeler, text_sources=text_sources,
    )

    semantic_columns = [
        "cluster_id", "semantic_name", "semantic_name_en", "semantic_name_ru",
        "semantic_description_ru", "semantic_scope_note_ru",
        "semantic_label_method", "semantic_label_status",
    ]
    updated_profiles = source_profiles.drop(
        columns=[c for c in semantic_columns if c != "cluster_id" and c in source_profiles.columns],
        errors="ignore",
    ).merge(new_profiles[semantic_columns], on="cluster_id", how="left", validate="one_to_one")
    if updated_profiles["semantic_name_ru"].isna().any():
        raise RuntimeError("Не удалось сформировать название для каждого сообщества.")

    name_by_cluster: dict[int, dict[str, str]] = {}
    for row in new_profiles.itertuples(index=False):
        name_by_cluster[int(row.cluster_id)] = {
            "name_en": str(row.semantic_name_en),
            "name_ru": str(row.semantic_name_ru),
            "description_ru": str(row.semantic_description_ru),
            "scope_note_ru": str(row.semantic_scope_note_ru),
            "label_method": str(row.semantic_label_method),
            "label_status": str(row.semantic_label_status),
        }

    updated_assignments = update_named_columns(source_assignments, name_by_cluster)
    updated_projection = update_named_columns(
        pd.read_csv(source_dir / "projection_2d.csv", encoding="utf-8-sig"),
        name_by_cluster,
    )
    boundary_path = source_dir / "boundary_documents.csv"
    updated_boundary = (
        update_named_columns(pd.read_csv(boundary_path, encoding="utf-8-sig"), name_by_cluster)
        if boundary_path.is_file() else None
    )

    # Use the source folder as a lexical prefix so Streamlit's reverse name sort
    # reliably lists the relabeled snapshot before the original, regardless of timezone.
    new_name = f"{source_dir.name}_llm_labels"
    output_dir = source_dir.parent / new_name
    if output_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{source_dir.name}_llm_labels_{stamp}"
        output_dir = source_dir.parent / new_name
    shutil.copytree(source_dir, output_dir)
    updated_profiles.to_csv(output_dir / "topic_profiles.csv", index=False, encoding="utf-8-sig")
    updated_assignments.to_csv(output_dir / "document_assignments.csv", index=False, encoding="utf-8-sig")
    updated_projection.to_csv(output_dir / "projection_2d.csv", index=False, encoding="utf-8-sig")
    if updated_boundary is not None:
        updated_boundary.to_csv(output_dir / "boundary_documents.csv", index=False, encoding="utf-8-sig")

    (output_dir / "topic_label_generation.json").write_text(
        json.dumps({
            "labeler": labeler_metadata,
            "source_result": str(source_dir.relative_to(root)),
            "label_records": label_audit,
            "note": "Only semantic metadata was regenerated. Existing community assignments, resolution sweep, graph, embeddings and 2D projection coordinates were preserved.",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = read_json(output_dir / "summary.json")
    summary["topic_labeler"] = labeler_metadata
    summary["topic_label_generation_file"] = "topic_label_generation.json"
    summary["topic_label_source_result"] = str(source_dir.relative_to(root))
    summary["topic_labeling_only"] = True
    summary["documents_with_abstract_or_text_for_labeling"] = int(sum(x == "abstract_or_text" for x in text_sources))
    summary["documents_title_only_for_labeling"] = int(sum(x == "title_only" for x in text_sources))
    summary["labeling_text_source_files"] = used_text_files
    summary["labeling_source_sha256"] = {
        "topic_profiles.csv": sha256_file(source_dir / "topic_profiles.csv"),
        "document_assignments.csv": sha256_file(source_dir / "document_assignments.csv"),
        "projection_2d.csv": sha256_file(source_dir / "projection_2d.csv"),
    }
    summary["topic_labeling_runtime_seconds"] = round(time.perf_counter() - started, 3)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Повторное семантическое именование существующих сообществ",
        "",
        f"- Исходный результат: `{source_dir.relative_to(root)}`",
        f"- Контрольный граф: `{args.run_id}`",
        f"- Количество сообществ: **{len(source_cluster_ids)}**",
        f"- Публикаций: **{len(ids)}**",
        f"- Модель: **{args.ollama_model}**",
        f"- Digest: `{labeler_metadata.get('model_digest', '')}`",
        f"- Аннотации/текст найдены для: **{summary['documents_with_abstract_or_text_for_labeling']}** публикаций",
        "",
        "## Семантические названия",
        "",
        "| ID | Название | Публикаций |",
        "|---:|---|---:|",
    ]
    doc_counts = source_assignments.groupby("cluster_id").size().to_dict()
    for row in updated_profiles.sort_values("documents", ascending=False).itertuples(index=False):
        report_lines.append(
            f"| {int(row.cluster_id)} | {row.semantic_name_ru} | "
            f"{int(doc_counts.get(int(row.cluster_id), 0))} |"
        )
    report_lines.extend([
        "",
        "## Что было изменено",
        "",
        "Перегенерированы только русские/английские названия, описания и примечания о неоднородности. "
        "Назначения публикаций, выбранный resolution, resolution_sweep.csv, исходный граф, "
        "SPECTER-векторы и координаты UMAP/PCA не перестраивались.",
        "",
        "Evidence и ответы модели сохранены в `topic_label_generation.json`. "
        "Исходный каталог результата не изменён.",
    ])
    (output_dir / "REPORT_LLM_LABELS_RU.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    latest_path = source_dir.parent / "latest.json"
    latest_path.write_text(
        json.dumps({"output_dir": new_name, "path": str(output_dir.relative_to(root))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nГОТОВО: {output_dir}")
    print("Перекластеризация: НЕТ")
    print(f"Сообществ с новыми семантическими метками: {len(source_cluster_ids)}")
    print(f"Публикаций: {len(ids)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
