from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from topic_understanding import TopicConfig, run_topic_understanding


def find_latest_assignments(root: Path) -> Path:
    paths = list((root / "artifacts" / "runs").glob("*/clusters.csv"))
    if not paths:
        paths = list((root / "reports").glob("**/cluster_assignments.csv"))
    if not paths:
        raise FileNotFoundError("No cluster assignments found. Run HDBSCAN/graph-cluster first.")
    return max(paths, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster Topic Understanding: title/abstract-based analytical cluster interpretation (no LLM).")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--assignments", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-primary-prevalence", type=float, default=0.12)
    parser.add_argument("--min-secondary-prevalence", type=float, default=0.08)
    args = parser.parse_args()

    root = args.project_root.resolve()
    assignments = args.assignments or find_latest_assignments(root)
    if not assignments.is_absolute():
        assignments = (root / assignments).resolve()
    processed = args.processed_dir or (root / "data" / "processed")
    if not processed.is_absolute():
        processed = (root / processed).resolve()
    output = args.output_dir or (assignments.parent / "semantic_profiles")
    if not output.is_absolute():
        output = (root / output).resolve()

    # Reuse canonical arXiv abstracts stored during ingestion. The analysis
    # falls back to the extracted PDF text only when no source abstract exists.
    sys.path.insert(0, str(root / "src"))
    abstract_by_id = {}
    try:
        from sciintel import db
        db.init_db()
        abstract_by_id = {
            str(row["id"]): str(row.get("abstract") or "")
            for row in db.rows("SELECT id, abstract FROM documents")
        }
    except Exception as exc:
        print(f"WARNING: canonical abstracts are unavailable; PDF text fallback will be used: {exc}")

    config = TopicConfig(
        min_primary_prevalence=args.min_primary_prevalence,
        min_secondary_prevalence=args.min_secondary_prevalence,
    )
    print("=" * 92)
    print("CLUSTER TOPIC UNDERSTANDING v1.0")
    print("Title + abstract analysis | no LLM | no group-ID-specific labels")
    print("Topic profiles | representative papers | coherence | internal subtopic diagnostics")
    print("=" * 92)
    print("Assignments:", assignments)
    print("Processed texts:", processed)
    print("Output:", output)
    print("\nAnalyzing publications...\n")
    files = run_topic_understanding(assignments, processed, output, config, abstract_by_id=abstract_by_id)
    profiles = pd.read_csv(files["profiles"], encoding="utf-8-sig")
    for _, r in profiles.iterrows():
        print(f"\n[{r['group_id']}] {r['automatic_label']}")
        print(f"  Documents: {r['document_count']}")
        print(f"  Status: {r['topic_profile_status']} (heuristic={r['topic_profile_score']})")
        print(f"  Aspects: {r['topic_aspects']}")
        print(f"  Abstract modes: {r['abstract_source_distribution']}")
        print(f"  Coherence: centroid={r['mean_title_abstract_centroid_similarity']}, pairwise={r['mean_pairwise_title_abstract_similarity']}")
        print(f"  Internal structure: {r['internal_subtopic_diagnostic']}")
        print(f"  Representatives: {r['representative_titles']}")
    print("\n" + "=" * 92)
    print("TOPIC UNDERSTANDING FINISHED")
    for k, v in files.items():
        print(f"{k}: {v}")
    print("=" * 92)


if __name__ == "__main__":
    main()
