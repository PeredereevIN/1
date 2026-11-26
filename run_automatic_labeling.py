from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from cluster_labeler import LabelerConfig, find_latest_assignments, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Automatically name HDBSCAN/DBSCAN clusters from document text (no LLM, no per-cluster rules)."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--assignments", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-chars", type=int, default=22000)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--min-group-prevalence", type=float, default=0.04)
    parser.add_argument("--include-noise", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    assignments = args.assignments or find_latest_assignments(root)
    if not assignments.is_absolute():
        assignments = (root / assignments).resolve()
    processed_dir = args.processed_dir or (root / "data" / "processed")
    if not processed_dir.is_absolute():
        processed_dir = (root / processed_dir).resolve()
    output_dir = args.output_dir or (
        root / "reports" / "automatic_cluster_labels"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    config = LabelerConfig(
        max_chars_per_document=args.max_chars,
        max_features=args.max_features,
        max_df=args.max_df,
        min_group_prevalence=args.min_group_prevalence,
        skip_noise=not args.include_noise,
    )

    print("=" * 84)
    print("AUTOMATIC CLUSTER LABELING ENGINE v8.0")
    print("Analytical method | No LLM | No group-ID-specific naming rules")
    print("PDF cleaning | contiguous source phrases only | alias normalization | prevalence + contrast | redundancy control | mixed-topic fallback")
    print("=" * 84)
    print("Assignments:", assignments)
    print("Processed texts:", processed_dir)
    print("Output:", output_dir)
    print("\nStarting automatic labeling...\n")

    outputs = run_pipeline(assignments, processed_dir, output_dir, config)

    import pandas as pd
    labels = pd.read_csv(outputs["labels"], encoding="utf-8-sig")
    for _, row in labels.iterrows():
        print(f"\n[{row['group_id']}] {row['automatic_label']}")
        print(f"  Documents: {row['document_count']}")
        print(f"  Evidence: {row['label_quality_flag']} ({row['label_evidence_score']})")
        print(f"  Key phrases: {row['key_phrases']}")
        print(f"  Title support: {row.get('leading_phrase_title_support_docs', 0)} docs")
        print(f"  Description: {row['automatic_description']}")

    print("\n" + "=" * 84)
    print("AUTOMATIC LABELING FINISHED")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
