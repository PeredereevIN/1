"""Minimal example for projects that already have cluster assignments."""
import pandas as pd
from cluster_labeler import AutomaticClusterLabeler

# Replace `assignments` with your post-HDBSCAN DataFrame.
# Required columns: document_id, group_id, title, text.
# The text field should contain extracted article text or at least abstract.

def add_automatic_labels(assignments: pd.DataFrame):
    labeled = AutomaticClusterLabeler().fit_predict(assignments)
    return {
        "cluster_cards": labeled["labels"],
        "document_table": labeled["assignments"],
        "label_evidence": labeled["candidates"],
    }
