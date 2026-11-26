# -*- coding: utf-8 -*-
"""Streamlit explorer for dynamically discovered scientific-paper topics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Scientific Intelligence Lab · Тематические группы", page_icon="🧭", layout="wide")

ROOT = Path(os.environ.get("SIL_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
REPORTS = ROOT / "reports" / "auto_topic_discovery"


def discover_runs() -> list[Path]:
    if not REPORTS.exists():
        return []
    paths = list(REPORTS.glob("*/*/topic_profiles.csv"))
    return sorted([p.parent for p in paths], key=lambda p: p.name, reverse=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_overrides(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in payload.items()}
        except Exception:
            return {}
    return {}


def save_overrides(path: Path, values: dict[str, str]) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_value(summary: dict[str, Any], key: str, default: str = "—") -> Any:
    value = summary.get(key, default)
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


st.title("🧭 Тематическая карта научных публикаций")
st.caption("Leiden автоматически определяет число сообществ. Названия и описания формируются локальной instruction-моделью по ключевым фразам и репрезентативным публикациям; назначения Leiden не изменяются.")

runs = discover_runs()
if not runs:
    st.error("Не найдены результаты автоматического тематического обнаружения.")
    st.code("python auto_topic_discovery.py --root . --run-id 20261120_015622_graph_b63e68", language="powershell")
    st.stop()

run_labels = [f"{p.parent.name} · {p.name}" for p in runs]
selected_label = st.sidebar.selectbox("Запуск анализа", run_labels, index=0)
out_dir = runs[run_labels.index(selected_label)]

try:
    profiles = read_csv(out_dir / "topic_profiles.csv")
    assignments = read_csv(out_dir / "document_assignments.csv")
    projection = read_csv(out_dir / "projection_2d.csv")
    sweep = read_csv(out_dir / "resolution_sweep.csv")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"Не удалось прочитать результат выбранного запуска: {exc}")
    st.stop()

if profiles.empty or assignments.empty or projection.empty:
    st.error("В выбранном запуске отсутствуют данные для визуализации.")
    st.stop()

profiles["cluster_id"] = pd.to_numeric(profiles["cluster_id"], errors="coerce").astype(int)
assignments["cluster_id"] = pd.to_numeric(assignments["cluster_id"], errors="coerce").astype(int)
projection["cluster_id"] = pd.to_numeric(projection["cluster_id"], errors="coerce").astype(int)
overrides_path = out_dir / "topic_label_overrides.json"
overrides = load_overrides(overrides_path)
profiles["display_name"] = profiles.apply(
    lambda r: overrides.get(str(int(r.cluster_id)), str(r.get("semantic_name_ru") or r.get("semantic_name") or f"Тема {int(r.cluster_id)+1}")),
    axis=1,
)
name_map = dict(zip(profiles.cluster_id.astype(int), profiles.display_name.astype(str)))
projection = projection.merge(profiles[["cluster_id", "display_name"]], on="cluster_id", how="left", suffixes=("", "_profile"))
projection["display_name"] = projection["display_name"].fillna(projection["semantic_name_ru"]).fillna(projection["semantic_name"])
assignments["display_name"] = assignments["cluster_id"].map(name_map)

st.sidebar.markdown("---")
st.sidebar.caption(f"Результат: `{out_dir.relative_to(ROOT)}`")
st.sidebar.caption(f"Входной граф: `{summary.get('run_id', '—')}`")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Публикации", f"{int(summary.get('documents', len(assignments))):,}")
c2.metric("Обнаруженные темы", int(summary.get("detected_communities", len(profiles))))
c3.metric("Resolution", f"{float(summary.get('chosen_resolution', 0)):.3f}")
c4.metric("Устойчивость ARI", f"{float(summary.get('selected_candidate_metrics', {}).get('stability_ari_mean', 0)):.3f}")
c5.metric("Silhouette", f"{float(summary.get('selected_candidate_metrics', {}).get('silhouette_mean', 0)):.3f}")

with st.expander("Методика автоматического выбора числа тем и именования", expanded=False):
    st.write("Число сообществ не задаётся вручную. Для набора значений resolution выполняются повторные запуски Leiden с разными seed. Выбирается вариант по заранее записанной составной оценке устойчивости (ARI между seed), cosine silhouette в исходном SPECTER-пространстве, модулярности и штрафов за вырожденные разбиения.")
    labeler_info = summary.get("topic_labeler", {})
    st.write(f"Модель семантического именования: {labeler_info.get('model') or 'эвристика ключевых фраз'}")
    st.write(f"Провайдер: {labeler_info.get('provider', '—')}; digest модели: {labeler_info.get('model_digest', '—')}")
    st.caption("Модель описывает уже найденные сообщества. Она не формирует эмбеддинги, не меняет рёбра графа и не переназначает публикации.")
    st.code(summary.get("selection_formula", "Формула отсутствует в summary.json"))
    st.caption(summary.get("selection_note", ""))
    st.write(f"Тексты публикаций: с аннотациями/полным текстом — {summary.get('documents_with_abstract_or_text', 0)}; только заголовки — {summary.get('documents_title_only', 0)}.")
    st.write(f"YAKE доступен: {summary.get('yake_available', False)}. RAKE: {summary.get('rake_available', False)}. Проекция: {summary.get('projection_method', '—')}.")

st.subheader("Интерактивная карта публикаций")
filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
cluster_choices = ["Все темы"] + [f"{int(r.cluster_id)} · {r.display_name} ({int(r.documents)})" for r in profiles.sort_values("documents", ascending=False).itertuples()]
with filter_col1:
    selected_topic = st.selectbox("Фокус на теме", cluster_choices)
with filter_col2:
    domain_values = sorted(x for x in assignments.domain.dropna().astype(str).unique() if x and x.lower() != "nan") if "domain" in assignments else []
    selected_domain = st.selectbox("Домен", ["Все домены"] + domain_values)
with filter_col3:
    color_options = ["Тематические группы"] + (["Домены"] if domain_values else [])
    color_mode = st.selectbox("Раскраска", color_options)

map_df = projection.copy()
# Attach domain metadata before applying filters.
if "domain" not in map_df.columns and "domain" in assignments.columns:
    map_df = map_df.merge(assignments[["document_id", "domain"]], on="document_id", how="left")
if selected_topic != "Все темы":
    selected_id = int(selected_topic.split(" · ", 1)[0])
    map_df = map_df[map_df.cluster_id == selected_id]
if selected_domain != "Все домены" and "domain" in map_df.columns:
    map_df = map_df[map_df.domain.astype(str) == selected_domain]
if color_mode == "Домены" and "domain" in map_df.columns:
    color_col = "domain"
else:
    color_col = "display_name"
if not map_df.empty:
    hover_cols = [c for c in ["document_id", "title", "domain", "cluster_id"] if c in map_df.columns]
    fig = px.scatter(
        map_df, x="x", y="y", color=color_col, hover_name="title" if "title" in map_df else None,
        hover_data=hover_cols, opacity=0.78,
        labels={"x": "Ось 1", "y": "Ось 2", "display_name": "Тематическая группа", "domain": "Домен"},
        title=f"Публикаций на карте: {len(map_df):,}",
    )
    fig.update_traces(marker={"size": 7})
    fig.update_layout(height=690, legend_title_text="Группы", margin={"l": 10, "r": 10, "t": 55, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Для выбранных фильтров публикаций нет.")
st.caption("Координаты UMAP/PCA используются только для визуального расположения точек. Метрики качества рассчитаны в исходном SPECTER-пространстве и на графе, а не по координатам карты.")

st.subheader("Каталог автоматически обнаруженных тем")
view_profiles = profiles.sort_values("documents", ascending=False).copy()
cols = st.columns(3)
for pos, row in enumerate(view_profiles.itertuples(index=False)):
    with cols[pos % 3]:
        with st.container(border=True):
            st.markdown(f"**{row.display_name}**")
            st.caption(f"Сообщество {int(row.cluster_id)} · {int(row.documents):,} публикаций · {float(row.share_percent):.1f}% корпуса")
            description = getattr(row, "semantic_description_ru", "")
            if pd.notna(description) and str(description).strip():
                st.write(str(description))
            scope_note = getattr(row, "semantic_scope_note_ru", "")
            if pd.notna(scope_note) and str(scope_note).strip():
                st.warning(str(scope_note))
            st.write(str(row.keywords)[:420] if pd.notna(row.keywords) else "Ключевые термины не извлечены")
            if pd.notna(row.dominant_domains) and row.dominant_domains:
                st.caption(f"Домены: {row.dominant_domains}")

st.subheader("Подробности выбранной темы")
profile_options = [int(x) for x in profiles.cluster_id.tolist()]
selected_cluster = st.selectbox(
    "Сообщество для изучения",
    profile_options,
    format_func=lambda cid: f"{cid} · {name_map.get(cid, f'Тема {cid+1}')} · {int(profiles.loc[profiles.cluster_id == cid, 'documents'].iloc[0])} публикаций",
)
selected_profile = profiles.loc[profiles.cluster_id == selected_cluster].iloc[0]
selected_docs = assignments[assignments.cluster_id == selected_cluster].copy()
left, right = st.columns([1, 2])
with left:
    st.metric("Публикаций в теме", int(selected_profile.documents))
    if "semantic_description_ru" in selected_profile and pd.notna(selected_profile.semantic_description_ru):
        st.write("**Описание**")
        st.write(str(selected_profile.semantic_description_ru))
    if "semantic_scope_note_ru" in selected_profile and pd.notna(selected_profile.semantic_scope_note_ru) and str(selected_profile.semantic_scope_note_ru).strip():
        st.warning(str(selected_profile.semantic_scope_note_ru))
    st.caption(f"Способ именования: {selected_profile.get('semantic_label_method', '—')}")
    st.write("**Ключевые фразы**")
    st.write(selected_profile.keywords or "—")
    st.write("**YAKE**")
    st.write(selected_profile.keywords_yake or "YAKE не вернул термины")
    st.write("**RAKE**")
    st.write(selected_profile.keywords_rake or "—")
    st.write("**Контрастный TF-IDF**")
    st.write(selected_profile.keywords_tfidf_contrast or "—")
    st.write("**Среднее сходство с центроидом**")
    st.write(f"{float(selected_profile.mean_cosine_to_centroid):.4f}")
with right:
    st.write("**Наиболее репрезентативные публикации**")
    for i, paper in enumerate(selected_docs.sort_values("cosine_to_own_centroid", ascending=False).head(12).itertuples(), start=1):
        st.markdown(f"{i}. **{paper.title}**")
        st.caption(f"{paper.document_id} · {paper.domain} · cosine={float(paper.cosine_to_own_centroid):.3f}")

st.subheader("Редактирование семантического названия")
with st.form("rename_topic"):
    current_name = overrides.get(str(selected_cluster), str(selected_profile.semantic_name_ru or selected_profile.semantic_name))
    new_name = st.text_input("Название темы", value=current_name, max_chars=120)
    submitted = st.form_submit_button("Сохранить название")
    if submitted:
        overrides[str(selected_cluster)] = new_name.strip() or current_name
        save_overrides(overrides_path, overrides)
        st.success("Название сохранено для этого запуска анализа.")
        st.rerun()

with st.expander("Пограничные публикации"):
    boundary_path = out_dir / "boundary_documents.csv"
    if boundary_path.exists():
        boundary = read_csv(boundary_path)
        if "cluster_id" in boundary:
            boundary = boundary[boundary.cluster_id == selected_cluster]
        st.dataframe(boundary.head(50), use_container_width=True, hide_index=True)

with st.expander("Автоматический выбор resolution"):
    display_sweep = sweep.sort_values("selection_score", ascending=False)
    st.dataframe(display_sweep, use_container_width=True, hide_index=True)
    if {"resolution", "selection_score", "stability_ari_mean", "silhouette_mean", "modularity_mean"}.issubset(sweep.columns):
        chart_df = sweep.melt(id_vars="resolution", value_vars=["selection_score", "stability_ari_mean", "silhouette_mean"], var_name="metric", value_name="score")
        fig_sweep = px.line(chart_df, x="resolution", y="score", color="metric", markers=True, title="Поведение критериев при изменении resolution")
        fig_sweep.update_layout(height=360)
        st.plotly_chart(fig_sweep, use_container_width=True)

st.subheader("Экспорт данных")
d1, d2, d3 = st.columns(3)
d1.download_button("Скачать назначения CSV", assignments.to_csv(index=False).encode("utf-8-sig"), "document_assignments.csv", "text/csv", use_container_width=True)
d2.download_button("Скачать профили тем CSV", profiles.to_csv(index=False).encode("utf-8-sig"), "topic_profiles.csv", "text/csv", use_container_width=True)
d3.download_button("Скачать summary JSON", json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"), "summary.json", "application/json", use_container_width=True)

st.caption("Автоматические названия являются гипотезами, сформированными по ключевым фразам. Проверяйте тематическую целостность групп по репрезентативным публикациям, а не только по названию.")
