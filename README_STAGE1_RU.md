# Этап 1. Сравнение Louvain и Leiden на одном графе

## Важное ограничение текущей поставки

В переданных в беседу файлах присутствуют назначения документов и 2D-координаты, но нет исходного `graph_edges.csv` контрольного запуска и актуальных SPECTER-векторов на 1 870 документов. Старый архив содержит другую матрицу (810 документов) и намеренно не используется. Поэтому этот пакет запускает эксперимент **в локальной рабочей копии пользователя**, где сохранены исходные артефакты запуска `20261120_015622_graph_b63e68`.

Скрипт не восстанавливает граф из координат x/y и не пересчитывает kNN. Он загружает точные рёбра и веса исходного запуска.

## Что сравнивается

- Louvain: сохранённые метки из `artifacts/runs/<run_id>/clusters.csv`.
- Leiden: `igraph.Graph.community_leiden(objective_function="modularity")`.
- Входной граф: `graph_edges.csv`, без изменений.
- Вес ребра: `similarity`.
- Resolution: 1.0.
- Минимальный размер: 5, как в исходном SIL graph pipeline.
- Seed igraph: 42.
- `n_iterations=-1`: Leiden выполняет итерации до стабилизации качества.
- Дополнительно сохраняется сырое Leiden-разбиение до фильтра размера.

В главном сравнении применён тот же постфильтр минимального размера к Leiden, который использует исходный SIL pipeline. Это нужно, чтобы сравнение назначений было сопоставимым.

## Установка и запуск (Windows PowerShell)

Из корня проекта:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\SIL_Stage1_Leiden_Comparison\RUN_STAGE1_LOUVAIN_LEIDEN.ps1 -ProjectRoot (Get-Location).Path
```

Wrapper устанавливает закреплённый `igraph==1.0.0`, если пакет отсутствует.

Явная установка вручную:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\SIL_Stage1_Leiden_Comparison\requirements-stage1.txt
```

## Входные файлы, обязательные для корректного сравнения

```text
artifacts/runs/20261120_015622_graph_b63e68/
  clusters.csv
  graph_edges.csv
  metrics.json
artifacts/embeddings/sentence-transformers_allenai-specter/
  manifest.json
  vectors.npy
```

Скрипт остановится с понятной ошибкой, если графовые артефакты отсутствуют, число вершин/рёбер не совпадает с `metrics.json`, IDs расходятся между графом и embedding manifest, веса невалидны или эмбеддинги не нормализованы.

## Результаты

```text
reports/community_detection_stage1/20261120_015622_graph_b63e68/
  summary.json
  REPORT_STAGE1_RU.md
  leiden_assignments.csv
  changed_documents.csv
  community_correspondence.csv
  community_sizes.csv
```

- `summary.json`: метрики, версии пакетов, параметры и SHA-256 входных данных.
- `REPORT_STAGE1_RU.md`: краткий отчёт.
- `leiden_assignments.csv`: исходные и новые метки на каждом документе.
- `changed_documents.csv`: документы, изменившие сообщество после сопоставимого фильтра.
- `community_correspondence.csv`: полная матрица Jaccard/пересечений для анализа split/merge плюс ориентировочное взаимно-однозначное сопоставление.
- `community_sizes.csv`: размеры групп.

Метрики включают модулярность на оставшихся вершинах, количество и размеры сообществ, шум, связность сообществ, ARI/NMI относительно слабой метки `domain`, ARI/NMI между разбиениями и Silhouette на общем наборе документов в исходном SPECTER cosine-пространстве. Для Silhouette используется один и тот же набор документов обеих моделей, максимум 1 500 точек.

## Чего этот этап не делает

- Не изменяет исходный Louvain run, SQLite, `latest_run.txt`, исходные метки или граф.
- Не меняет k, порог сходства, mutual kNN или SPECTER-векторы.
- Не утверждает, что Leiden тематически лучше только по более высокой модульности.
- Не проверяет многозапусковую устойчивость; это следующий этап после просмотра первого результата.

## Контроль качества

Локальные структурные тесты:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\SIL_Stage1_Leiden_Comparison\tests -v
```

Версия `igraph==1.0.0` закреплена для воспроизводимости. API Leiden принимает objective `modularity`, веса рёбер, resolution, beta и число итераций; детали см. в официальной документации python-igraph.
