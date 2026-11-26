# Локальная модель для семантического именования

## Выбранная модель

`qwen3.5:4b` через Ollama. В опубликованной сборке Ollama указаны Q4_K_M-квантизация и размер около 4 GB; модель распространяется по Apache 2.0 и заявляет поддержку 201 языка и диалектов. Это instruction-модель, то есть она генерирует текстовые ответы, в отличие от SPECTER, который кодирует публикации в векторы.

## Установка в Windows

1. Установить Ollama: https://ollama.com/download
2. Открыть PowerShell:

```powershell
ollama pull qwen3.5:4b
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
``` 

Во время генерации в другом окне можно проверить распределение на GPU командой:

```powershell
ollama ps
nvidia-smi
```

3. Запустить пайплайн из корня проекта:

```powershell
.\auto_topic_discovery_app\RUN_AUTO_TOPIC_DISCOVERY.ps1
```

Параметры по умолчанию:
- модель: `qwen3.5:4b`;
- API: `http://127.0.0.1:11434`;
- контекст: 4096 токенов;
- temperature: 0.1;
- seed генерации: 42;
- thinking mode: отключён;
- structured output: JSON schema.

Для альтернативной локальной Ollama-модели:

```powershell
.\auto_topic_discovery_app\RUN_AUTO_TOPIC_DISCOVERY.ps1 -LabelerModel "ИМЯ_МОДЕЛИ"
```

Если надо запустить прежний словарно-ключевой механизм без LLM:

```powershell
.\auto_topic_discovery_app\RUN_AUTO_TOPIC_DISCOVERY.ps1 -Labeler heuristic
```

## Что получает модель

На каждое сообщество отдельный запрос с:
- ключевыми фразами YAKE и RAKE;
- контрастными TF-IDF-терминами;
- частотами тематических доменов;
- шестью публикациями, ближайшими к центроиду;
- фрагментами аннотаций (если они были найдены в `data/` или `artifacts/`).

Модель отвечает русским и английским названием, кратким описанием и замечанием о возможной неоднородности группы.

## Где проверить результат

В каталоге результата:

`reports/auto_topic_discovery/<RUN_ID>/<TIMESTAMP>/`

- `topic_profiles.csv` — названия и описания;
- `topic_label_generation.json` — фактические evidence и ответы модели;
- `summary.json` — Ollama version, model digest и параметры генерации.

В Streamlit выводятся описание группы и замечание о неоднородности; название можно отредактировать.

Модель вызывается по локальному API. При стандартном адресе `127.0.0.1` данные не отправляются облачному провайдеру. Если пользователь вручную задаст удалённый `-OllamaUrl`, запросы пойдут на этот адрес.
