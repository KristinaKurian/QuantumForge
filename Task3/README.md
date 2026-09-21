# Task 3 — Создание векторного индекса

## Выбранная embedding-модель

Используется:

```text
sentence-transformers/multi-qa-mpnet-base-cos-v1
```

Модель предназначена для semantic search. Размер embedding определяется
автоматически через `get_sentence_embedding_dimension()` и сохраняется
в `Task3/index/manifest.json`.

## Что изменено для стабильной работы на macOS

Версия Task 3 специально сделана без multiprocessing:

- embedding model всегда работает на `CPU`;
- MPS/Metal для индексации не используется;
- `TOKENIZERS_PARALLELISM=false`;
- FAISS / PyTorch / BLAS ограничены одним native thread;
- embeddings создаются батчами по 4 чанка;
- каждый batch сразу добавляется в FAISS;
- полный массив embeddings не хранится в RAM;
- временные объекты удаляются после каждого batch;
- вызывается `gc.collect()`;
- FAISS индекс сначала пишется во временный файл, затем атомарно
  переименовывается.

Это уменьшает вероятность `segmentation fault`, `leaked semaphore`
и скачков памяти при построении индекса.

---

## Chunking

В Task 2 один документ соответствует одной сущности, поэтому используется
простой детерминированный word-based splitter:

```text
chunk_size = 180 words
overlap = 30 words
```

Для каждого чанка сохраняются:

- `source`;
- `title`;
- `chunk_id`;
- `chunk_index`;
- `word_start`;
- `word_end`;
- `word_count`;
- `text`.

Таким образом, бот сможет показывать источник и позицию найденного фрагмента.

---

## 1. Проверить окружение

```bash
python Task3/check_runtime.py
```

Команда только показывает версии библиотек и подтверждает, что Task 3
будет использовать CPU.

---

## 2. Построить индекс

Из корня репозитория:

```bash
python Task3/build_index.py --force
```

При работе будет отображаться:

```text
Documents: ...
Chunks: ...
Loading embedding model on CPU: ...
Embedding dimension: ...
Encoded 4/... chunks
Encoded 8/... chunks
...
Index successfully built
```

Результат:

```text
Task3/index/
├── faiss.index
├── chunks.jsonl
├── manifest.json
└── BUILD_REPORT.md
```

`manifest.json` и `BUILD_REPORT.md` формируются автоматически и содержат
фактическое число документов, чанков и время построения.

---

## 3. Проверить поиск

```bash
python Task3/search_index.py "Who trained Lio Arken on Mirehaven?"
```

Ещё примеры:

```bash
python Task3/search_index.py "What happened during Directive Blackglass?"
```

```bash
python Task3/search_index.py "How was the first Void Core destroyed?"
```

---

## 4. Запустить smoke test

```bash
python Task3/smoke_test.py
```

Проверяются три контрольных запроса. Ожидается, что правильный документ
попадает в `top-3`.

---

## Если предыдущий запуск завершился аварийно

Удалите только недостроенный индекс:

```bash
rm -f Task3/index/faiss.index
rm -f Task3/index/faiss.index.tmp
rm -f Task3/index/chunks.jsonl
rm -f Task3/index/manifest.json
rm -f Task3/index/BUILD_REPORT.md
```

Затем:

```bash
python Task3/build_index.py --force
```

Предупреждение `resource_tracker: leaked semaphore` после старого
`segmentation fault` является следствием аварийного завершения процесса.
Новая версия Task 3 не создаёт multiprocessing pool и не использует
worker processes при построении embeddings.
