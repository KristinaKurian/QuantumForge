# Task 3 — Векторный индекс

## Реализация

| Параметр | Значение |
|---|---|
| Embedding model | `sentence-transformers/multi-qa-mpnet-base-cos-v1` |
| Device | CPU |
| Chunk size | 180 слов |
| Overlap | 30 слов |
| Index | FAISS `IndexFlatIP` |
| Similarity | cosine через normalized inner product |
| Batch size | 4 |
| Multiprocessing | отключён |

Для каждого чанка сохраняются `source`, `title`, `chunk_id`, позиция в документе и текст.

Индексация выполняется небольшими батчами: embeddings сразу добавляются в FAISS, поэтому полный массив векторов не хранится в памяти.

## Построение

```bash
python Task3/build_index.py --force
```

Результат:

```text
Task3/index/
├── faiss.index
├── chunks.jsonl
├── manifest.json
└── BUILD_REPORT.md
```

`manifest.json` и `BUILD_REPORT.md` содержат фактические параметры и время построения.

## Проверка retrieval

```bash
python Task3/search_index.py "Who trained Lio Arken on Mirehaven?"
python Task3/smoke_test.py
```

Smoke test проверяет, что ожидаемые документы попадают в `top-3`.
