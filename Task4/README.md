# Task 4 — RAG-бот

## Pipeline

```text
Question
   ↓
Sentence-Transformers
   ↓
FAISS search (Top-K)
   ↓
Relevance threshold
   ├── low score ──→ "Я не знаю"
   ↓
Retrieved context
   ↓
Few-shot examples + system prompt
   ↓
LLM (Ollama or OpenAI)
   ↓
Answer + evidence + sources
```

## Few-shot

В prompt добавлены два примера из Astraforge:

- `What is the Void Core?`
- `Who trains Lio Arken on Mirehaven?`

Они используют факты, которые реально присутствуют в `knowledge_base`.

## CoT / объяснение

Модель получает инструкцию сначала проанализировать контекст, но не раскрывать скрытую внутреннюю цепочку рассуждений. Вместо raw Chain-of-Thought ответ содержит **короткое проверяемое Evidence summary**:

```text
Answer: ...
Evidence:
1. Факт из найденного документа.
2. Второй факт, если он нужен.
Sources: ...
```

Так сохраняется требование задания об объяснимости, но пользователю показываются только проверяемые шаги, основанные на источниках.

## Запуск

Сначала построить индекс:

```bash
python Task3/build_index.py
```

### Вариант 1 — локальная LLM через Ollama

```bash
ollama pull qwen3:8b
ollama serve

export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen3:8b
export OLLAMA_BASE_URL=http://localhost:11434

uvicorn Task4.app:app --reload
```

### Вариант 2 — OpenAI

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.6-luna

uvicorn Task4.app:app --reload
```

## Проверка

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Who trained Lio Arken on Mirehaven?"}'
```

Для диагностики retrieval:

```bash
curl "http://localhost:8000/search?q=What%20is%20the%20Void%20Core"
```

## «Я не знаю»

По умолчанию:

```text
RAG_TOP_K=4
RAG_MIN_SCORE=0.42
```

`RAG_MIN_SCORE` нужно откалибровать после реального запуска по `Task4/test_cases.json`. Если нерелевантные вопросы получают слишком высокий score — порог повышается. Если валидные вопросы отклоняются — понижается.

В `examples.md` подготовлены 5 answerable и 5 out-of-domain вопросов для требуемых скриншотов.
