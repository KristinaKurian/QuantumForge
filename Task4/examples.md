# Диалоги для проверки и скриншотов

После запуска API отправляйте запросы на `POST /ask`.

## 5 запросов, на которые бот должен ответить

1. `Who trained Lio Arken on Mirehaven?`
2. `What is the Void Core?`
3. `Why was Directive Blackglass so effective?`
4. `Who are Lio Arken's closest known family members?`
5. `How did the Free Systems Coalition survive the Siege of Nivora?`

Ответ должен содержать факты из найденных документов и список источников.

## 5 запросов, на которые бот должен ответить «Я не знаю»

1. `Who founded QuantumForge Software?`
2. `What is the capital of Finland?`
3. `What database does the Astraforge payroll service use?`
4. `Who won the football world championship in 2026?`
5. `What is the population of Kharis in 1420?`

Эти вопросы не покрываются базой Astraforge. Для них срабатывает retrieval threshold и LLM не вызывается.

## Почему это важно

Проверяются два независимых механизма:

1. `RAG_MIN_SCORE` блокирует генерацию, если retrieval не нашёл достаточно близкий контекст.
2. System prompt запрещает использовать внешние знания и требует `Я не знаю`, если найденных фактов недостаточно.
