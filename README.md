# RAG Homework Indexer

CLI-утилита на Kotlin для индексации документов через Ollama embeddings.

## Сборка

```powershell
gradle jar
```

Готовый runnable jar появится в `build/libs/rag-indexer.jar`.

## Запуск

Запускать jar можно из любой директории. Эта директория считается рабочей: программа рекурсивно ищет переданное имя документа во вложенных каталогах.

```powershell
java -jar D:\RagHomework\build\libs\rag-indexer.jar document.pdf --strategy fixed --size 1200 --overlap 150 --pages 10
```

```powershell
java -jar D:\RagHomework\build\libs\rag-indexer.jar notes.md --strategy semantic --max-chars 1600 --threshold 0.72
```

По умолчанию используется Ollama `http://localhost:11434` и модель `nomic-embed-text`.

Перед запуском убедитесь, что Ollama поднята и модель загружена:

```powershell
ollama pull nomic-embed-text
ollama serve
```

Результат сохраняется рядом с jar как `<document>-index.json`.
