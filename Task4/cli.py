from __future__ import annotations

from Task4.rag_service import RAGService


def main() -> None:
    rag = RAGService()
    print("Astraforge RAG. Type 'exit' to stop.")

    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = rag.answer(question)
        print(result["answer"])
        print(f"\nretrieval score: {result['top_score']:.4f}")


if __name__ == "__main__":
    main()
