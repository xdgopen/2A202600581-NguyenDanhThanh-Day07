from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
)


FAQ_PATH = Path("data/momo_faq.md")

BENCHMARKS = [
    {
        "query": "MoMo là gì và có những chức năng chính nào?",
        "gold": "MoMo là Trợ thủ Tài chính, hỗ trợ chuyển nhận tiền, thanh toán dịch vụ, quản lý chi tiêu, tín dụng, đầu tư và kinh doanh.",
        "expected_question": "MoMo là gì?",
        "filter": None,
    },
    {
        "query": "Liên hệ chăm sóc khách hàng MoMo qua những kênh nào?",
        "gold": "Có thể gửi yêu cầu ở mục Trợ giúp, gọi hotline 1900 5454 41 hoặc email hotro@momo.vn.",
        "expected_question": "Cách liên hệ với MoMo",
        "filter": None,
    },
    {
        "query": "Phí rút tiền MoMo khi vượt 30 triệu trong tháng là bao nhiêu?",
        "gold": "Miễn phí nếu tổng rút trong tháng không quá 30 triệu đồng; vượt 30 triệu phí 0,5%; vượt 100 triệu phí 1%.",
        "expected_question": "Sử dụng MoMo có mất phí không?",
        "filter": None,
    },
    {
        "query": "Gỡ ứng dụng MoMo thì tiền và lịch sử giao dịch có mất không?",
        "gold": "Không. Gỡ ứng dụng không ảnh hưởng tài khoản hoặc giao dịch; dữ liệu lưu trên máy chủ và phục hồi khi cài lại.",
        "expected_question": "Khi gỡ bỏ ứng dụng trên điện thoại thì tiền và thông tin giao dịch có bị mất không?",
        "filter": {"category": "Quản lý Tài khoản & Bảo mật"},
    },
    {
        "query": "Mua vé xem phim trên MoMo thực hiện các bước nào?",
        "gold": "Chọn Mua vé xem phim, chọn phim/suất chiếu, chọn ghế, chọn combo nếu có, kiểm tra thanh toán, rồi quét QR/Barcode tại rạp.",
        "expected_question": "Cách mua vé xem phim trên MoMo",
        "filter": None,
    },
]

STRATEGIES = {
    "fixed_size": FixedSizeChunker(chunk_size=500, overlap=50),
    "fixed_size_large": FixedSizeChunker(chunk_size=900, overlap=120),
    "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
    "recursive": RecursiveChunker(chunk_size=500),
    "by_category": None,
    "qa_custom": None,
}


class HashingKeywordEmbedder:
    """Offline lexical embedder for deterministic FAQ retrieval demos."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._backend_name = "offline hashing keyword embedder"

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[0-9a-zà-ỹđ]+", text.lower())


def parse_momo_faq(path: Path = FAQ_PATH) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    documents: list[Document] = []
    current_category = ""
    current_question = ""
    current_lines: list[str] = []

    def flush() -> None:
        if not current_question or not current_lines:
            return
        content = f"{current_question}\n" + "\n".join(current_lines).strip()
        documents.append(
            Document(
                id=f"momo_faq_{len(documents) + 1}",
                content=content,
                metadata={
                    "source": str(path),
                    "domain": "momo_faq",
                    "language": "vi",
                    "category": current_category,
                    "question": current_question,
                    "strategy": "qa_custom",
                },
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            flush()
            current_category = re.sub(r"^##\s*\d+\.\s*", "", line).strip()
            current_question = ""
            current_lines = []
        elif line.startswith("### "):
            flush()
            current_question = line.replace("### ", "", 1).strip()
            current_lines = []
        elif current_question:
            current_lines.append(line)

    flush()
    return documents


def parse_momo_categories(path: Path = FAQ_PATH) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    documents: list[Document] = []
    matches = list(re.finditer(r"^##\s*\d+\.\s*(.+)$", text, flags=re.MULTILINE))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        category = match.group(1).strip()
        content = text[start:end].strip()
        documents.append(
            Document(
                id=f"momo_category_{index + 1}",
                content=content,
                metadata={
                    "source": str(path),
                    "domain": "momo_faq",
                    "language": "vi",
                    "category": category,
                    "question": category,
                    "strategy": "by_category",
                },
            )
        )

    return documents


def category_at_position(text: str, position: int) -> str:
    category = "unknown"
    for match in re.finditer(r"^##\s*\d+\.\s*(.+)$", text, flags=re.MULTILINE):
        if match.start() > position:
            break
        category = match.group(1).strip()
    return category


def build_chunk_documents(strategy_name: str, raw_text: str) -> list[Document]:
    if strategy_name == "qa_custom":
        return parse_momo_faq()
    if strategy_name == "by_category":
        return parse_momo_categories()

    chunker = STRATEGIES[strategy_name]
    chunks = chunker.chunk(raw_text)
    documents: list[Document] = []
    search_from = 0

    for index, chunk in enumerate(chunks, start=1):
        position = raw_text.find(chunk, search_from)
        if position < 0:
            position = raw_text.find(chunk.strip())
        if position < 0:
            position = 0
        search_from = max(search_from, position + len(chunk))

        question_match = re.search(r"^###\s*(.+)$", chunk, flags=re.MULTILINE)
        documents.append(
            Document(
                id=f"{strategy_name}_{index}",
                content=chunk,
                metadata={
                    "source": str(FAQ_PATH),
                    "domain": "momo_faq",
                    "language": "vi",
                    "category": category_at_position(raw_text, position),
                    "question": question_match.group(1).strip() if question_match else "",
                    "strategy": strategy_name,
                },
            )
        )

    return documents


def concise_llm(prompt: str) -> str:
    context_match = re.search(r"\[1\].*?(?=\n\nQuestion:)", prompt, flags=re.DOTALL)
    if not context_match:
        return "Không tìm thấy đủ thông tin trong knowledge base."
    context = context_match.group(0)
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    return " ".join(lines[1:4])


def answer_from_results(question: str, results: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        source = result.get("metadata", {}).get("question") or result.get("metadata", {}).get("category") or result.get("id")
        context_blocks.append(f"[{index}] Source: {source}\n{result['content']}")
    context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return concise_llm(prompt)


def print_chunking_baseline(raw_text: str) -> None:
    print("=== Chunking Baseline: data/momo_faq.md ===")
    comparison = ChunkingStrategyComparator().compare(raw_text, chunk_size=500)
    for name, stats in comparison.items():
        print(f"{name}: count={stats['count']}, avg_length={stats['avg_length']:.1f}")


def is_relevant(result: dict[str, Any], benchmark: dict[str, Any]) -> bool:
    expected = benchmark["expected_question"].lower()
    question = str(result.get("metadata", {}).get("question", "")).lower()
    content = str(result.get("content", "")).lower()
    return expected in question or expected in content


def coherence_score(documents: list[Document]) -> float:
    if not documents:
        return 0.0

    coherent = 0
    for doc in documents:
        has_question = bool(doc.metadata.get("question")) or "### " in doc.content
        has_answer_text = len(doc.content.strip()) >= 80
        starts_as_fragment = doc.content.strip().startswith(("-", "B2:", "B3:", "B4:", "Bước 2", "Bước 3"))
        coherent += int(has_question and has_answer_text and not starts_as_fragment)
    return coherent / len(documents)


def evaluate_strategy(strategy_name: str, raw_text: str) -> dict[str, Any]:
    documents = build_chunk_documents(strategy_name, raw_text)
    store = EmbeddingStore(
        collection_name=f"momo_{strategy_name}",
        embedding_fn=HashingKeywordEmbedder(),
    )
    store.add_documents(documents)
    agent = KnowledgeBaseAgent(store=store, llm_fn=concise_llm)

    rows = []
    top1_hits = 0
    top3_hits = 0
    grounding_hits = 0
    precision_values = []

    for benchmark in BENCHMARKS:
        if benchmark["filter"]:
            results = store.search_with_filter(
                benchmark["query"],
                top_k=3,
                metadata_filter=benchmark["filter"],
            )
        else:
            results = store.search(benchmark["query"], top_k=3)

        relevance = [is_relevant(result, benchmark) for result in results]
        top1_relevant = bool(relevance and relevance[0])
        top3_relevant = any(relevance)
        top1_question = str(results[0].get("metadata", {}).get("question", "")) if results else ""
        grounding_ok = top1_question == benchmark["expected_question"]
        precision = sum(relevance) / len(results) if results else 0.0
        top1_hits += int(top1_relevant)
        top3_hits += int(top3_relevant)
        grounding_hits += int(grounding_ok)
        precision_values.append(precision)

        top1 = results[0] if results else {}
        top1_label = top1.get("metadata", {}).get("question") or top1.get("content", "")[:70].replace("\n", " ")
        rows.append(
            {
                "query": benchmark["query"],
                "gold": benchmark["gold"],
                "filter": benchmark["filter"],
                "top1": top1_label,
                "score": top1.get("score", 0.0),
                "top1_relevant": top1_relevant,
                "top3_relevant": top3_relevant,
                "grounding_ok": grounding_ok,
                "precision_at_3": precision,
                "agent_answer": answer_from_results(benchmark["query"], results),
            }
        )

    avg_length = sum(len(doc.content) for doc in documents) / len(documents) if documents else 0.0
    return {
        "strategy": strategy_name,
        "documents": documents,
        "count": len(documents),
        "avg_length": avg_length,
        "coherence": coherence_score(documents),
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "grounding_hits": grounding_hits,
        "precision_at_3": sum(precision_values) / len(precision_values),
        "rows": rows,
    }


def print_strategy_comparison(evaluations: list[dict[str, Any]]) -> None:
    print("\n=== Phase 2 Strategy Comparison ===")
    print("strategy | chunks | avg_len | coherence | top1 | top3 | grounding | avg_precision@3")
    for item in evaluations:
        print(
            f"{item['strategy']} | {item['count']} | {item['avg_length']:.1f} | "
            f"{item['coherence']:.2f} | {item['top1_hits']}/5 | {item['top3_hits']}/5 | "
            f"{item['grounding_hits']}/5 | "
            f"{item['precision_at_3']:.2f}"
        )


def print_metadata_utility(raw_text: str) -> None:
    print("\n=== Metadata Utility: Filter vs No Filter ===")
    benchmark = next(item for item in BENCHMARKS if item["filter"])
    for strategy_name in STRATEGIES:
        documents = build_chunk_documents(strategy_name, raw_text)
        store = EmbeddingStore(
            collection_name=f"momo_filter_{strategy_name}",
            embedding_fn=HashingKeywordEmbedder(),
        )
        store.add_documents(documents)
        unfiltered = store.search(benchmark["query"], top_k=3)
        filtered = store.search_with_filter(benchmark["query"], top_k=3, metadata_filter=benchmark["filter"])

        unfiltered_hit = any(is_relevant(result, benchmark) for result in unfiltered)
        filtered_hit = any(is_relevant(result, benchmark) for result in filtered)
        unfiltered_top1 = unfiltered[0].get("metadata", {}).get("question") if unfiltered else ""
        filtered_top1 = filtered[0].get("metadata", {}).get("question") if filtered else ""
        print(
            f"{strategy_name}: unfiltered_hit={unfiltered_hit}, filtered_hit={filtered_hit}, "
            f"unfiltered_top1={unfiltered_top1!r}, filtered_top1={filtered_top1!r}"
        )


def print_detailed_best_results(best: dict[str, Any]) -> None:
    print(f"\n=== Detailed Results: {best['strategy']} ===")
    relevant_top3 = 0
    for index, row in enumerate(best["rows"], start=1):
        relevant_top3 += int(row["top3_relevant"])
        print(f"\n{index}. Query: {row['query']}")
        print(f"Gold: {row['gold']}")
        print(f"Filter: {row['filter']}")
        print(f"Top-1: {row['top1']} | score={row['score']:.3f}")
        print(f"Precision@3: {row['precision_at_3']:.2f}")
        print(f"Top-3 relevant: {row['top3_relevant']}")
        print(f"Grounding OK: {row['grounding_ok']}")
        print(f"Grounded agent answer: {row['agent_answer']}")

    print(f"\nRelevant in top-3: {relevant_top3}/{len(BENCHMARKS)}")


def print_self_evaluation(evaluations: list[dict[str, Any]], raw_text: str) -> None:
    best = max(
        evaluations,
        key=lambda item: (item["grounding_hits"], item["top1_hits"], item["top3_hits"], -item["avg_length"]),
    )
    print("\n=== Cách Tự Đánh Giá Kết Quả Retrieval ===")
    print("1. Retrieval Precision")
    print(
        f"Best strategy: {best['strategy']} with top1={best['top1_hits']}/5, "
        f"top3={best['top3_hits']}/5, grounding={best['grounding_hits']}/5, "
        f"avg_precision@3={best['precision_at_3']:.2f}."
    )
    print("2. Chunk Coherence")
    print(
        "Q&A chunks are most coherent for FAQ because each chunk keeps one question "
        "with its complete answer and metadata."
    )
    print("3. Metadata Utility")
    print("The account/security query uses category filtering; compare filtered vs unfiltered hits below.")
    print("4. Grounding Quality")
    print("Displayed agent answers are copied/summarized from retrieved top context, so each answer is traceable to Top-1.")
    print("5. Data Strategy Impact")
    print(
        "The MoMo file has strong Markdown Q&A structure; the custom parser uses that structure, "
        "while generic chunkers can split headings from answers."
    )
    print_metadata_utility(raw_text)
    print_detailed_best_results(best)


def run_benchmark() -> None:
    raw_text = FAQ_PATH.read_text(encoding="utf-8")
    print_chunking_baseline(raw_text)

    evaluations = [evaluate_strategy(strategy_name, raw_text) for strategy_name in STRATEGIES]
    print_strategy_comparison(evaluations)

    docs = parse_momo_faq()
    print("\n=== Parsed FAQ Documents ===")
    print(f"count={len(docs)}")
    categories = sorted({doc.metadata["category"] for doc in docs})
    print("categories=" + ", ".join(categories))

    print_self_evaluation(evaluations, raw_text)


if __name__ == "__main__":
    run_benchmark()
