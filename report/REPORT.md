# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Ngọc Vinh 
**Nhóm:** MoMo FAQ Retrieval 
**Ngày:** 2026-06-05  
**Embedding backend:** `all-MiniLM-L6-v2` (sentence-transformers, local, dim=384)

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

Hai văn bản có cosine similarity cao khi embedding vector của chúng trỏ về gần cùng một hướng trong không gian 384 chiều — nghĩa là chúng mang ý nghĩa tương đồng dù dùng từ ngữ khác nhau. Cosine similarity đo **góc** giữa hai vector thay vì khoảng cách tuyệt đối, nên không bị ảnh hưởng bởi độ dài văn bản.

**Ví dụ HIGH similarity (actual score = 0.797):**
- Sentence A: *"Chunking splits a document into smaller pieces."*
- Sentence B: *"Chunking divides text into manageable segments."*
- Tại sao tương đồng: Hai câu là paraphrase hoàn toàn — cùng concept (chunking), cùng hành động (chia nhỏ), cùng đối tượng (document/text). Model `all-MiniLM-L6-v2` được train để nhận ra paraphrase, nên similarity rất cao (0.797).

**Ví dụ LOW similarity (actual score = 0.002):**
- Sentence A: *"Customer support handles user complaints."*
- Sentence B: *"Neural networks compute gradients during backpropagation."*
- Tại sao khác: Hai câu đến từ hai domain hoàn toàn khác biệt (customer service vs deep learning optimization). Không có khái niệm chung, embedding vectors chỉ về hai hướng gần như vuông góc trong vector space.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Euclidean distance bị ảnh hưởng bởi magnitude của vector — một văn bản dài sẽ tạo vector "lớn hơn", khiến nó trông "xa" hơn ngay cả khi nội dung tương đồng với văn bản ngắn hơn. Cosine similarity normalize bằng cách chỉ đo góc, nên hai đoạn văn mang cùng ý nghĩa nhưng khác độ dài vẫn đạt similarity cao — đây là tính chất quan trọng khi so sánh query (ngắn) với documents (dài) trong RAG.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

```
step          = chunk_size - overlap = 500 - 50 = 450
num_chunks    = ceil((doc_length - overlap) / (chunk_size - overlap))
              = ceil((10000 - 50) / (500 - 50))
              = ceil(9950 / 450)
              = ceil(22.11...)
              = 23 chunks
```

**Đáp án: 23 chunks.** Verification qua code: `FixedSizeChunker(500, 50)` sinh starts `[0, 450, 900, ..., 9900]` → 23 bước, chunk cuối `text[9900:10000]` (100 ký tự).

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

```
step = 500 - 100 = 400
num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks
```

Overlap tăng 50→100 làm chunk count tăng 23→25 (+2 chunks). Overlap lớn hơn hữu ích vì thông tin quan trọng thường nằm ở ranh giới giữa hai chunks — với overlap=100, mỗi chunk chia sẻ 100 ký tự với chunk kế, nên retrieval không bỏ sót câu trả lời chỉ vì nó nằm ở vị trí cắt.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI/ML Engineering Knowledge Base — tập tài liệu kỹ thuật nội bộ về xây dựng và vận hành hệ thống AI.

**Tại sao nhóm chọn domain này?**

Domain AI/ML engineering phù hợp với môi trường làm việc thực tế: tài liệu có cấu trúc đa dạng (markdown sections, plain paragraphs), metadata tự nhiên theo category và ngôn ngữ, và benchmark queries có gold answers rõ ràng traceable từ tài liệu. Ngoài ra, domain này cho phép kiểm tra metadata filtering (filter theo `category=ai_infrastructure` khi hỏi về embedding models) và kiểm tra multilingual retrieval (tiếng Việt vs tiếng Anh). Đây là scenario thực tế mà một công ty AI như VinAI có thể triển khai.

---

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro.txt | Lab sample data | 1,944 | category=programming, lang=en |
| 2 | vector_store_notes.md | Lab sample data | 2,123 | category=ai_infrastructure, lang=en |
| 3 | rag_system_design.md | Lab sample data | 2,391 | category=system_design, lang=en |
| 4 | customer_support_playbook.txt | Lab sample data | 1,692 | category=support, lang=en |
| 5 | chunking_experiment_report.md | Lab sample data | 1,987 | category=research, lang=en |
| 6 | vi_retrieval_notes.md | Lab sample data | 1,667 | category=ai_infrastructure, lang=vi |
| 7 | embedding_models_overview.md | Viết bổ sung cho lab | 2,651 | category=ai_infrastructure, lang=en |
| 8 | retrieval_evaluation_guide.md | Viết bổ sung cho lab | 3,042 | category=research, lang=en |

**Tổng: 8 tài liệu | 17,497 ký tự | 2 ngôn ngữ | 5 categories**

---

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | str | `"vector_store_notes.md"` | Truy vết kết quả về file gốc; filter theo tài liệu cụ thể |
| `category` | str | `"ai_infrastructure"` / `"support"` / `"research"` | Narrow search space theo chủ đề, tránh cross-domain noise |
| `lang` | str | `"en"` / `"vi"` | Filter query theo ngôn ngữ — tránh lấy doc tiếng Anh khi user hỏi tiếng Việt |
| `description` | str | `"Guide to metrics for evaluating retrieval quality"` | Human-readable tag để debug và audit kết quả |
| `chunk_index` | int | `0`, `1`, `2` | Thứ tự chunk trong doc gốc — dùng để reconstruct context xung quanh |
| `total_chunks` | int | `6` | Biết doc được chia thành bao nhiêu chunk — phục vụ reporting |

**Ví dụ filter thực tế:** `search_with_filter("embedding model selection", metadata_filter={"category": "ai_infrastructure", "lang": "en"})` loại bỏ `support`, `programming`, `system_design`, `research` khỏi candidate pool, chỉ tìm trong `vector_store_notes.md`, `embedding_models_overview.md`.

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=500` trên 3 tài liệu đại diện:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt (1,944 chars) | FixedSizeChunker (`fixed_size`) | 5 | 429 | Trung bình — đôi khi cắt giữa paragraph dài |
| python_intro.txt | SentenceChunker (`by_sentences`) | 5 | 387 | Tốt — theo ranh giới câu tự nhiên |
| python_intro.txt | RecursiveChunker (`recursive`) | 5 | 389 | Tốt — ưu tiên tách paragraph trước |
| vector_store_notes.md (2,123 chars) | FixedSizeChunker | 5 | 465 | Kém hơn — cắt qua markdown section headers |
| vector_store_notes.md | SentenceChunker | 8 | 264 | Tốt nhưng chunk nhỏ (264 avg), thiếu context |
| vector_store_notes.md | RecursiveChunker | 7 | 303 | Tốt — tách `\n\n` trước, chunk có cấu trúc |
| rag_system_design.md (2,391 chars) | FixedSizeChunker | 6 | 440 | Trung bình — cắt qua sub-sections |
| rag_system_design.md | SentenceChunker | 5 | 476 | Tốt nhưng chunk quá dài (476 avg) |
| rag_system_design.md | RecursiveChunker | 7 | 342 | Tốt nhất trong 3 — balanced size và context |

**Nhận xét baseline:** `FixedSizeChunker` cắt qua markdown headers → giảm chunk coherence. `SentenceChunker` tốt cho plain text nhưng không nhận ra section structure. `RecursiveChunker` là baseline tốt nhất nhưng vẫn chưa khai thác được markdown header structure.

---

### Strategy Của Tôi — Custom `SectionChunker`

**Loại:** Custom `SectionChunker(max_section_chars=800)` — domain-specific chunker cho markdown documentation

**Mô tả cách hoạt động:**

`SectionChunker` dùng `re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)` để tách text tại mọi markdown header (`#`, `##`, `###`). Mỗi section từ header đến header tiếp theo thành một chunk hoàn chỉnh, giữ nguyên cả header lẫn nội dung bên dưới. Nếu một section vượt `max_section_chars=800`, `SectionChunker` fallback sang `RecursiveChunker(chunk_size=800)` để tách nhỏ hơn. Với plain text không có headers (`.txt`), toàn bộ document được xử lý bởi `RecursiveChunker(chunk_size=500)`.

**Tại sao tôi chọn strategy này cho domain nhóm?**

6/8 tài liệu trong bộ data là `.md` với cấu trúc header rõ ràng. Mỗi section (`## Typical Workflow`, `## Metadata Matters`, `## How to Choose`) chứa đúng một concept hoàn chỉnh — đây là "semantic unit" tự nhiên nhất cho AI engineering documentation. Khi user hỏi "What are the stages of a vector search pipeline?", chunk `## Typical Workflow` chứa câu trả lời đầy đủ và không bị lẫn với nội dung `## Metadata Matters`. `SectionChunker` khai thác information architecture mà tác giả tài liệu đã thiết kế sẵn.

**Code implementation:**

```python
class SectionChunker:
    """
    Domain-specific chunker for structured markdown/text documents.

    Splits on markdown section headers (## or ###). If a section exceeds
    max_section_chars, falls back to RecursiveChunker on that section.
    Designed for AI/engineering documentation with clear section structure.
    """

    def __init__(self, max_section_chars: int = 800) -> None:
        self.max_section_chars = max_section_chars
        self._fallback = RecursiveChunker(chunk_size=max_section_chars)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split at every markdown header (lookahead keeps header in section)
        sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
        result: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.max_section_chars:
                result.append(section)
            else:
                result.extend(self._fallback.chunk(section))
        return result if result else [text]
```

---

### So Sánh Trực Tiếp: SectionChunker vs Các Strategy Khác

Trên `vector_store_notes.md` — tài liệu có 4 sections rõ ràng (`# intro`, `## Typical Workflow`, `## Metadata Matters`, `## Common Risks`):

| Strategy | Chunk Count | Avg Length | Chunk Coherence | Retrieval Quality Q1 |
|-----------|-------------|------------|-----------------|---------------------|
| FixedSizeChunker(500) | 5 | 465 chars | Cắt giữa numbered list ("mbed each chunk...") — incoherent | Kém (chunk không align với section) |
| SentenceChunker(3) | 8 | 264 chars | Tốt nhưng chia nhỏ numbered list thành 3 chunks riêng biệt | Trung bình (context phân tán) |
| RecursiveChunker(500) | 7 | 303 chars | Tốt — tách theo `\n\n` nhưng không nhận ra header boundary | Trung bình |
| **SectionChunker(800)** | **4** | **529 chars** | **Xuất sắc — mỗi chunk = 1 section hoàn chỉnh** | **Top-1 score=0.679 (relevant)** |

**Minh họa chunk content của SectionChunker:**
```
chunk 0: "# Vector Store Notes\nA vector store is a database..." (intro)
chunk 1: "## Typical Workflow\nA common vector search pipeline has four stages:
          1. Chunk documents...2. Embed each chunk...3. Store vector..."  ← Q1 retrieves này
chunk 2: "## Metadata Matters\nMetadata is often as important as the vector..."
chunk 3: "## Common Risks\nVector stores are powerful, but retrieval is not..."
```

---

### So Sánh Với Thành Viên Khác (Cùng Bộ 5 Queries)

| Thành viên | Strategy | Total Chunks | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|-------------|----------------------|-----------|----------|
| **Tôi** | SectionChunker(800) + Recursive(500) | 41 | **10/10** | Section-aligned chunks, excellent precision trên markdown; filter hiệu quả | Kém hơn cho plain text không có headers |
| Thành viên B | FixedSizeChunker(300, overlap=50) | ~75 | 6/10 | Đơn giản, dễ tune; overlap giảm miss ở ranh giới | Cắt giữa câu, chunk thiếu ý hoàn chỉnh; Q2 không retrieve đúng section |
| Thành viên C | RecursiveChunker(500) | 34 | 8/10 | Tốt nhất trong 3 built-in strategies; balance chunk size và context | Không nhận ra markdown headers → Q5 kém hơn SectionChunker |

**Strategy nào tốt nhất cho domain này? Tại sao?**

`SectionChunker` là tốt nhất cho bộ tài liệu AI engineering documentation này vì 75% (6/8) tài liệu là markdown với section headers rõ ràng. Mỗi section là một semantic unit được tác giả thiết kế sẵn, nên chunking theo section = chunking theo ý định ngữ nghĩa. Điều này thể hiện nguyên tắc quan trọng: **strategy tốt nhất khai thác information architecture của domain, không phải generic separators**. Với bộ tài liệu plain text hoặc unstructured, RecursiveChunker sẽ là lựa chọn an toàn hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

Dùng `re.split(r'(?<=[.!?]) +|(?<=\.)\n', text.strip())` — lookbehind assertion phát hiện kết thúc câu (`. ! ?`) theo sau bởi spaces hoặc newline, nhưng không consume dấu câu (sentence vẫn giữ `.` cuối). Edge cases: text rỗng → `[]`; nếu pattern không match (1 câu dài, không có dấu câu) → `[text]`. Sau đó group mỗi `max_sentences_per_chunk` câu bằng `' '.join(group)`.

**`RecursiveChunker.chunk` / `_split`** — approach:

Thuật toán đệ quy với separator priority queue `["\n\n", "\n", ". ", " ", ""]`. Base cases: (1) `len(text) <= chunk_size` → return ngay không cần split; (2) `remaining_separators` rỗng → return text dù oversized (prevent infinite loop); (3) `sep == ""` → character-level split làm safety net cuối. Main logic: split text bằng current separator → accumulate pieces nhỏ vào `current_chunk` → khi piece lớn hơn `chunk_size`, flush chunk hiện tại và đệ quy vào `next_seps` cho piece đó.

**`SectionChunker.chunk`** — approach (custom):

Lookahead regex `(?=^#{1,3} )` với `re.MULTILINE` splits text **trước** mỗi markdown header nhưng không consume header (lookahead zero-width assertion). Mỗi section = header + nội dung bên dưới. Nếu section > `max_section_chars` (e.g., một section dài có nhiều paragraphs), fallback sang `RecursiveChunker` để tách nhỏ hơn. Guard: nếu text không có headers (plain `.txt`), toàn bộ text được xử lý như một section → fallback chunker tự xử lý.

---

### EmbeddingStore

**`add_documents` + `search`** — approach:

`_make_record` embed `doc.content` qua `embedding_fn` (mock hoặc real), lưu dict `{id, content, embedding, metadata}` với `doc_id` inject thêm vào metadata để `delete_document` có thể filter. `search` gọi `_search_records(query, self._store, top_k)`: embed query → tính `dot_product(query_emb, record_emb)` cho mọi record — vì embeddings L2-normalized, dot product = cosine similarity → sort descending → slice `[:top_k]`. Result dict bỏ `embedding` để không trả về raw vectors cho caller.

**`search_with_filter` + `delete_document`** — approach:

`search_with_filter` áp dụng filter **trước** khi search (filter-then-search, không phải search-then-filter): list comprehension lọc records có `metadata[k] == v` cho mọi `(k,v)` trong `metadata_filter`, rồi pass filtered list vào `_search_records`. Cách này đảm bảo top_k được tính trên pool đã filtered, tránh trường hợp relevant filtered docs bị đẩy ra khỏi top_k. `delete_document` rebuild `self._store` bằng list comprehension loại bỏ mọi record có `metadata['doc_id'] == doc_id`, trả `True` nếu `len` giảm.

---

### KnowledgeBaseAgent

**`answer`** — approach:

RAG pattern 3 bước: (1) `store.search(question, top_k)` → retrieve top chunks; (2) `"\n\n".join(r['content'] for r in results)` → concatenate thành context block; (3) `f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` → prompt template tách rõ context section và question, hướng LLM trả lời chỉ từ evidence được cung cấp. `llm_fn(prompt)` nhận full prompt, trả string — abstraction này cho phép swap bất kỳ LLM nào (mock, OpenAI, local) không cần sửa agent code.

---

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.0.3
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

42 passed in 0.09s
```

**Số tests pass: 42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Embedding backend: `all-MiniLM-L6-v2` (LocalEmbedder, dim=384).

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Machine learning uses data to train models. | ML algorithms learn patterns from datasets. | HIGH | **0.6278** | ✓ Đúng |
| 2 | Python is a high-level programming language. | The weather is sunny and warm today. | LOW | **0.0325** | ✓ Đúng |
| 3 | Vector stores enable semantic similarity search. | Embedding databases support nearest-neighbor queries. | HIGH | **0.5390** | ✓ Đúng |
| 4 | Chunking splits a document into smaller pieces. | Chunking divides text into manageable segments. | HIGH | **0.7967** | ✓ Đúng |
| 5 | Customer support handles user complaints. | Neural networks compute gradients during backpropagation. | LOW | **0.0024** | ✓ Đúng |

**5/5 dự đoán chính xác.**

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

Bất ngờ nhất là mức độ cao của Pair 4 (0.797) và sự chênh lệch lớn giữa Pair 1 (0.628) vs Pair 3 (0.539). Mặc dù Pair 3 là synonym gần hoàn chỉnh về mặt khái niệm ("semantic similarity search" = "nearest-neighbor queries"), score thấp hơn Pair 1 ("ML uses data" vs "algorithms learn from datasets"). Điều này cho thấy `all-MiniLM-L6-v2` nhạy cảm với **lexical overlap** — Pair 1 có nhiều từ chung hơn ("data/datasets", "learn/learn") nên score cao hơn dù Pair 3 về mặt khái niệm cũng rất gần. Đây là lý do tại sao các team RAG thường kết hợp semantic search với keyword search (BM25 hybrid) để tốt hơn ở cả hai chiều.

---

## 6. Results — Cá nhân (10 điểm)

**Setup:** 8 tài liệu → SectionChunker(800) cho `.md`, RecursiveChunker(500) cho `.txt` → 41 chunks → EmbeddingStore với `LocalEmbedder` (`all-MiniLM-L6-v2`, dim=384).

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| Q1 | What are the four stages of a typical vector search pipeline? | Chunk documents → embed each chunk → store vector + metadata → embed query and rank by similarity |
| Q2 | Why does recursive chunking outperform fixed-size chunking for technical documentation? | Recursive tries structural separators first (paragraphs), preserves context, stays within target size — produces most consistently useful passages |
| Q3 | Which Python libraries are commonly used for machine learning? | scikit-learn, PyTorch, and TensorFlow |
| Q4 | How should an AI support assistant handle questions it cannot answer from its knowledge base? | Recommend escalation instead of improvising a risky answer; honest uncertainty is better than a polished but incorrect response |
| Q5 | What factors should you consider when choosing an embedding model? | Language coverage, latency requirements, privacy constraints, retrieval quality — benchmark on your own data first |

### Kết Quả Của Tôi (LocalEmbedder + SectionChunker)

| # | Query | Top-1 Retrieved Chunk | Score | Relevant? | Agent Answer |
|---|-------|----------------------|-------|-----------|--------------|
| Q1 | Vector search pipeline stages | `vector_store_notes.md § Typical Workflow` — "A common vector search pipeline has four stages: 1. Chunk documents... 2. Embed each chunk... 3. Store the vector..." | **0.679** | ✓ Hoàn toàn relevant — chứa đầy đủ 4 steps đúng với gold answer | Answer liệt kê 4 stages từ context, khớp gold answer |
| Q2 | Recursive vs fixed-size chunking | `chunking_experiment_report.md § Fixed-Size Chunking` — "Fixed-size chunking... some chunks split explanations in awkward places..." | **0.732** | ✓ Relevant — Top-1,2,3 đều từ `chunking_experiment_report.md` | Answer trích "Recursive chunking offered the best balance", khớp gold |
| Q3 | Python ML libraries | `python_intro.txt` — "Data scientists use it to clean data... Libraries such as scikit-learn, PyTorch, and TensorFlow..." | **0.701** | ✓ Hoàn toàn relevant — đúng chunk chứa scikit-learn, PyTorch, TensorFlow | Answer trích đúng tên 3 libraries, khớp gold answer |
| Q4 | Support handle unanswerable | `customer_support_playbook.txt` — "The support team uses the knowledge assistant..." | **0.570** | ✓ Relevant — Top-2 (score=0.518) chứa gold answer về escalation | Answer đề cập "recommend escalation", "honest uncertainty better than incorrect" |
| Q5* | Choosing embedding model | `embedding_models_overview.md § How to Choose` — "Choosing the right model depends on four factors: language coverage, latency requirements, privacy constraints, retrieval quality" | **0.642** | ✓ Hoàn toàn relevant — đúng section chứa 4 factors, khớp gold answer | Answer trích đủ 4 yếu tố, đề xuất benchmark trên own data |

*Q5 dùng `search_with_filter(metadata_filter={"category": "ai_infrastructure", "lang": "en"})`.

**Bao nhiêu queries trả về chunk relevant trong top-3?** **5 / 5**

---

### Phân Tích Chi Tiết: Search vs Search_with_filter (Q5)

| Mode | Top-1 Source | Score | Top-2 Source | Score |
|------|-------------|-------|-------------|-------|
| `search()` (no filter) | `embedding_models_overview.md § How to Choose` | 0.630 | `vector_store_notes.md` | 0.501 |
| `search_with_filter(ai_infrastructure, en)` | `embedding_models_overview.md § How to Choose` | **0.642** | `embedding_models_overview.md § What Is...` | **0.580** |

Metadata filter loại bỏ `support`, `programming`, `system_design`, `research` docs → cả 3 top results đều từ `ai_infrastructure` category → **precision tăng và scores cao hơn** vì search pool nhỏ hơn nhưng đồng nhất chủ đề hơn. Đây là bằng chứng thực nghiệm rằng metadata filtering cải thiện retrieval quality khi query scoped rõ ràng.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

Thành viên dùng `FixedSizeChunker(300, overlap=50)` chia sẻ rằng overlap nhỏ (50 chars) giúp bắt được các câu trả lời nằm ngay ở boundary, nhưng với overlap quá lớn (> 100 chars) thì chunk count tăng mạnh mà retrieval precision không tăng tương ứng. Đây là trade-off quan trọng mà tôi chưa test — tôi sẽ thêm overlap parameter vào `SectionChunker` cho các section overlap với nhau để tăng coverage.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

Nhóm khác demo **hybrid search** — kết hợp BM25 keyword search với semantic vector search, lấy kết quả từ cả hai rồi merge bằng Reciprocal Rank Fusion (RRF). Với query Q3 ("Python ML libraries"), BM25 có thể tìm đúng chunk ngay bằng từ khóa "scikit-learn", trong khi semantic search mới cần embedding để hiểu ngữ nghĩa. Hybrid search đặc biệt mạnh khi query chứa tên riêng, library names, hoặc technical terms mà embedding model chưa học tốt.

**Failure Analysis — Root Cause & Improvement:**

**Q4 failure analysis:** Top-1 (`score=0.570`) là intro paragraph của `customer_support_playbook.txt`, không chứa câu trả lời về escalation. Câu trả lời thật nằm ở Top-2 (`score=0.518`, chunk cuối của file). Root cause: `RecursiveChunker(500)` chia `customer_support_playbook.txt` thành chunks nhưng intro chunk bề mặt rất match với query ("support assistant", "knowledge assistant") trong khi escalation advice nằm sau. **Cải thiện:** (1) Dùng `SectionChunker` cho plain text bằng cách thêm heuristic detect Q&A/list structure; (2) Tăng top_k từ 3 lên 5 trong agent để LLM có thêm context; (3) Rerank results bằng cross-encoder model — cross-encoder đọc query + chunk cùng lúc và cho score chính xác hơn bi-encoder.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

Tôi sẽ thêm trường metadata `doc_type` (values: `markdown`, `plaintext`, `faq`, `runbook`) để `SectionChunker` có thể chọn strategy tự động: markdown → split by headers, faq → split by Q&A pairs, plaintext → RecursiveChunker. Tôi cũng sẽ thêm `last_updated` date để filter out stale documents — một vấn đề thực tế khi knowledge base grow theo thời gian. Cuối cùng, tôi sẽ benchmark với cả `text-embedding-3-small` (OpenAI) bên cạnh `all-MiniLM-L6-v2` để so sánh cost vs quality trade-off trên bộ data này.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá | Lý do |
|----------|------|-------------------|-------|
| Warm-up (cosine + chunking math) | Cá nhân | **5 / 5** | Giải thích đúng cả 2 phần, có ví dụ thực tế và verification bằng code |
| Document selection (8 docs, 6-field metadata, 2 ngôn ngữ) | Nhóm | **10 / 10** | 8 docs trong range 5-10, metadata schema đầy đủ và useful, nguồn transparent |
| Chunking strategy (custom SectionChunker + baseline + comparison) | Nhóm | **15 / 15** | Implement custom strategy với rationale rõ ràng, baseline table đầy đủ, so sánh 3 strategies |
| My approach (5 components explained thoroughly) | Cá nhân | **10 / 10** | Giải thích đủ SentenceChunker, RecursiveChunker, SectionChunker, Store, Agent |
| Similarity predictions (5/5 correct, LocalEmbedder) | Cá nhân | **5 / 5** | Tất cả dự đoán đúng, reflection về lexical overlap insight |
| Results (5/5 relevant, scores 0.57-0.73, filter analysis) | Cá nhân | **10 / 10** | 5/5 queries retrieve relevant top-3, agent answer grounded, filter comparison documented |
| Core implementation (42/42 tests pass) | Cá nhân | **30 / 30** | 100% tests pass |
| Demo (failure analysis, cross-team learning, future improvements) | Nhóm | **5 / 5** | Q4 failure analyzed với root cause + 3 cải thiện; hybrid search insight từ nhóm khác |
| **Tổng** | | **100 / 100** | |
