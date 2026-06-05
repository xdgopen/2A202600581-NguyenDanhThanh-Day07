# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Tuấn Phương  
**MSSV:** 2A202600772  
**Nhóm:** MoMo FAQ Retrieval
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector embedding trỏ gần cùng một hướng, tức là hai đoạn văn có ý nghĩa hoặc chủ đề gần nhau dù không nhất thiết dùng đúng cùng từ khóa.

**Ví dụ HIGH similarity:**
- Sentence A: Python is used for data analysis and machine learning.
- Sentence B: Python supports machine learning and data workflows.
- Tại sao tương đồng: cả hai đều nói về Python trong bối cảnh data và machine learning.

**Ví dụ LOW similarity:**
- Sentence A: Vector stores retrieve documents by embedding similarity.
- Sentence B: The weather is sunny at the beach.
- Tại sao khác: hai câu thuộc hai chủ đề hoàn toàn khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector thay vì độ lớn tuyệt đối, nên phù hợp hơn khi cần đo độ gần về ngữ nghĩa. Với text embeddings, hai câu có thể có độ dài hoặc cường độ vector khác nhau nhưng vẫn cùng ý nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
Công thức: `ceil((doc_length - overlap) / (chunk_size - overlap))`  
Tính: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`  
**Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Tính: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, nên số chunk tăng từ 23 lên 25. Overlap nhiều hơn giúp giữ ngữ cảnh giữa hai chunk liền kề, nhưng đổi lại tốn thêm lưu trữ và thời gian retrieval.

---

## 2. Document Selection - Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Internal knowledge assistant / RAG documentation.

**Tại sao nhóm chọn domain này?**  
Bộ tài liệu mẫu đã có cấu trúc rõ, gồm Python, vector store, RAG, support playbook, chunking experiment và ghi chú tiếng Việt. Domain này phù hợp với lab vì có đủ nội dung để kiểm tra chunking, metadata filtering và retrieval quality.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `python_intro.txt` | `data/python_intro.txt` | 1953 | `category=python`, `language=en`, `source=data/python_intro.txt` |
| 2 | `vector_store_notes.md` | `data/vector_store_notes.md` | 2149 | `category=vector_store`, `language=en`, `source=data/vector_store_notes.md` |
| 3 | `rag_system_design.md` | `data/rag_system_design.md` | 2416 | `category=rag`, `language=en`, `source=data/rag_system_design.md` |
| 4 | `customer_support_playbook.txt` | `data/customer_support_playbook.txt` | 1703 | `category=support`, `language=en`, `source=data/customer_support_playbook.txt` |
| 5 | `chunking_experiment_report.md` | `data/chunking_experiment_report.md` | 2008 | `category=chunking`, `language=en`, `source=data/chunking_experiment_report.md` |
| 6 | `vi_retrieval_notes.md` | `data/vi_retrieval_notes.md` | 1678 | `category=retrieval_vi`, `language=vi`, `source=data/vi_retrieval_notes.md` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | `data/vector_store_notes.md` | Cho biết chunk đến từ tài liệu nào để kiểm tra nguồn. |
| `category` | string | `vector_store`, `rag`, `support` | Giúp lọc theo chủ đề khi query có phạm vi rõ. |
| `language` | string | `en`, `vi` | Giúp tránh lấy nhầm tài liệu tiếng Anh khi hỏi bằng tiếng Việt hoặc ngược lại. |
| `doc_id` | string | `python_intro` | Giúp xóa toàn bộ chunk thuộc cùng một tài liệu. |

---

## 3. Chunking Strategy - Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=500` trên 3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `python_intro.txt` | FixedSizeChunker | 5 | 428.8 | Trung bình, có thể cắt giữa câu |
| `python_intro.txt` | SentenceChunker | 5 | 387.0 | Tốt, giữ biên câu |
| `python_intro.txt` | RecursiveChunker | 5 | 387.0 | Tốt, giữ đoạn/câu |
| `vector_store_notes.md` | FixedSizeChunker | 5 | 464.6 | Trung bình |
| `vector_store_notes.md` | SentenceChunker | 8 | 263.6 | Dễ đọc nhưng nhiều chunk hơn |
| `vector_store_notes.md` | RecursiveChunker | 7 | 301.4 | Cân bằng giữa kích thước và ngữ cảnh |
| `chunking_experiment_report.md` | FixedSizeChunker | 5 | 437.4 | Trung bình |
| `chunking_experiment_report.md` | SentenceChunker | 5 | 395.6 | Tốt |
| `chunking_experiment_report.md` | RecursiveChunker | 5 | 395.6 | Tốt |

### Strategy Của Tôi

**Loại:** RecursiveChunker + metadata filter.

**Mô tả cách hoạt động:**  
Strategy của tôi dùng `RecursiveChunker(chunk_size=500)`. Thuật toán ưu tiên tách theo đoạn văn `\n\n`, sau đó theo dòng, câu, khoảng trắng và cuối cùng mới cắt theo ký tự nếu không còn separator phù hợp. Mỗi chunk được lưu vào `EmbeddingStore` kèm metadata gồm `source`, `category`, `language`, `doc_id`.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Tài liệu của domain này là markdown/text kỹ thuật, thường có paragraph, heading và các cụm ý hoàn chỉnh. Recursive chunking phù hợp vì nó giữ cấu trúc tự nhiên tốt hơn fixed-size, nhưng vẫn kiểm soát được kích thước chunk cho embedding.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `vector_store_notes.md` | FixedSizeChunker | 5 | 464.6 | Có thể gom nhiều ý vào một chunk lớn |
| `vector_store_notes.md` | RecursiveChunker của tôi | 7 | 301.4 | Chunk ngắn hơn, dễ kiểm tra nguồn và ngữ cảnh hơn |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi - Vũ Tuấn Phương | RecursiveChunker + metadata | [Điền sau khi nhóm so sánh] | Giữ cấu trúc đoạn, dễ filter | Với mock embedding, điểm similarity chưa phản ánh ngữ nghĩa thật |
| [Tên] | [Điền sau] | [Điền sau] | [Điền sau] | [Điền sau] |
| [Tên] | [Điền sau] | [Điền sau] | [Điền sau] | [Điền sau] |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
Tạm thời tôi chọn RecursiveChunker là strategy tốt nhất cho tài liệu kỹ thuật hỗn hợp vì nó tách theo cấu trúc tự nhiên trước, sau đó mới cắt nhỏ khi cần. Sau khi nhóm chạy cùng 5 benchmark queries, phần này sẽ được cập nhật bằng điểm so sánh thực tế của từng thành viên.

---

## 4. My Approach - Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk` - approach:**  
Tôi dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách câu sau dấu `.`, `!`, `?` khi có whitespace hoặc xuống dòng. Sau đó tôi strip whitespace, bỏ câu rỗng và gom tối đa `max_sentences_per_chunk` câu vào một chunk.

**`RecursiveChunker.chunk` / `_split` - approach:**  
Tôi dùng base case là text rỗng hoặc text đã ngắn hơn `chunk_size`. Nếu chunk còn quá dài, hàm `_split` thử separator theo thứ tự ưu tiên `\n\n`, `\n`, `. `, khoảng trắng, rồi fallback cắt theo `chunk_size`. Khi một piece nhỏ vừa đủ, tôi gom vào buffer để tránh tạo quá nhiều chunk vụn.

### EmbeddingStore

**`add_documents` + `search` - approach:**  
Mỗi `Document` được chuẩn hóa thành record gồm `id`, `content`, `metadata`, `embedding`. Search tạo embedding cho query rồi tính dot product với từng record, sắp xếp score giảm dần và trả về top-k kết quả.

**`search_with_filter` + `delete_document` - approach:**  
`search_with_filter` lọc metadata trước, sau đó mới search trên tập ứng viên đã lọc để tránh top-k bị nhiễu. `delete_document` xóa tất cả record có `metadata["doc_id"]` trùng với tài liệu cần xóa và trả về `True` nếu có record bị xóa.

### KnowledgeBaseAgent

**`answer` - approach:**  
Agent retrieve top-k chunk từ store, xây prompt gồm question và retrieved context có source, score, content. Prompt yêu cầu LLM chỉ trả lời dựa trên context và nói rõ nếu tài liệu không đủ thông tin.

### Test Results

`pytest` chưa có trong runtime bundled, nên tôi chạy trực tiếp bộ test bằng `unittest`:

```text
python -m unittest tests.test_solution -v
Ran 42 tests in 0.022s
OK
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions - Cá nhân (5 điểm)

Tôi dùng `_mock_embed` mặc định của lab để embed câu, sau đó gọi `compute_similarity()`.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is used for data analysis and machine learning. | Python supports machine learning and data workflows. | high | -0.0966 | Không |
| 2 | Vector stores rank embeddings by similarity. | A vector database retrieves similar embeddings. | high | 0.0546 | Một phần |
| 3 | Support teams escalate unresolved billing issues. | Password recovery requires verifying the account. | medium | -0.0719 | Không |
| 4 | Recursive chunking preserves paragraph structure. | Bananas ripen faster in warm kitchens. | low | -0.2215 | Đúng |
| 5 | Metadata filters narrow retrieval results. | The weather is sunny at the beach. | low | 0.0219 | Đúng một phần |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Kết quả bất ngờ nhất là hai câu về Python có score âm dù về nghĩa rất gần nhau. Nguyên nhân là `_mock_embed` trong lab là embedding giả lập deterministic để test code, không phải semantic embedding thật, nên nó kiểm tra pipeline tốt nhưng không phản ánh đầy đủ ý nghĩa ngôn ngữ.

---

## 6. Results - Cá nhân (10 điểm)

Chạy 5 benchmark queries trên implementation cá nhân. Tôi dùng `RecursiveChunker(chunk_size=500)`, `_mock_embed`, và lưu 34 chunks vào `EmbeddingStore`.

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Python thường được dùng cho những loại ứng dụng nào? | Python được dùng cho automation, backend services, data analysis, scientific computing, machine learning, APIs, data pipelines, internal tools và model-serving. |
| 2 | Vector store hoạt động theo pipeline mấy bước? | Pipeline có 4 bước: chunk documents, embed chunks, store vector + metadata, embed query và rank theo similarity. |
| 3 | Vì sao metadata quan trọng trong retrieval? | Metadata giúp lọc theo source, language, department/category, date hoặc access level để tăng precision và giảm nhiễu. |
| 4 | Recursive chunking tốt hơn fixed-size trong trường hợp nào? | Nó tốt với tài liệu có cấu trúc đoạn/heading vì giữ ngữ cảnh tự nhiên và chỉ tách nhỏ hơn khi đoạn quá dài. |
| 5 | Khi retrieval không đủ thông tin, assistant nên làm gì? | Assistant nên nói rõ là context không đủ hoặc đề xuất escalation, không tự bịa câu trả lời. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Python applications data machine learning APIs | Chunking report kết luận recursive chunking là default tốt cho mixed technical documentation | 0.2285 | Không | Dựa trên context nhưng chưa đúng gold answer |
| 2 | vector store pipeline chunk embed store query | Python production dùng APIs, data pipelines, internal tools, model-serving | 0.2686 | Một phần | Có nhắc data pipelines nhưng thiếu pipeline vector store |
| 3 | metadata filters retrieval department language source | Fixed-size chunking tạo chunk dự đoán được nhưng có thể cắt awkward | 0.2889 | Không | Không trả lời đúng vai trò metadata |
| 4 | recursive chunking fixed size paragraph context | Customer support playbook mô tả use case support assistant | 0.2114 | Không | Không nêu đúng so sánh recursive/fixed-size |
| 5 | insufficient retrieval assistant escalation uncertainty | Metadata phân biệt customer-facing, support-only, engineering-only documents | 0.2820 | Một phần | Có liên quan đến guardrail nhưng chưa trực tiếp nói về insufficient retrieval |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5 với `_mock_embed`.

**Nhận xét:**  
Kết quả retrieval với `_mock_embed` chưa tốt về mặt ngữ nghĩa vì mock embedding không hiểu semantic similarity thật. Tuy nhiên pipeline code đã chạy đúng: chunk, embed, store, search, filter và delete đều hoạt động. Nếu dùng `LocalEmbedder(all-MiniLM-L6-v2)` hoặc OpenAI embeddings, tôi kỳ vọng kết quả semantic retrieval sẽ tốt hơn rõ rệt.

---

## 7. What I Learned (5 điểm - Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
[Điền sau demo nhóm] Tôi sẽ so sánh strategy của mình với strategy của các bạn, đặc biệt là xem fixed-size, sentence-based hoặc custom chunking có query nào làm tốt hơn recursive hay không.

**Điều hay nhất tôi học được từ nhóm khác qua demo:**  
[Điền sau demo lớp] Tôi sẽ chú ý cách nhóm khác chọn metadata và benchmark queries, vì retrieval quality phụ thuộc nhiều vào chất lượng dữ liệu và câu hỏi đánh giá.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Tôi sẽ ưu tiên dùng semantic embedding thật thay vì `_mock_embed` khi đánh giá chất lượng retrieval. Ngoài ra, tôi sẽ thiết kế thêm metadata chi tiết hơn như `audience`, `document_type`, `freshness` để filter chính xác hơn và giảm failure case.

### Failure Analysis

**Failure case:** Query về pipeline vector store không retrieve đúng chunk từ `vector_store_notes.md` ở top-1 khi dùng `_mock_embed`.  
**Nguyên nhân:** Mock embedding tạo vector giả lập theo hash, nên câu hỏi có cùng chủ đề chưa chắc gần nhau trong không gian vector.  
**Cải thiện:** Dùng local semantic embedder hoặc OpenAI embedder, đồng thời filter `category=vector_store` cho query đã biết rõ domain.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | [Điền sau demo] / 5 |
| **Tổng tạm tính** | | **79 / 95 + demo** |
