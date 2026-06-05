# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Hoàng Hải 
**Nhóm:** AI Knowledge Assistant  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector embedding có hướng gần giống nhau, tức là hai đoạn văn có ý nghĩa hoặc ngữ cảnh tương tự nhau. Điểm càng gần 1 thì nội dung càng giống về mặt biểu diễn vector.

**Ví dụ HIGH similarity:**
- Sentence A: Python is used for data analysis and machine learning.
- Sentence B: Python supports machine learning, APIs, and data workflows.
- Tại sao tương đồng: Cả hai câu đều nói về Python và các ứng dụng trong dữ liệu, AI hoặc hệ thống phần mềm.

**Ví dụ LOW similarity:**
- Sentence A: Metadata filters can reduce noisy retrieval results.
- Sentence B: The cake should bake for thirty minutes.
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau: retrieval trong AI và nấu ăn.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector, nên phù hợp để đo mức độ giống nhau về ý nghĩa thay vì độ lớn tuyệt đối của vector. Với text embeddings, hướng vector thường quan trọng hơn khoảng cách thô vì nhiều embedding đã được chuẩn hóa hoặc có độ lớn không phản ánh trực tiếp ý nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`  
> `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`  
> Đáp án: **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, nên số chunk tăng từ 23 lên **25 chunks**. Overlap nhiều hơn giúp giữ ngữ cảnh giữa hai chunk liên tiếp, nhưng đổi lại tốn thêm dung lượng lưu trữ và chi phí embedding/search.

---

## 2. Document Selection - Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI knowledge assistant, vector store, chunking, RAG và tài liệu hỗ trợ nội bộ.

**Tại sao nhóm chọn domain này?**
> Domain này phù hợp với mục tiêu lab vì có nhiều tài liệu ngắn, có cấu trúc rõ, và liên quan trực tiếp đến embedding, vector store, metadata filtering và RAG. Bộ tài liệu cũng có cả tiếng Anh và tiếng Việt, giúp quan sát thêm ảnh hưởng của ngôn ngữ và metadata đến retrieval.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro.txt | `data/python_intro.txt` | 1944 | `source`, `extension`, `topic=python`, `language=en` |
| 2 | vector_store_notes.md | `data/vector_store_notes.md` | 2123 | `source`, `extension`, `topic=vector_store`, `language=en` |
| 3 | rag_system_design.md | `data/rag_system_design.md` | 2391 | `source`, `extension`, `topic=rag`, `language=en` |
| 4 | customer_support_playbook.txt | `data/customer_support_playbook.txt` | 1692 | `source`, `extension`, `topic=support`, `language=en` |
| 5 | chunking_experiment_report.md | `data/chunking_experiment_report.md` | 1987 | `source`, `extension`, `topic=chunking`, `language=en` |
| 6 | vi_retrieval_notes.md | `data/vi_retrieval_notes.md` | 1667 | `source`, `extension`, `topic=retrieval`, `language=vi` |       

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | `data/vector_store_notes.md` | Giúp truy vết câu trả lời về tài liệu gốc. |
| `extension` | string | `.md`, `.txt` | Phân biệt định dạng tài liệu để xử lý hoặc debug. |
| `topic` | string | `rag`, `chunking`, `support` | Giúp filter theo chủ đề khi query đã có phạm vi rõ. |
| `language` | string | `en`, `vi` | Hữu ích khi query hoặc người dùng yêu cầu tài liệu theo ngôn ngữ. |

---

## 3. Chunking Strategy - Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=200` trên 3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt | FixedSizeChunker (`fixed_size`) | 11 | 194.9 | Trung bình, có thể cắt giữa câu |
| python_intro.txt | SentenceChunker (`by_sentences`) | 5 | 387.0 | Tốt, giữ câu hoàn chỉnh nhưng chunk hơi dài |
| python_intro.txt | RecursiveChunker (`recursive`) | 14 | 137.6 | Tốt, ưu tiên đoạn/câu rồi mới cắt nhỏ |
| vector_store_notes.md | FixedSizeChunker (`fixed_size`) | 12 | 195.2 | Trung bình |
| vector_store_notes.md | SentenceChunker (`by_sentences`) | 8 | 263.6 | Khá tốt |
| vector_store_notes.md | RecursiveChunker (`recursive`) | 18 | 116.4 | Tốt nhất cho tài liệu có heading/đoạn |
| rag_system_design.md | FixedSizeChunker (`fixed_size`) | 14 | 189.4 | Trung bình |
| rag_system_design.md | SentenceChunker (`by_sentences`) | 5 | 476.0 | Giữ nghĩa tốt nhưng chunk dài |
| rag_system_design.md | RecursiveChunker (`recursive`) | 21 | 112.3 | Tốt, chunk nhỏ và có cấu trúc |

### Strategy Của Tôi

**Loại:** RecursiveChunker với `chunk_size=200`.

**Mô tả cách hoạt động:**
> Strategy này thử tách văn bản theo thứ tự ưu tiên: đoạn văn (`\n\n`), dòng (`\n`), câu (`. `), khoảng trắng, rồi cuối cùng mới cắt theo ký tự nếu không còn separator phù hợp. Nếu một phần vẫn dài hơn `chunk_size`, hàm `_split` tiếp tục đệ quy với separator nhỏ hơn. Cách này giúp chunk giữ cấu trúc tự nhiên của tài liệu tốt hơn so với cắt cứng theo số ký tự.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Bộ tài liệu có nhiều markdown heading, đoạn văn giải thích và danh sách workflow, nên recursive chunking phù hợp vì ưu tiên giữ các khối nội dung có nghĩa. Strategy này cân bằng giữa việc không làm chunk quá dài và không cắt ngang ý quan trọng.

**Code snippet (nếu custom):**
```python
# Không dùng custom chunker. Tôi chọn RecursiveChunker có sẵn:
chunker = RecursiveChunker(chunk_size=200)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| chunking_experiment_report.md | FixedSizeChunker baseline | 11 | 198.8 | Ổn định nhưng đôi lúc cắt ngang câu |
| chunking_experiment_report.md | **RecursiveChunker của tôi** | 18 | 108.9 | Tốt hơn để inspect vì chunk ngắn và rõ ý |
| customer_support_playbook.txt | FixedSizeChunker baseline | 10 | 187.2 | Có thể lẫn nhiều ý trong một chunk |
| customer_support_playbook.txt | **RecursiveChunker của tôi** | 14 | 119.5 | Phù hợp hơn cho nội dung support nhiều bước |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Hoàng Hải | RecursiveChunker, `chunk_size=200` | 5 | Giữ cấu trúc đoạn tốt, chunk dễ đọc | Tạo nhiều chunk hơn; với `_mock_embed` điểm retrieval chưa cao |
| Thành viên 2 | SentenceChunker, 3 câu/chunk | 6 | Giữ câu hoàn chỉnh, dễ đọc thủ công | Chunk có thể quá dài và chứa nhiều ý |
| Thành viên 3 | FixedSizeChunker, `chunk_size=200`, `overlap=20` | 6 | Dễ kiểm soát kích thước, chunk count ổn định | Dễ cắt ngang câu hoặc ý đang giải thích |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Nếu chỉ nhìn điểm retrieval với `_mock_embed`, FixedSizeChunker và SentenceChunker đang nhỉnh hơn một chút với 6/10. Tuy nhiên khi inspect thủ công, RecursiveChunker vẫn có chunk coherence tốt nhất vì giữ heading, đoạn văn và các ý nhỏ rõ hơn. Nhóm kết luận rằng với embedding semantic thật, RecursiveChunker là ứng viên tốt nhất để thử tiếp; còn với mock embedding, điểm số chủ yếu dùng để kiểm tra pipeline và so sánh tương đối.

---

## 4. My Approach - Cá Nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk` - approach:**
> Tôi dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách câu sau các dấu `.`, `!`, `?` kèm khoảng trắng hoặc xuống dòng. Sau đó tôi `strip()` từng câu, bỏ câu rỗng, rồi gom tối đa `max_sentences_per_chunk` câu thành một chunk.

**`RecursiveChunker.chunk` / `_split` - approach:**
> Hàm `chunk` xử lý các base case như text rỗng hoặc text ngắn hơn `chunk_size`. Hàm `_split` thử separator theo thứ tự ưu tiên; nếu một piece vẫn quá dài thì gọi đệ quy với separator tiếp theo, và nếu hết separator thì fallback sang cắt theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search` - approach:**
> `add_documents` tạo record chuẩn gồm `id`, `doc_id`, `content`, `metadata`, và `embedding`, sau đó lưu vào list in-memory. `search` embed query, tính dot product giữa query embedding và từng document embedding, sắp xếp score giảm dần rồi trả về top-k.

**`search_with_filter` + `delete_document` - approach:**
> `search_with_filter` lọc metadata trước, sau đó mới search trên tập candidate đã lọc để tránh lấy kết quả ngoài phạm vi. `delete_document` xóa tất cả record có `metadata["doc_id"]` trùng với document cần xóa và trả về `True` nếu có record bị xóa.

### KnowledgeBaseAgent

**`answer` - approach:**
> Agent gọi `store.search(question, top_k)` để lấy context liên quan, đánh số từng chunk trong prompt, rồi gọi `llm_fn(prompt)`. Prompt có cấu trúc gồm instruction, context, question và phần `Answer:` để mô phỏng RAG pattern cơ bản.

### Test Results

```text
pytest tests/ -q
..........................................                               [100%]
42 passed, 140 warnings in 2.68s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions - Cá Nhân (5 điểm)

Sử dụng `_mock_embed` và `compute_similarity()`.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is used for data analysis and machine learning. | Python supports machine learning, APIs, and data workflows. | high | 0.065 | Không |
| 2 | A vector store retrieves similar embeddings. | Vector databases rank documents by similarity. | high | 0.066 | Không |
| 3 | Support agents should escalate unresolved billing issues. | Employees should inspect retrieved chunks for traceability. | low | -0.072 | Có |
| 4 | Metadata filters can reduce noisy retrieval results. | The cake should bake for thirty minutes. | low | -0.071 | Có |
| 5 | RAG answers should be grounded in retrieved context. | A retrieval system should avoid hallucination by using evidence. | high | 0.008 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Các cặp 1, 2 và 5 đáng lẽ giống nhau về nghĩa nhưng score thực tế vẫn thấp. Nguyên nhân là lab đang dùng `_mock_embed`, một embedding deterministic để test code chứ không phải model semantic thật, nên kết quả similarity không phản ánh đầy đủ ý nghĩa ngôn ngữ. Điều này cho thấy cần phân biệt giữa kiểm tra pipeline chạy đúng và đánh giá retrieval quality thật.

---

## 6. Results - Benchmark Cá Nhân & Nhóm (10 điểm)

Nhóm chạy cùng 5 benchmark queries trên bộ tài liệu trong `data/`. Mỗi strategy index các chunk riêng rồi search top-3; riêng query 4 dùng metadata filter `topic=support` để kiểm tra tác dụng của filtering.

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What are common uses of Python in AI systems? | Python được dùng cho automation, backend, data analysis, scientific computing, machine learning, và trong RAG để kết nối embedding models, vector stores và application logic. |
| 2 | What are the four stages of a vector search pipeline? | Chunk documents, embed each chunk, store vector and metadata, embed the query and rank stored vectors by similarity. |
| 3 | How should a RAG assistant reduce hallucinations? | RAG nên retrieve tài liệu liên quan trước, inject context vào prompt, yêu cầu model trả lời dựa trên evidence, và nói rõ khi context yếu hoặc mâu thuẫn. |
| 4 | For customer support content, why should metadata filtering separate public and internal notes? | Metadata giúp phân biệt customer-facing, internal support-only và engineering-only documents, tránh trả nhầm nội dung nội bộ hoặc sai audience. |
| 5 | What failure cases should teams record when evaluating retrieval quality? | Teams nên ghi lại failure cases như outdated documents outranking current runbooks, small chunks losing caveats, multilingual confusion, poor chunk boundaries và stale articles. |

### Kết Quả So Sánh Trong Nhóm

| Strategy | Chunk Count | Top-3 Relevant | Retrieval Score (/10) | Nhận xét |
|----------|-------------|----------------|-----------------------|----------|
| FixedSizeChunker | 68 | 4 / 5 | 6 / 10 | Điểm tốt nhất/tied, nhưng một số chunk bị cắt giữa câu. |
| SentenceChunker | 32 | 4 / 5 | 6 / 10 | Ít chunk nhất và dễ đọc, nhưng chunk dài nên có thể lẫn nhiều ý. |
| RecursiveChunker | 98 | 3 / 5 | 5 / 10 | Chunk rõ ý nhất khi inspect thủ công, nhưng `_mock_embed` làm score không ổn định. |

### Metadata Filtering Check

| Strategy | Query 4 unfiltered top-3 | Query 4 filtered top-3 |
|----------|--------------------------|------------------------|
| FixedSizeChunker | support, python, vector_store | support, support, support |
| SentenceChunker | rag, vector_store, python | support, support, support |
| RecursiveChunker | rag, python, retrieval | support, support, support |

**Kết luận metadata:** Filter `topic=support` giúp query 4 rõ hơn rất nhiều. Khi không filter, SentenceChunker và RecursiveChunker retrieve nhầm sang RAG/Python/Retrieval; khi filter trước, top-3 đều đến từ `customer_support_playbook.txt`.

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What are common uses of Python in AI systems? | `customer_support_playbook`: support team uses knowledge assistant | 0.306 | Không | Context không đúng trọng tâm Python |
| 2 | What are the four stages of a vector search pipeline? | `chunking_experiment_report`: sentence-based chunking | 0.305 | Một phần | Top-3 có `vector_store_notes`, nhưng top-1 chưa đúng |
| 3 | How should a RAG assistant reduce hallucinations? | `rag_system_design`: proposed architecture và grounding | 0.274 | Có | Nêu retrieve context trước và answer dựa trên evidence |
| 4 | For customer support content, why should metadata filtering separate public and internal notes? | `customer_support_playbook`: failed queries, missing docs, stale articles | 0.184 | Có | Nêu filter giúp tránh lộ internal-only procedures |
| 5 | What failure cases should teams record when evaluating retrieval quality? | `chunking_experiment_report`: chunk sizes and dense sections | 0.280 | Không | Context nói về chunking, chưa đủ failure cases của RAG |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

**Nhận xét ngắn:**
> Strategy cá nhân của tôi là RecursiveChunker. Kết quả top-3 relevant đạt 3/5 và retrieval score đạt 5/10. Điểm yếu chính không nằm ở pipeline mà ở `_mock_embed`: nó deterministic để test code, nhưng không hiểu semantic tốt như embedding thật, nên một số query cùng chủ đề vẫn retrieve nhầm.

---

## 7. What I Learned (5 điểm - Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được rằng cùng một bộ tài liệu nhưng thay đổi chunking strategy có thể làm kết quả retrieval khác rõ rệt. SentenceChunker có số chunk ít và dễ đọc, FixedSizeChunker có điểm mock retrieval khá ổn, còn RecursiveChunker dễ inspect nhất vì chunk bám theo cấu trúc tài liệu.

**Điều hay nhất tôi học được từ nhóm khác qua demo:**
> Tôi học được rằng metadata filtering rất quan trọng khi tài liệu có nhiều nhóm người dùng hoặc nhiều loại quyền truy cập. Trong benchmark của nhóm, query support là ví dụ rõ nhất: không filter thì top-3 có thể lệch sang RAG/Python, còn filter `topic=support` thì cả top-3 đều đúng domain.

**Failure case nhóm tìm được:**
> Query 5 về failure cases retrieve nhầm `chunking_experiment_report` thay vì `rag_system_design`. Nguyên nhân có thể là query có các từ "failure", "evaluating", "retrieval quality" trùng với nội dung chunking experiment, trong khi gold answer nằm trong phần evaluation/failure cases của RAG design. Cách cải thiện là thêm metadata `section=evaluation`, dùng embedding semantic thật, và chunk theo heading để giữ đoạn "Evaluation Plan" rõ hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ gán metadata rõ hơn cho từng tài liệu, ví dụ `topic`, `audience`, `language`, `doc_type` và `last_updated`. Tôi cũng sẽ chunk tài liệu thành các đoạn nhỏ trước khi add vào vector store thay vì index toàn bộ file như một document, vì retrieval theo chunk sẽ sát câu hỏi hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 6 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **82 / 100** |
