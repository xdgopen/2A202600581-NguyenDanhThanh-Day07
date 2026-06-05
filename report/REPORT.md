# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Danh Thành
**Nhóm:** MoMo FAQ Retrieval
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
Hai text chunks có high cosine similarity nghĩa là vector embedding của chúng trỏ gần cùng hướng, tức là nội dung/ngữ nghĩa có liên quan mạnh.

**Ví dụ HIGH similarity:**
- Sentence A: MoMo hỗ trợ chuyển tiền nhanh.
- Sentence B: Người dùng có thể chuyển và nhận tiền tức thì qua MoMo.
- Tại sao tương đồng: Cả hai đều nói về chức năng chuyển/nhận tiền của MoMo.

**Ví dụ LOW similarity:**
- Sentence A: Phí rút tiền vượt 30 triệu là 0,5%.
- Sentence B: Mua vé xem phim cần chọn ghế và suất chiếu.
- Tại sao khác: Hai câu thuộc hai nhu cầu khác nhau: phí giao dịch và mua vé xem phim.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector nên đo được độ giống về nghĩa tốt hơn, ít bị ảnh hưởng bởi độ dài hoặc magnitude của embedding.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
`ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23` chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
`ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25` chunks. Overlap nhiều hơn làm tăng số chunk nhưng giúp giữ context ở ranh giới chunk tốt hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Customer support FAQ cho ví điện tử MoMo.

Nhóm chọn domain này vì FAQ có cấu trúc rõ theo chuyên mục và câu hỏi, phù hợp để thử retrieval theo RAG. Người dùng thường hỏi trực tiếp về phí, hạn mức, bảo mật, nạp/rút tiền hoặc thanh toán nên có thể tạo benchmark queries dễ kiểm chứng từ tài liệu.

### Data Inventory

File chính: `data/momo_faq.md` dài 5,293 ký tự. Khi chạy benchmark, file được parse thành 17 Q&A documents.

| # | Tên tài liệu/chunk | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------------|-------|----------|-----------------|
| 1 | MoMo là gì? | data/momo_faq.md | 550 | domain, language, category, question, source |
| 2 | Cách liên hệ với MoMo | data/momo_faq.md | 216 | domain, language, category, question, source |
| 3 | Tiền trong tài khoản MoMo có phải là tiền thật? | data/momo_faq.md | 300 | domain, language, category, question, source |
| 4 | Sử dụng MoMo có mất phí không? | data/momo_faq.md | 719 | domain, language, category, question, source |
| 5 | Hạn mức giao dịch mỗi ngày? | data/momo_faq.md | 544 | domain, language, category, question, source |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | data/momo_faq.md | Biết chunk đến từ file nào để cite/debug |
| domain | string | momo_faq | Tách dữ liệu MoMo khỏi domain khác |
| language | string | vi | Lọc theo ngôn ngữ |
| category | string | Quản lý Tài khoản & Bảo mật | Filter theo nhóm nội dung |
| question | string | Cách liên hệ với MoMo | Dùng làm nhãn chunk và kiểm tra relevance |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên `data/momo_faq.md` với `chunk_size=500`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| momo_faq.md | FixedSizeChunker (`fixed_size`) | 12 | 486.9 | Trung bình, dễ cắt ngang câu/FAQ |
| momo_faq.md | SentenceChunker (`by_sentences`) | 27 | 194.5 | Tốt ở mức câu, nhưng có thể tách câu hỏi khỏi câu trả lời |
| momo_faq.md | RecursiveChunker (`recursive`) | 17 | 309.6 | Tốt hơn vì ưu tiên paragraph/header boundaries |

### Strategy Của Nguyễn Danh Thành - 2A202600581

**Loại:** custom FAQ Q&A chunking.

**Mô tả cách hoạt động:**  
Strategy parse file Markdown theo heading `##` làm `category` và heading `###` làm `question`. Mỗi chunk gồm một câu hỏi và toàn bộ phần trả lời ngay dưới câu hỏi đó. Nhờ vậy chunk giữ nguyên ý hỏi-đáp, không bị cắt giữa quy trình hoặc giữa bảng phí.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
FAQ MoMo có cấu trúc tự nhiên theo Q&A. Khi người dùng hỏi, intent thường khớp trực tiếp với một câu hỏi trong FAQ, nên chunk theo từng Q&A giúp retrieval chính xác và dễ giải thích hơn fixed-size chunking.

**Code snippet:** xem `parse_momo_faq()` trong `momo_benchmark.py`.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| momo_faq.md | best baseline: RecursiveChunker | 17 | 309.6 | Khá tốt, nhưng metadata theo câu hỏi chưa rõ |
| momo_faq.md | của tôi: Q&A chunking | 17 | khoảng 311.4 | Tốt hơn cho FAQ vì mỗi chunk có `category` và `question` |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Danh Thành - 2A202600581 | FixedSizeChunker | 8 | Top-3 tốt, đơn giản | Chunk có thể chứa nhiều FAQ, gây nhiễu context |
| Vũ Ngọc Vinh - 2A202600864 | FixedSizeChunker large overlap | 5 | Giữ context rộng, ít chunk | Chunk quá dài nên top-1/grounding dễ lệch |
| Vũ Hải Dương - 2A202600632 | SentenceChunker | 5 | Chunk ngắn, dễ đọc từng câu | Dễ tách câu hỏi khỏi câu trả lời, filter kém |
| Hoàng Hải - 2A202600948 | RecursiveChunker | 8 | Giữ paragraph/header tốt hơn sentence | Một số top-1 vẫn lệch vì chunk chứa nhiều ý |
| Vũ Thành Lộc - 2A202600684 | CategoryChunker (`by_category`) | 7 | Recall rộng, top-3 tốt | Chunk quá dài, grounding kém vì chứa nhiều Q&A |
| Vũ Tuấn Phương - 2A202600772 | Q&A chunking + metadata filter | 10 | Giữ trọn câu hỏi/trả lời, filter tốt | Phụ thuộc cấu trúc Markdown chuẩn |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
Q&A chunking phù hợp nhất vì tài liệu là FAQ. Khi chạy cùng 5 benchmark queries, `qa_custom` đạt top-1 5/5, top-3 5/5 và grounding 5/5. Các strategy chunk rộng như `fixed_size_large` và `by_category` cho thấy giữ nhiều context chưa chắc tốt hơn vì agent dễ lấy nhầm phần không trực tiếp trả lời câu hỏi.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk` — approach:**  
Dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách theo ranh giới câu sau `.`, `!`, `?`, sau đó gom theo `max_sentences_per_chunk`. Với input rỗng hoặc chỉ whitespace, hàm trả về list rỗng.

**`RecursiveChunker.chunk` / `_split` — approach:**  
Base case là text đã nhỏ hơn `chunk_size`. Nếu quá lớn, thuật toán thử separator theo thứ tự `\n\n`, `\n`, `. `, space, rồi fallback cắt theo character khi không còn separator.

### EmbeddingStore

**`add_documents` + `search` — approach:**  
Mỗi `Document` được normalize thành record gồm `id`, `content`, `metadata`, `embedding`. Search embed query rồi tính dot product với các record, sort giảm dần theo score và trả về top-k.

**`search_with_filter` + `delete_document` — approach:**  
`search_with_filter` lọc metadata trước rồi mới search để tránh top-k bị nhiễu. `delete_document` xóa mọi record có `metadata["doc_id"]` trùng với document id cần xóa.

### KnowledgeBaseAgent

**`answer` — approach:**  
Agent gọi store search để lấy top-k chunks, đóng gói context theo dạng numbered sources, sau đó đưa vào prompt. Prompt yêu cầu trả lời dựa trên context và nói không đủ thông tin nếu context thiếu.

### Test Results

```text
42 passed in 0.02s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Embedding dùng cho phần này: `_mock_embed`, sau đó gọi `compute_similarity()`.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | MoMo hỗ trợ chuyển tiền nhanh. | Người dùng có thể chuyển và nhận tiền tức thì qua MoMo. | high | -0.076 | Không |
| 2 | MoMo có hotline chăm sóc khách hàng. | Email hỗ trợ của MoMo là hotro@momo.vn. | high | 0.198 | Tương đối |
| 3 | Phí rút tiền vượt 30 triệu là 0,5%. | Mua vé xem phim cần chọn ghế và suất chiếu. | low | 0.041 | Có |
| 4 | Gỡ ứng dụng không làm mất tiền. | Dữ liệu tài khoản được lưu trên máy chủ. | high | 0.168 | Tương đối |
| 5 | Tài khoản MoMo có thể liên kết ngân hàng. | Con mèo đang ngủ trên ghế sofa. | low | -0.102 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Pair 1 đáng lẽ rất giống nhau nhưng score âm vì `_mock_embed` là embedding giả deterministic, không hiểu ngữ nghĩa thật. Điều này cho thấy mock embedding phù hợp để test code path, còn muốn đánh giá retrieval semantic thật thì cần local/OpenAI embedding backend.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | MoMo là gì và có những chức năng chính nào? | MoMo là Trợ thủ Tài chính, hỗ trợ chuyển nhận tiền, thanh toán dịch vụ, quản lý chi tiêu, tín dụng, đầu tư và kinh doanh. |
| 2 | Liên hệ chăm sóc khách hàng MoMo qua những kênh nào? | Gửi yêu cầu ở mục Trợ giúp, gọi hotline 1900 5454 41 hoặc email hotro@momo.vn. |
| 3 | Phí rút tiền MoMo khi vượt 30 triệu trong tháng là bao nhiêu? | Vượt 30 triệu phí 0,5%; vượt 100 triệu phí 1%; dưới/đến 30 triệu miễn phí. |
| 4 | Gỡ ứng dụng MoMo thì tiền và lịch sử giao dịch có mất không? | Không mất; dữ liệu lưu trên hệ thống máy chủ và phục hồi khi cài lại. |
| 5 | Mua vé xem phim trên MoMo thực hiện các bước nào? | Chọn mua vé, chọn phim/suất chiếu, ghế, combo nếu có, kiểm tra thanh toán, quét QR/Barcode tại rạp. |

### Kết Quả Của Tôi

Chạy bằng `momo_benchmark.py` với Q&A chunking và offline hashing keyword embedder.

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | MoMo là gì và có những chức năng chính nào? | MoMo là gì? | 0.320 | Có | MoMo là Trợ thủ Tài chính, có chuyển/nhận tiền và các dịch vụ tài chính. |
| 2 | Liên hệ chăm sóc khách hàng MoMo qua những kênh nào? | Cách liên hệ với MoMo | 0.452 | Có | Trợ giúp trong app, hotline 1900 5454 41, email hotro@momo.vn. |
| 3 | Phí rút tiền MoMo khi vượt 30 triệu trong tháng là bao nhiêu? | Sử dụng MoMo có mất phí không? | 0.505 | Có | Miễn phí đến 30 triệu; vượt 30 triệu phí 0,5%; vượt 100 triệu phí 1%. |
| 4 | Gỡ ứng dụng MoMo thì tiền và lịch sử giao dịch có mất không? | Khi gỡ bỏ ứng dụng... | 0.518 | Có | Không mất, dữ liệu lưu trên server và phục hồi khi cài lại. |
| 5 | Mua vé xem phim trên MoMo thực hiện các bước nào? | Cách mua vé xem phim trên MoMo | 0.534 | Có | Chọn phim/suất chiếu, ghế, combo, thanh toán, quét mã tại rạp. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

### So Sánh Strategy Theo 5 Tiêu Chí Tự Đánh Giá

Chạy cùng bộ dữ liệu `data/momo_faq.md`, cùng 5 benchmark queries và cùng offline hashing keyword embedder.

**1. Retrieval Precision**

| Strategy | Chunks | Avg Length | Top-1 Hit | Top-3 Hit | Grounding | Avg Precision@3 |
|----------|--------|------------|-----------|-----------|-----------|-----------------|
| fixed_size | 12 | 486.9 | 4 / 5 | 5 / 5 | 4 / 5 | 0.33 |
| fixed_size_large | 7 | 859.0 | 4 / 5 | 4 / 5 | 1 / 5 | 0.27 |
| by_sentences | 27 | 194.5 | 2 / 5 | 3 / 5 | 1 / 5 | 0.20 |
| recursive | 17 | 309.6 | 3 / 5 | 5 / 5 | 3 / 5 | 0.33 |
| by_category | 4 | 1321.5 | 5 / 5 | 5 / 5 | 0 / 5 | 0.47 |
| qa_custom | 17 | 297.2 | 5 / 5 | 5 / 5 | 5 / 5 | 0.33 |

`qa_custom` tốt nhất vì câu trả lời đúng luôn nằm ở top-1 và đúng Q&A cụ thể. `by_category` có precision@3 cao do chunk rộng nhưng grounding thấp, nên không phải strategy tốt nhất cho agent answer.

**2. Chunk Coherence**

| Strategy | Coherence Score | Nhận xét |
|----------|-----------------|----------|
| fixed_size | 1.00 | Đủ dài và thường chứa heading, nhưng có thể chứa nhiều FAQ trong một chunk |
| fixed_size_large | 1.00 | Giữ nhiều context nhưng dễ trộn nhiều Q&A, làm answer thiếu tập trung |
| by_sentences | 0.52 | Chunk ngắn, dễ mất context và metadata theo câu hỏi |
| recursive | 0.76 | Giữ paragraph tốt hơn sentence, nhưng chưa đảm bảo mỗi chunk là một Q&A hoàn chỉnh |
| by_category | 1.00 | Giữ đủ category nhưng quá dài, chứa nhiều câu hỏi khác nhau |
| qa_custom | 1.00 | Mỗi chunk là một câu hỏi + câu trả lời hoàn chỉnh |

**3. Metadata Utility**

Query kiểm tra filter: `Gỡ ứng dụng MoMo thì tiền và lịch sử giao dịch có mất không?` với filter `category = Quản lý Tài khoản & Bảo mật`.

| Strategy | Unfiltered Hit | Filtered Hit | Nhận xét |
|----------|----------------|--------------|----------|
| fixed_size | True | True | Vẫn tìm đúng vì chunk chứa heading bảo mật |
| fixed_size_large | False | False | Chunk quá rộng làm score bị nhiễu sang nội dung tiền/tài khoản khác |
| by_sentences | True | False | Filter làm mất kết quả vì metadata/category bị suy yếu khi chunk quá nhỏ |
| recursive | True | True | Filter giữ được kết quả trong top-3 dù top-1 có thể là câu bảo mật khác |
| by_category | True | True | Filter chọn đúng category, nhưng chunk vẫn quá rộng |
| qa_custom | True | True | Filter hoạt động tốt nhất vì category được parse trực tiếp từ heading |

**4. Grounding Quality**

Agent answer của `qa_custom` bám trực tiếp top-1 retrieved chunk cho cả 5 query. Ví dụ query phí rút tiền trả đúng nội dung: miễn phí đến 30 triệu, vượt 30 triệu phí 0,5%, vượt 100 triệu phí 1%. Mỗi answer có thể trace lại field `question` trong metadata, nên dễ kiểm chứng nguồn.

**5. Data Strategy Impact**

MoMo FAQ có cấu trúc Markdown rất rõ: `##` là category và `###` là câu hỏi. Strategy tận dụng cấu trúc domain (`qa_custom`) cho kết quả tốt hơn generic chunkers vì retrieval unit khớp với cách người dùng hỏi. Nếu dữ liệu không còn heading chuẩn, cần fallback sang recursive chunking hoặc parser linh hoạt hơn.

**Failure case đáng chú ý:** `by_sentences` thất bại khi dùng metadata filter cho query gỡ ứng dụng. Nguyên nhân là sentence chunking tạo nhiều chunk nhỏ, có chunk chứa câu trả lời nhưng không giữ đầy đủ heading/category hoặc câu hỏi. `by_category` là failure case khác: top-3 có relevant chunk nhưng answer dễ lệch vì chunk quá dài. Cải thiện: chunk theo Q&A trước, sau đó mới split những Q&A quá dài.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
Chunk size không phải lúc nào càng nhỏ càng tốt. Với FAQ, giữ nguyên một cặp câu hỏi-trả lời thường hiệu quả hơn chia đều theo số ký tự.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
Metadata filter rất hữu ích khi query thuộc một category rõ, ví dụ bảo mật/tài khoản. Filter trước retrieval giúp giảm nhiễu và tăng khả năng agent dùng đúng context.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Tôi sẽ bổ sung nhiều file FAQ hơn từ các nhóm dịch vụ khác của MoMo và chuẩn hóa metadata như `product`, `intent`, `fee_related`, `requires_account_verified`. Nếu có mạng hoặc model cache, tôi sẽ chạy thêm semantic embedding thật để so sánh với keyword embedding.

---

## 8. Báo Cáo Nhóm Theo SCORING.md (40 điểm)

### 8.1 Strategy Design (15 điểm)

Nhóm 6 thành viên dùng cùng bộ dữ liệu `data/momo_faq.md`, cùng 5 benchmark queries và cùng offline hashing keyword embedder. Mỗi thành viên thử một strategy riêng để so sánh tác động của chunking, chunk size, overlap và metadata.

| Thành viên | Strategy | Mô tả | Rationale | Kết quả chính |
|------------|----------|-------|-----------|---------------|
| Nguyễn Danh Thành - 2A202600581 | `fixed_size` | Chunk 500 ký tự, overlap 50 | Baseline đơn giản, phù hợp văn bản dài | Top-1 4/5, top-3 5/5, grounding 4/5 |
| Vũ Ngọc Vinh - 2A202600864 | `fixed_size_large` | Chunk 900 ký tự, overlap 120 | Thử giữ context rộng hơn baseline | Top-1 4/5, top-3 4/5, grounding 1/5 |
| Vũ Hải Dương - 2A202600632 | `by_sentences` | Gom 3 câu mỗi chunk | Muốn chunk ngắn, dễ đọc | Top-1 2/5, top-3 3/5, grounding 1/5 |
| Hoàng Hải - 2A202600948 | `recursive` | Split theo `\n\n`, `\n`, `. `, space | Giữ paragraph/header tốt hơn fixed-size | Top-1 3/5, top-3 5/5, grounding 3/5 |
| Vũ Thành Lộc - 2A202600684 | `by_category` | Mỗi `##` category là một chunk | Tối ưu recall theo chuyên mục lớn | Top-1 5/5, top-3 5/5, grounding 0/5 |
| Vũ Tuấn Phương - 2A202600772 | `qa_custom` | Mỗi `###` Q&A là một chunk, kèm metadata category/question | Khai thác đúng cấu trúc FAQ | Top-1 5/5, top-3 5/5, grounding 5/5 |

**So sánh với baseline:**  
Baseline `fixed_size` khá mạnh vì FAQ ngắn và chunk 500 ký tự thường giữ được heading. `fixed_size_large` cho thấy overlap/context rộng hơn không tự động cải thiện retrieval: chunk dài hơn làm tăng nhiễu và giảm grounding. `qa_custom` tốt hơn vì mỗi chunk là một đơn vị hỏi-đáp hoàn chỉnh, metadata rõ và answer dễ trace về đúng câu hỏi.

**Kết luận strategy:**  
Strategy tốt nhất cho MoMo FAQ là `qa_custom`. `by_category` có recall rộng nhưng không phù hợp để generate answer vì chunk quá dài; `fixed_size_large` cũng bị nhiễu tương tự ở mức thấp hơn; `by_sentences` dễ mất context; `recursive` là fallback tốt nếu file Markdown không đủ chuẩn để parse Q&A.

### 8.2 Document Set Quality (10 điểm)

| Tiêu chí | Đánh giá |
|----------|----------|
| Chủ đề rõ ràng | FAQ chăm sóc khách hàng MoMo: tổng quan, bảo mật, ngân hàng, nạp/rút tiền, thanh toán |
| Kích thước dữ liệu | 1 file Markdown, 5,293 ký tự, parse thành 17 Q&A chunks |
| Nguồn minh bạch | `data/momo_faq.md` trong repo |
| Metadata hữu ích | `source`, `domain`, `language`, `category`, `question`, `strategy` |
| Phù hợp benchmark | Có đủ thông tin để trả lời 5 queries về chức năng, CSKH, phí, bảo mật, mua vé |

Điểm mạnh của document set là cấu trúc Markdown rõ ràng. Hạn chế là chỉ có 1 file thay vì 5-10 file riêng, nên nhóm mô phỏng document set bằng cách parse thành nhiều Q&A documents có metadata riêng.

### 8.3 Retrieval Quality (10 điểm)

Quy tắc theo `SCORING.md`: mỗi query tối đa 2 điểm. Nhóm chấm dựa trên top-3 có relevant chunk và agent answer có đúng/đủ thông tin không.

| Strategy | Q1 | Q2 | Q3 | Q4 | Q5 | Tổng /10 | Nhận xét |
|----------|----|----|----|----|----|----------|----------|
| `fixed_size` | 2 | 2 | 2 | 2 | 1 | 9 | Top-3 tốt, nhưng một số answer có nhiễu do chunk chứa nhiều nội dung |
| `fixed_size_large` | 1 | 1 | 2 | 0 | 1 | 5 | Context rộng nhưng nhiều nhiễu, query filter bảo mật thất bại |
| `by_sentences` | 1 | 1 | 1 | 1 | 0 | 4 | Chunk nhỏ làm mất context, metadata filter kém |
| `recursive` | 2 | 1 | 2 | 1 | 2 | 8 | Top-3 tốt nhưng top-1 đôi lúc lệch sang FAQ gần nghĩa |
| `by_category` | 1 | 1 | 1 | 1 | 1 | 5 | Relevant trong top-3 nhưng answer thiếu chính xác vì chunk quá rộng |
| `qa_custom` | 2 | 2 | 2 | 2 | 2 | 10 | Top-1 đúng cả 5 query, answer grounded theo đúng Q&A |

**Benchmark queries thống nhất của nhóm:**

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | MoMo là gì và có những chức năng chính nào? | MoMo là Trợ thủ Tài chính, hỗ trợ chuyển nhận tiền, thanh toán dịch vụ, quản lý chi tiêu, tín dụng, đầu tư và kinh doanh. |
| 2 | Liên hệ chăm sóc khách hàng MoMo qua những kênh nào? | Gửi yêu cầu ở mục Trợ giúp, gọi hotline 1900 5454 41 hoặc email hotro@momo.vn. |
| 3 | Phí rút tiền MoMo khi vượt 30 triệu trong tháng là bao nhiêu? | Vượt 30 triệu phí 0,5%; vượt 100 triệu phí 1%; dưới/đến 30 triệu miễn phí. |
| 4 | Gỡ ứng dụng MoMo thì tiền và lịch sử giao dịch có mất không? | Không mất; dữ liệu lưu trên hệ thống máy chủ và phục hồi khi cài lại. |
| 5 | Mua vé xem phim trên MoMo thực hiện các bước nào? | Chọn mua vé, chọn phim/suất chiếu, ghế, combo nếu có, kiểm tra thanh toán, quét QR/Barcode tại rạp. |

### 8.4 Demo (5 điểm)

**Thông điệp demo:**  
Với FAQ có heading rõ, strategy nên khai thác cấu trúc dữ liệu thay vì chỉ cắt theo độ dài. Nhóm demo bằng command:

```bash
./.venv/bin/python momo_benchmark.py
```

**Điểm cần trình bày:**
- `qa_custom` thắng vì chunk đúng đơn vị tri thức Q&A và metadata rõ.
- `by_category` chứng minh top-k relevance chưa đủ; chunk quá rộng làm grounding yếu.
- `fixed_size_large` cho thấy tăng chunk size/overlap có thể làm retrieval nhiễu hơn.
- `by_sentences` chứng minh chunk quá nhỏ làm mất context và filter có thể phản tác dụng.
- Metadata filter hữu ích nhất khi metadata được gán ở đúng cấp Q&A.

**Bài học nhóm:**  
Retrieval quality không chỉ là “có relevant chunk trong top-3”, mà còn là chunk có đủ nhỏ, đủ coherent và đủ traceable để agent trả lời đúng. Với domain FAQ, thiết kế data/chunking quan trọng không kém embedding model.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **85 / 100** |
