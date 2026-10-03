# Retrieval comparison

Configuration: database=postgresql, embedding_model=BAAI/bge-small-en-v1.5, reranker_model=cross-encoder/ms-marco-MiniLM-L6-v2, bm25={'k1': 1.2, 'b': 0.75, 'stemming': True, 'stopwords': True}, fusion=rrf, dense_weight=1.0, sparse_weight=1.0, rrf_k=60, fusion_depth=50, rerank_candidates=30, hnsw_ef_search=100, top_k=10

### all (63 questions)

| System | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Median latency |
| --- | --- | --- | --- | --- | --- |
| Dense only | 0.542 | 0.643 | 0.494 | 0.497 | 18.3 ms |
| BM25 only | 0.678 | 0.766 | 0.664 | 0.652 | 3.0 ms |
| Hybrid | 0.673 | 0.768 | 0.621 | 0.614 | 28.2 ms |
| Hybrid + reranker | 0.698 | 0.829 | 0.751 | 0.699 | 1580.9 ms |

### natural-language (37 questions)

| System | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Median latency |
| --- | --- | --- | --- | --- | --- |
| Dense only | 0.639 | 0.710 | 0.607 | 0.593 | 18.5 ms |
| BM25 only | 0.582 | 0.669 | 0.571 | 0.555 | 3.3 ms |
| Hybrid | 0.642 | 0.745 | 0.596 | 0.585 | 28.4 ms |
| Hybrid + reranker | 0.605 | 0.791 | 0.714 | 0.641 | 1588.5 ms |

### exact-match (26 questions)

| System | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Median latency |
| --- | --- | --- | --- | --- | --- |
| Dense only | 0.405 | 0.549 | 0.332 | 0.360 | 17.8 ms |
| BM25 only | 0.815 | 0.904 | 0.798 | 0.791 | 2.8 ms |
| Hybrid | 0.717 | 0.802 | 0.658 | 0.656 | 27.9 ms |
| Hybrid + reranker | 0.831 | 0.883 | 0.803 | 0.782 | 1517.7 ms |

### NDCG@10 by category

| Category | n | Dense only | BM25 only | Hybrid | Hybrid + reranker |
| --- | --- | --- | --- | --- | --- |
| change | 2 | 0.000 | 0.899 | 0.193 | 0.807 |
| code | 7 | 0.609 | 0.665 | 0.630 | 0.696 |
| documentation | 9 | 0.837 | 0.507 | 0.780 | 0.733 |
| exact-match | 26 | 0.360 | 0.791 | 0.656 | 0.782 |
| incident | 6 | 0.586 | 0.355 | 0.448 | 0.467 |
| runbook | 13 | 0.510 | 0.569 | 0.548 | 0.604 |

### Rank of the first relevant result (- = not in the top 10)

| Question | Dense only | BM25 only | Hybrid | Hybrid + reranker | Text |
| --- | --- | --- | --- | --- | --- |
| RQ-01 | 1 | 1 | 1 | 1 | How do we handle database connection exhaustion? |
| RQ-02 | - | 1 | - | 1 | Redis writes are failing with 'OOM command not allowed'. What should I check? |
| RQ-03 | 4 | 1 | 2 | 7 | A Kafka consumer group keeps falling further behind. How do I troubleshoot it? |
| RQ-04 | 2 | 5 | 1 | 1 | Customers get 401 errors right after the signing keys were rotated. What do we d |
| RQ-05 | 1 | 1 | 1 | 1 | What is the procedure to roll back a bad release? |
| RQ-06 | 1 | 1 | 1 | 1 | Postgres keeps logging 'deadlock detected' on writes. How should we respond? |
| RQ-07 | 2 | 2 | 2 | 1 | Pods keep restarting because they are OOMKilled. |
| RQ-08 | 4 | 4 | 5 | 3 | Mobile clients are receiving HTTP 429 Too Many Requests from the API. |
| RQ-09 | 1 | 1 | 1 | 1 | Product pages still show the old price after a price update. |
| RQ-10 | - | - | - | 8 | Our stock levels no longer match what the warehouse system reports. |
| RQ-11 | - | - | 2 | 3 | Order confirmation emails are arriving hours late. |
| RQ-12 | 1 | 2 | 2 | 4 | The API gateway is returning 504 Gateway Timeout for many requests. |
| RQ-13 | - | - | - | - | Card authorizations are failing because the payment processor returns errors. |
| RQ-14 | 1 | 3 | 1 | 1 | Where is payment database configuration? |
| RQ-15 | 1 | 1 | 1 | 1 | What timeout and retry settings does order-service use when it calls other servi |
| RQ-16 | 3 | - | 4 | 2 | Which services depend on inventory-service? |
| RQ-17 | 1 | 2 | 1 | 1 | How does canary analysis decide whether a deployment continues? |
| RQ-18 | 10 | 1 | 5 | 7 | Is it safe to rename a field in a Kafka event? |
| RQ-19 | 1 | 1 | 1 | 1 | How long are access tokens valid and how does the gateway validate them? |
| RQ-20 | 1 | 8 | 1 | 1 | Which alerts are defined for cart-service and what are their thresholds? |
| RQ-21 | 1 | 3 | 1 | 3 | What columns does the payments table have? |
| RQ-22 | 1 | - | 3 | 1 | What request rate limits apply to API clients? |
| RQ-23 | 4 | 5 | 4 | 1 | Where is the database connection pool size set for payment-service? |
| RQ-24 | 1 | 4 | 2 | 1 | Where is the token bucket rate limiter implemented? |
| RQ-25 | 1 | 1 | 1 | 1 | Which code prevents charging a customer twice when a payment request is retried? |
| RQ-26 | 4 | 1 | 2 | 4 | Where does checkout reserve inventory and authorize the payment? |
| RQ-27 | 1 | 1 | 1 | 1 | How does the gateway cache the JWKS signing keys? |
| RQ-28 | 5 | 1 | 3 | 6 | Where is user input escaped before the OpenSearch query is built? |
| RQ-29 | 2 | 4 | 2 | 1 | How are shopping carts stored in Redis and when do they expire? |
| RQ-30 | 1 | 1 | 3 | 1 | Why did payment-service fail? |
| RQ-31 | 1 | 1 | 1 | 3 | Why did payment requests start returning HTTP 500 errors after deployment v2.8.1 |
| RQ-32 | 3 | - | 7 | 6 | What caused inventory stock to drift away from the warehouse system? |
| RQ-33 | 1 | 2 | 1 | 1 | Why were valid tokens rejected by the API gateway after a key rotation? |
| RQ-34 | 1 | 6 | 1 | 1 | What happened when the cart Redis cluster ran out of memory? |
| RQ-35 | - | - | - | 3 | Consumers failed to process product events after a field was renamed. What happe |
| RQ-36 | - | 1 | - | 1 | Which pull requests reduced the database connection pool size? |
| RQ-37 | - | 1 | 2 | 1 | What changed in payment-service v2.8.1? |
| RQ-38 | 1 | 1 | 1 | 1 | Our Kafka consumers keep logging 'Revoking previously assigned partitions'. |
| RQ-39 | - | 2 | - | 1 | Requests are rejected with 'token is not yet valid'. |
| RQ-40 | - | 1 | 1 | 1 | Pods log 'Temporary failure in name resolution' when calling other services. |
| RQ-41 | - | 2 | 8 | 1 | What does 'unknown signing key' mean and how do we fix it? |
| RQ-42 | 5 | 1 | 1 | 1 | payment-service logs 'QueuePool limit of size 20 overflow 10 reached, connection |
| RQ-43 | - | 1 | 1 | 1 | What happened in INC-0406? |
| RQ-44 | 6 | 1 | 2 | 1 | Give me the details of INC-0329. |
| RQ-45 | - | 1 | 2 | 2 | What was the root cause of INC-0380? |
| RQ-46 | - | 1 | 5 | 10 | Which pull requests shipped in DEP-0296? |
| RQ-47 | - | 1 | 8 | 3 | What was released in payment-service v2.8.2? |
| RQ-48 | 2 | 1 | 1 | 1 | Why was inventory-service v1.17.2 rolled back? |
| RQ-49 | - | 1 | 1 | 1 | What is RECONCILIATION_TOLERANCE_UNITS used for? |
| RQ-50 | 5 | 4 | 2 | 1 | Where is JWKS_CACHE_TTL_SECONDS defined? |
| RQ-51 | 3 | 1 | 1 | 2 | What does PAYMENT_SERVICE_DB_POOL_TIMEOUT_SECONDS control? |
| RQ-52 | 10 | 2 | 2 | 4 | What is the value of RISK_DECLINE_THRESHOLD? |
| RQ-53 | 1 | 1 | 1 | 1 | What does MAX_STACKED_PROMOTIONS limit? |
| RQ-54 | 2 | 2 | 1 | 1 | Show me the IdempotencyStore class. |
| RQ-55 | 9 | - | - | 1 | Where is CheckoutOrchestrator defined? |
| RQ-56 | 1 | 2 | 1 | 2 | When is CircuitOpenError raised? |
| RQ-57 | 5 | 1 | 1 | 1 | Which code raises InsufficientStock? |
| RQ-58 | 5 | 1 | 3 | 5 | What does escape_query do? |
| RQ-59 | - | 1 | 5 | 2 | How does consume_refresh_token work? |
| RQ-60 | 1 | 1 | 1 | 1 | Where is effective_price implemented? |
| RQ-61 | 1 | 1 | 1 | 1 | Which pull requests changed OrderSaga._advance? |
| RQ-62 | 1 | 1 | 1 | 1 | Which incidents involved redis-payments? |
| RQ-63 | 8 | - | 8 | 1 | What does the PaymentServiceDBPoolSaturated alert mean? |
