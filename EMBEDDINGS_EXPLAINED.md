# Understanding Embeddings in StockSquad

## What Are Embeddings?

Embeddings are **numerical representations of text** that capture semantic meaning. They transform human language into vectors (arrays of numbers) that computers can mathematically compare.

### The Magic: Similar Meaning = Similar Numbers

```
Text                              →  Embedding Vector (1536 dimensions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Apple stock rising"              →  [0.23, -0.45, 0.78, ..., 0.12]
"AAPL shares going up"            →  [0.25, -0.43, 0.80, ..., 0.14]  ← Very close!
"Strong performance from Apple"   →  [0.22, -0.47, 0.79, ..., 0.11]  ← Also close!

"Pizza delivery service"          →  [-0.67, 0.89, -0.32, ..., 0.45]  ← Far away!
```

**Key Insight**: If two pieces of text mean similar things, their embedding vectors will be **mathematically close** in 1536-dimensional space, even if they use completely different words!

---

## The Model: text-embedding-ada-002

### Specifications

| Property | Details |
|----------|---------|
| **Provider** | Azure OpenAI (OpenAI model) |
| **Vector Size** | 1,536 dimensions (each text → array of 1,536 floats) |
| **Max Input** | 8,191 tokens (~6,000 words) |
| **Training** | Trained on massive text corpus for semantic understanding |
| **Cost** | ~$0.0001 per 1K tokens (extremely cheap) |
| **Latency** | ~50-200ms per request |

### Why Ada-002?

1. **Best-in-class accuracy** for retrieval tasks (beating competitors on benchmarks)
2. **Cost-effective** (100x cheaper than GPT-4 calls)
3. **Fast** (real-time embedding generation)
4. **Industry standard** (used by thousands of applications)
5. **Stable** (won't be deprecated soon)

---

## How StockSquad Uses Embeddings

### Use Case 1: Storing Analysis in Memory

When OrchestratorAgent completes an analysis, we need to store it so future analyses can reference it.

**The Process** (in `memory/long_term.py`):

```python
def store_analysis(self, ticker, analysis_summary, full_analysis):
    # Step 1: Create a concise text summary
    summary = f"""Stock analysis for {ticker} on 2026-03-31.
    Apple Inc - Technology sector.
    Price: $175.23, up 5.2% over the year.
    Strong financials with P/E of 28.5.
    Concerns about supply chain risks."""

    # Step 2: Convert text to embedding vector
    embedding = self.openai_client.embeddings.create(
        input=summary,
        model="text-embedding-ada-002"
    )
    # Returns: [0.234, -0.456, 0.789, ..., 0.123]  (1,536 numbers)

    # Step 3: Store in ChromaDB
    self.collection.add(
        ids=["AAPL_20260331_143022"],       # Unique ID
        embeddings=[embedding.data[0].embedding],  # The vector
        documents=[summary],                # The original text
        metadatas=[{"ticker": "AAPL", "date": "2026-03-31"}]
    )

    # Step 4: Also save full analysis as JSON
    # (for detailed retrieval later)
```

**What's stored:**
- ✅ Vector (1,536 floats) → for similarity search
- ✅ Text summary → for display
- ✅ Metadata → for filtering
- ✅ Full JSON file → for complete details

---

### Use Case 2: Retrieving Past Analyses

When analyzing AAPL again, the system checks memory:

```python
# In agents/orchestrator.py
past_analyses = self.long_term_memory.retrieve_past_analyses(
    ticker="AAPL",
    limit=3
)
```

**Behind the scenes:**

```python
# In memory/long_term.py
def retrieve_past_analyses(self, ticker, limit):
    # Query ChromaDB with filter
    results = self.collection.get(
        where={"ticker": "AAPL"},  # Filter by ticker
        limit=limit
    )

    # Returns: Most recent analyses for this ticker
    # No embedding comparison needed - just metadata filtering!
    return results
```

This is a **simple filter-based retrieval** (not using embeddings for similarity).

---

### Use Case 3: Semantic Search (The Real Magic!)

This is where embeddings shine. You can search for concepts without exact keyword matches:

```python
# Example: Find analyses discussing specific themes
results = memory.semantic_search(
    query="What were concerns about data center expansion?",
    ticker="NVDA"
)
```

**What happens:**

```python
def semantic_search(self, query, ticker=None, limit=5):
    # Step 1: Convert your query to an embedding
    query_embedding = self._generate_embedding(query)
    # "concerns about data center expansion" → [0.12, -0.89, 0.45, ...]

    # Step 2: Search ChromaDB for similar vectors
    results = self.collection.query(
        query_embeddings=[query_embedding],
        where={"ticker": ticker} if ticker else None,
        n_results=limit
    )

    # ChromaDB compares your query vector to all stored vectors
    # Returns: Analyses with the most similar embeddings

    # Step 3: Calculate similarity scores
    for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
        similarity = 1 - distance  # Convert distance to similarity
        # distance = 0.2 → similarity = 0.8 (80% similar)
        # distance = 0.8 → similarity = 0.2 (20% similar)
```

**Real Example:**

| Your Query | Stored Analysis Summary | Similarity | Match? |
|------------|------------------------|------------|--------|
| "data center concerns" | "NVDA analysis: Strong growth in data center segment, but supply constraints remain" | 0.85 | ✅ High match! |
| "data center concerns" | "NVDA analysis: Gaming revenue declined, AI chips sold out" | 0.72 | ✅ Medium match |
| "data center concerns" | "NVDA analysis: CEO announced new GPU architecture" | 0.45 | ❌ Low match |
| "data center concerns" | "AAPL analysis: iPhone sales strong in China" | 0.15 | ❌ No match |

**The magic**: The system found relevant analyses even though they used different words:
- You said: "concerns"
- Analysis said: "constraints remain"
- Embedding model knew these are semantically related!

---

## How Vector Similarity Works

### Cosine Similarity (Used by ChromaDB)

ChromaDB measures similarity using **cosine distance**, which compares vector directions:

```
Vector A: [0.5, 0.8, 0.2]     "Apple stock is rising"
Vector B: [0.4, 0.9, 0.1]     "AAPL shares going up"
                              ↓
Cosine Distance = 0.02        (very small = very similar)
Similarity Score = 0.98       (98% similar)


Vector A: [0.5, 0.8, 0.2]     "Apple stock is rising"
Vector C: [-0.6, -0.3, 0.9]   "Pizza delivery service"
                              ↓
Cosine Distance = 0.87        (large = very different)
Similarity Score = 0.13       (13% similar)
```

**Formula** (for the mathematically curious):

```
similarity = 1 - distance

distance = 1 - (A · B) / (||A|| * ||B||)

Where:
  A · B = dot product (sum of element-wise multiplication)
  ||A|| = magnitude of vector A
  ||B|| = magnitude of vector B
```

---

## Why This Matters for StockSquad

### Problem: Keyword Search Fails

**Traditional keyword search** (like grep or SQL LIKE):

```sql
SELECT * FROM analyses WHERE summary LIKE '%data center%'
```

**Limitations:**
- ❌ Misses "server farm", "cloud infrastructure", "compute capacity"
- ❌ Misses "datacenter" (one word vs two)
- ❌ Can't understand "GPU demand for AI workloads" is related
- ❌ No ranking by relevance

### Solution: Semantic Search with Embeddings

```python
results = memory.semantic_search("data center growth")
```

**Advantages:**
- ✅ Finds "server expansion", "cloud build-out", "infrastructure investment"
- ✅ Understands "GPU sales for AI" is related to data center context
- ✅ Returns results ranked by semantic similarity (most relevant first)
- ✅ Works across different phrasings and synonyms

### Real-World StockSquad Scenarios

**Scenario 1: Comparing analyses over time**

```python
# Find all analyses mentioning supply chain issues
results = memory.semantic_search(
    query="supply chain disruptions and shortages",
    limit=10
)

# Returns analyses mentioning:
# - "component availability constraints"
# - "manufacturing delays"
# - "inventory challenges"
# - "logistics bottlenecks"
# Even if they never said "supply chain"!
```

**Scenario 2: Cross-ticker pattern detection**

```python
# Find any stock with margin pressure concerns
results = memory.semantic_search(
    query="declining profit margins and cost pressures"
)

# Might return:
# - AAPL: "Services margin compression due to mix shift"
# - MSFT: "Cloud margins declining as competition intensifies"
# - NVDA: "Gross margin pressure from data center pricing"
```

---

## Under the Hood: How Ada-002 Was Trained

**Training Process** (simplified):

1. **Massive text corpus**: OpenAI trained on billions of text pairs
2. **Contrastive learning**: Model learned to place similar texts close together
3. **Optimization**: Minimized distance between paraphrases, maximized distance between unrelated text
4. **Result**: Model that "understands" semantic relationships

**What the model learned:**

- "stock rising" ≈ "shares gaining" ≈ "equity advancing"
- "profit margin" ≈ "profitability" ≈ "earnings quality"
- "Q4 earnings" ≈ "fourth quarter results" ≈ "year-end financials"
- "bearish outlook" ≠ "bullish sentiment" (opposite meanings → far apart)

---

## ChromaDB: The Vector Database

### Why ChromaDB?

| Feature | Benefit |
|---------|---------|
| **Vector-native** | Optimized for embedding storage & search |
| **Fast similarity search** | Uses HNSW algorithm (Hierarchical Navigable Small World) |
| **Local-first** | Runs locally, no external service needed |
| **Python-friendly** | Simple API, easy integration |
| **Metadata filtering** | Combine vector search with filters (e.g., ticker="AAPL") |

### What ChromaDB Does

```python
# When you add an analysis
collection.add(
    embeddings=[vector],  # ChromaDB builds an HNSW index
    documents=[text],
    metadatas=[metadata]
)

# When you search
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)

# ChromaDB:
# 1. Uses HNSW index to quickly find nearest neighbors
# 2. Calculates cosine distances
# 3. Returns top-k most similar items
# All in milliseconds, even with thousands of vectors!
```

---

## Performance Characteristics

### Embedding Generation

```
Input:  "Stock analysis for AAPL..."  (50 words)
Model:  text-embedding-ada-002
Time:   ~100ms
Cost:   ~$0.000005 (essentially free)
Output: [1536 floats] = ~6KB
```

### Similarity Search (ChromaDB)

```
Database: 1,000 stored analyses
Query:    "revenue growth concerns"
Time:     ~10ms (using HNSW index)
Result:   Top 5 most similar analyses
```

**Scalability**: Efficient up to ~1M vectors on consumer hardware

---

## Limitations & Gotchas

### 1. Context Window

- **Max input**: 8,191 tokens (~6,000 words)
- **Solution**: StockSquad stores concise summaries (100-300 words)

### 2. Single Vector Per Analysis

- Each analysis gets **one** embedding vector
- Can't capture multiple distinct themes well
- **Future improvement**: Chunk analyses, store multiple embeddings

### 3. Embedding Drift

- If OpenAI updates the model, embeddings change
- **Mitigation**: Version lock the model, or re-embed everything on updates

### 4. Not Good for Exact Matches

- Embeddings capture semantics, not exact strings
- For exact ticker/date lookup, use metadata filtering (which we do!)

---

## Code Walkthrough: Key Files

### Where embeddings are generated

**File**: `memory/long_term.py:46-58`

```python
def _generate_embedding(self, text: str) -> List[float]:
    response = self.openai_client.embeddings.create(
        input=text,
        model=self.settings.azure_openai_embedding_deployment_name,
    )
    return response.data[0].embedding
```

### Where embeddings are stored

**File**: `memory/long_term.py:74-97`

```python
def store_analysis(...):
    # Generate embedding
    embedding = self._generate_embedding(analysis_summary)

    # Store in ChromaDB
    self.collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[analysis_summary],
        metadatas=[doc_metadata],
    )
```

### Where semantic search happens

**File**: `memory/long_term.py:204-229`

```python
def semantic_search(self, query: str, ...):
    # Embed the query
    query_embedding = self._generate_embedding(query)

    # Search ChromaDB
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
    )
```

---

## Practical Experiment: See It in Action

### Try This (After Setup):

```python
# Run multiple analyses
python main.py analyze AAPL
python main.py analyze MSFT
python main.py analyze NVDA

# Now try semantic search in Python:
from memory.long_term import LongTermMemory

memory = LongTermMemory()

# Search across all tickers
results = memory.semantic_search("cloud computing revenue growth")

# See what matches!
for r in results:
    print(f"{r['ticker']}: {r['summary']}")
    print(f"Similarity: {r['similarity_score']:.2%}\n")
```

**Expected**: Might find MSFT (Azure), AAPL (iCloud), even if "cloud" wasn't the main topic!

---

## Further Reading

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Vector Search Explained](https://www.pinecone.io/learn/vector-search/)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

---

## Summary

**Embeddings** = Text → Numbers that capture meaning

**text-embedding-ada-002** = OpenAI's best embedding model (1,536 dimensions)

**StockSquad uses it for**:
1. Storing analysis summaries with semantic meaning
2. Retrieving past analyses by ticker (metadata filter)
3. Semantic search across all analyses (similarity matching)

**The magic**: Find related analyses even with different words, enabling the system to learn from past analyses and provide context-aware insights!

**Cost**: Negligible (~$0.0001 per analysis)

**Speed**: Fast (~100ms to embed, ~10ms to search)

**Result**: Powerful long-term memory that "understands" stock analysis concepts!
