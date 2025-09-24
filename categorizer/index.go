package categorizer

import (
	"math"
	"sort"
	"sync"
)

// VectorItem represents an entry within a vector index.
type VectorItem struct {
	Label  string
	Source string
	Vector []float32
}

// Hit is an internal structure used when combining scores.
type Hit struct {
	Label  string
	Score  float32
	Source string
	Vector []float32
}

// VectorIndex provides nearest neighbour search capabilities.
type VectorIndex interface {
	Replace(items []VectorItem)
	Search(vec []float32, k int) []Hit
	Size() int
}

// InMemoryIndex is a brute-force vector index with cosine similarity.
type InMemoryIndex struct {
	mu    sync.RWMutex
	items []VectorItem
}

// NewInMemoryIndex constructs an empty index.
func NewInMemoryIndex() *InMemoryIndex {
	return &InMemoryIndex{}
}

// Replace swaps the stored items atomically.
func (idx *InMemoryIndex) Replace(items []VectorItem) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.items = make([]VectorItem, len(items))
	for i, it := range items {
		idx.items[i] = VectorItem{
			Label:  it.Label,
			Source: it.Source,
			Vector: cloneVector(it.Vector),
		}
	}
}

// Size returns the current number of vectors stored.
func (idx *InMemoryIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.items)
}

// Search performs cosine similarity against all stored items and returns the top-k hits.
func (idx *InMemoryIndex) Search(vec []float32, k int) []Hit {
	idx.mu.RLock()
	items := idx.items
	idx.mu.RUnlock()
	if len(items) == 0 || len(vec) == 0 || k <= 0 {
		return nil
	}
	hits := make([]Hit, 0, len(items))
	for _, it := range items {
		score := cosineSimilarity(vec, it.Vector)
		hits = append(hits, Hit{
			Label:  it.Label,
			Score:  score,
			Source: it.Source,
			Vector: cloneVector(it.Vector),
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		return hits[i].Score > hits[j].Score
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		na += fa * fa
		nb += fb * fb
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (sqrtFloat64(na) * sqrtFloat64(nb)))
}

func sqrtFloat64(v float64) float64 {
	return mathSqrt(v)
}

// mathSqrt is isolated for testing overrides (use math.Sqrt by default).
var mathSqrt = func(v float64) float64 { return math.Sqrt(v) }
