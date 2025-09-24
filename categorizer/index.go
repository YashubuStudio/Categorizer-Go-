package categorizer

import (
	"container/heap"
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

// Items returns a defensive copy of the stored items for inspection/debugging.
func (idx *InMemoryIndex) Items() []VectorItem {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	out := make([]VectorItem, len(idx.items))
	for i, it := range idx.items {
		out[i] = VectorItem{
			Label:  it.Label,
			Source: it.Source,
			Vector: cloneVector(it.Vector),
		}
	}
	return out
}

// Search performs cosine similarity against all stored items and returns the top-k hits.
func (idx *InMemoryIndex) Search(vec []float32, k int) []Hit {
	idx.mu.RLock()
	items := idx.items
	idx.mu.RUnlock()
	if len(items) == 0 || len(vec) == 0 || k <= 0 {
		return nil
	}
	if k > len(items) {
		k = len(items)
	}
	hitHeap := &hitMinHeap{}
	heap.Init(hitHeap)
	for _, it := range items {
		score := cosineSimilarity(vec, it.Vector)
		candidate := Hit{
			Label:  it.Label,
			Score:  score,
			Source: it.Source,
			Vector: it.Vector,
		}
		if hitHeap.Len() < k {
			heap.Push(hitHeap, candidate)
			continue
		}
		worst := (*hitHeap)[0]
		if hitBetter(candidate, worst) {
			heap.Pop(hitHeap)
			heap.Push(hitHeap, candidate)
		}
	}
	if hitHeap.Len() == 0 {
		return nil
	}
	hits := make([]Hit, hitHeap.Len())
	for i := range hits {
		hits[i] = heap.Pop(hitHeap).(Hit)
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score == hits[j].Score {
			return hits[i].Label < hits[j].Label
		}
		return hits[i].Score > hits[j].Score
	})
	return hits
}

type hitMinHeap []Hit

func (h hitMinHeap) Len() int { return len(h) }

func (h hitMinHeap) Less(i, j int) bool {
	if h[i].Score == h[j].Score {
		return h[i].Label > h[j].Label
	}
	return h[i].Score < h[j].Score
}

func (h hitMinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *hitMinHeap) Push(x any) {
	*h = append(*h, x.(Hit))
}

func (h *hitMinHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

func hitBetter(a, b Hit) bool {
	if a.Score == b.Score {
		return a.Label < b.Label
	}
	return a.Score > b.Score
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
