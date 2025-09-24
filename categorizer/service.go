package categorizer

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
)

// Service orchestrates embedding, ranking and clustering based on the plan specification.
type Service struct {
	embedder Embedder

	cfgMu sync.RWMutex
	cfg   Config

	seedsIdx *InMemoryIndex
	ndcIdx   *InMemoryIndex

	logger *log.Logger
}

// NewService constructs a service with the given embedder and configuration.
func NewService(ctx context.Context, embedder Embedder, cfg Config, logger *log.Logger) (*Service, error) {
	if embedder == nil {
		return nil, errors.New("embedder is required")
	}
	cfg.ApplyDefaults()
	s := &Service{
		embedder: embedder,
		cfg:      cfg,
		seedsIdx: NewInMemoryIndex(),
		ndcIdx:   NewInMemoryIndex(),
		logger:   logger,
	}
	if cfg.UseNDC {
		if err := s.LoadNDCDictionary(ctx, DefaultNDCEntries()); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// Close releases embedder resources.
func (s *Service) Close() error {
	if s.embedder != nil {
		return s.embedder.Close()
	}
	return nil
}

// Config returns a copy of the current configuration.
func (s *Service) Config() Config {
	s.cfgMu.RLock()
	defer s.cfgMu.RUnlock()
	return s.cfg.Clone()
}

// UpdateConfig replaces the configuration.
func (s *Service) UpdateConfig(cfg Config) {
	cfg.ApplyDefaults()
	s.cfgMu.Lock()
	s.cfg = cfg
	s.cfgMu.Unlock()
}

// LoadNDCDictionary embeds and stores the provided entries.
func (s *Service) LoadNDCDictionary(ctx context.Context, entries []NDCEntry) error {
	if len(entries) == 0 {
		s.ndcIdx.Replace(nil)
		s.logf("NDC dictionary cleared")
		return nil
	}
	texts := make([]string, len(entries))
	labels := make([]string, len(entries))
	for i, entry := range entries {
		normalized := NormalizeText(entry.Label)
		texts[i] = fmt.Sprintf("%s %s", entry.Code, normalized)
		labels[i] = fmt.Sprintf("%s:%s", entry.Code, normalized)
	}
	vecs, err := s.embedder.EmbedTexts(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed ndc dictionary: %w", err)
	}
	items := make([]VectorItem, len(entries))
	for i := range entries {
		items[i] = VectorItem{
			Label:  labels[i],
			Source: "ndc",
			Vector: vecs[i],
		}
	}
	s.ndcIdx.Replace(items)
	s.logf("Loaded %d NDC entries", len(items))
	return nil
}

// LoadSeeds embeds the provided seed categories and replaces the current index.
func (s *Service) LoadSeeds(ctx context.Context, seeds []string) error {
	cleaned := make([]string, 0, len(seeds))
	seen := make(map[string]struct{})
	for _, seed := range seeds {
		seed = strings.TrimSpace(seed)
		if seed == "" {
			continue
		}
		normalized := NormalizeText(seed)
		if _, ok := seen[normalized]; ok {
			continue
		}
		seen[normalized] = struct{}{}
		cleaned = append(cleaned, normalized)
	}
	if len(cleaned) == 0 {
		s.seedsIdx.Replace(nil)
		s.logf("Seed list cleared")
		return nil
	}
	vecs, err := s.embedder.EmbedTexts(ctx, cleaned)
	if err != nil {
		return fmt.Errorf("embed seeds: %w", err)
	}
	items := make([]VectorItem, len(cleaned))
	for i, label := range cleaned {
		items[i] = VectorItem{
			Label:  label,
			Source: "seed",
			Vector: vecs[i],
		}
	}
	s.seedsIdx.Replace(items)
	s.logf("Loaded %d seed categories", len(items))
	return nil
}

// SeedCount returns how many seed categories are indexed.
func (s *Service) SeedCount() int {
	return s.seedsIdx.Size()
}

// ClassifyAll embeds all texts and returns ranked suggestions.
func (s *Service) ClassifyAll(ctx context.Context, texts []string) ([]ResultRow, error) {
	cfg := s.Config()
	normTexts := NormalizeAll(texts)
	vecs, err := s.embedder.EmbedTexts(ctx, normTexts)
	if err != nil {
		return nil, fmt.Errorf("embed texts: %w", err)
	}
	rows := make([]ResultRow, 0, len(texts))
	for i, vec := range vecs {
		rows = append(rows, s.rankForVector(vec, texts[i], cfg))
	}
	return rows, nil
}

func (s *Service) rankForVector(vec []float32, originalText string, cfg Config) ResultRow {
	topK := cfg.TopK
	if topK <= 0 {
		topK = 3
	}
	seedHits := filterHits(s.seedsIdx.Search(vec, topK*3), cfg.MinScore)
	ndcHits := filterHits(s.ndcIdx.Search(vec, topK*3), cfg.MinScore)

	var suggestions []Suggestion
	var ndcSuggestions []Suggestion

	switch cfg.Mode {
	case ModeSeeded:
		if cfg.Cluster.Enabled {
			seedHits = clusterHits(seedHits, cfg.Cluster.Threshold)
		}
		seedHits = limitHits(seedHits, topK)
		suggestions = hitsToSuggestions(seedHits)
	case ModeSplit:
		if cfg.Cluster.Enabled {
			seedHits = clusterHits(seedHits, cfg.Cluster.Threshold)
			ndcHits = clusterHits(ndcHits, cfg.Cluster.Threshold)
		}
		suggestions = hitsToSuggestions(limitHits(seedHits, topK))
		if cfg.UseNDC {
			ndcSuggestions = hitsToSuggestions(limitHits(ndcHits, topK))
		}
	case ModeMixed:
		weighted := make([]Hit, 0, len(seedHits)+len(ndcHits))
		for _, h := range seedHits {
			h.Score += cfg.SeedBias
			weighted = append(weighted, h)
		}
		if cfg.UseNDC {
			weighted = append(weighted, ndcHits...)
		}
		if cfg.Cluster.Enabled {
			weighted = clusterHits(weighted, cfg.Cluster.Threshold)
		}
		sort.Slice(weighted, func(i, j int) bool {
			if weighted[i].Score == weighted[j].Score {
				return weighted[i].Label < weighted[j].Label
			}
			return weighted[i].Score > weighted[j].Score
		})
		suggestions = hitsToSuggestions(limitHits(weighted, topK))
	default:
		// Fallback to seeded behaviour if mode unknown.
		if cfg.Cluster.Enabled {
			seedHits = clusterHits(seedHits, cfg.Cluster.Threshold)
		}
		suggestions = hitsToSuggestions(limitHits(seedHits, topK))
	}

	return ResultRow{
		Text:           originalText,
		Suggestions:    suggestions,
		NDCSuggestions: ndcSuggestions,
	}
}

func filterHits(hits []Hit, minScore float32) []Hit {
	if minScore <= 0 {
		return hits
	}
	out := hits[:0]
	for _, h := range hits {
		if h.Score >= minScore {
			out = append(out, h)
		}
	}
	return out
}

func limitHits(hits []Hit, k int) []Hit {
	if len(hits) <= k {
		return hits
	}
	return hits[:k]
}

func hitsToSuggestions(hits []Hit) []Suggestion {
	out := make([]Suggestion, len(hits))
	for i, h := range hits {
		out[i] = Suggestion{
			Label:  h.Label,
			Score:  h.Score,
			Source: h.Source,
		}
	}
	return out
}

func (s *Service) logf(format string, args ...any) {
	if s.logger != nil {
		s.logger.Printf(format, args...)
	}
}
