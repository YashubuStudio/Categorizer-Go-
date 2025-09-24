package categorizer

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"log"
	"sort"
	"strings"
	"sync"
	"time"
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
	if logger != nil {
		logger.Printf("NewService configuration: %+v", cfg)
		logger.Printf("NewService embedder model: %s", embedder.ModelID())
	}
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
	start := time.Now()
	if len(entries) == 0 {
		s.ndcIdx.Replace(nil)
		s.logf("NDC dictionary cleared")
		return nil
	}
	s.logf("LoadNDCDictionary received %d entries", len(entries))
	for i, entry := range entries {
		s.logf("LoadNDCDictionary entry[%d]: code=%q label=%q", i, entry.Code, entry.Label)
	}
	s.logf("Loading %d NDC entries", len(entries))
	texts := make([]string, len(entries))
	labels := make([]string, len(entries))
	for i, entry := range entries {
		normalized := NormalizeText(entry.Label)
		texts[i] = fmt.Sprintf("%s %s", entry.Code, normalized)
		labels[i] = fmt.Sprintf("%s:%s", entry.Code, normalized)
		s.logf("LoadNDCDictionary normalized entry[%d]: text=%q labelKey=%q", i, texts[i], labels[i])
	}
	vecs, err := s.embedder.EmbedTexts(ctx, texts)
	if err != nil {
		s.logf("Failed to embed NDC dictionary after %s: %v", time.Since(start), err)
		return fmt.Errorf("embed ndc dictionary: %w", err)
	}
	for i, vec := range vecs {
		s.logf("LoadNDCDictionary embedding[%d]: label=%q %s", i, labels[i], formatVectorDebug(vec))
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
	s.logf("Loaded %d NDC entries in %s", len(items), time.Since(start))
	return nil
}

// LoadSeeds embeds the provided seed categories and replaces the current index.
func (s *Service) LoadSeeds(ctx context.Context, seeds []string) error {
	start := time.Now()
	s.logf("LoadSeeds raw input: %v", seeds)
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
	s.logf("LoadSeeds normalized unique seeds: %v", cleaned)
	s.logf("Embedding %d seed categories", len(cleaned))
	vecs, err := s.embedder.EmbedTexts(ctx, cleaned)
	if err != nil {
		s.logf("Failed to embed seeds after %s: %v", time.Since(start), err)
		return fmt.Errorf("embed seeds: %w", err)
	}
	for i, vec := range vecs {
		s.logf("LoadSeeds embedding[%d]: label=%q %s", i, cleaned[i], formatVectorDebug(vec))
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
	s.logf("Loaded %d seed categories in %s", len(items), time.Since(start))
	return nil
}

// SeedCount returns how many seed categories are indexed.
func (s *Service) SeedCount() int {
	return s.seedsIdx.Size()
}

// SeedLabels returns the normalized seed labels currently indexed.
func (s *Service) SeedLabels() []string {
	items := s.seedsIdx.Items()
	labels := make([]string, len(items))
	for i, it := range items {
		labels[i] = it.Label
	}
	return labels
}

// ClassifyAll embeds all texts and returns ranked suggestions.
func (s *Service) ClassifyAll(ctx context.Context, texts []string) ([]ResultRow, error) {
	start := time.Now()
	total := len(texts)
	s.logf("ClassifyAll start: %d texts (seeds=%d ndc=%d)", total, s.SeedCount(), s.ndcIdx.Size())
	for i, text := range texts {
		s.logf("ClassifyAll input[%d]: %q", i, text)
	}
	normalizeStart := time.Now()
	cfg := s.Config()
	s.logf("ClassifyAll configuration: %+v", cfg)
	s.logf("ClassifyAll embedder model: %s", s.embedder.ModelID())
	normTexts := NormalizeAll(texts)
	normalizeDur := time.Since(normalizeStart)
	for i, norm := range normTexts {
		s.logf("ClassifyAll normalized[%d]: %q -> %q", i, texts[i], norm)
	}
	embedStart := time.Now()
	vecs, err := s.embedder.EmbedTexts(ctx, normTexts)
	if err != nil {
		s.logf("ClassifyAll failed during embedding after %s: %v", time.Since(embedStart), err)
		return nil, fmt.Errorf("embed texts: %w", err)
	}
	embedDur := time.Since(embedStart)
	for i, vec := range vecs {
		s.logf("ClassifyAll embedding[%d]: %s", i, formatVectorDebug(vec))
	}
	rows := make([]ResultRow, 0, len(texts))
	rankStart := time.Now()
	for i, vec := range vecs {
		s.logf("ClassifyAll ranking index %d", i)
		rows = append(rows, s.rankForVector(vec, texts[i], cfg))
	}
	rankDur := time.Since(rankStart)
	s.logf("ClassifyAll completed: %d texts (normalize=%s embed=%s rank=%s total=%s)", len(rows), normalizeDur, embedDur, rankDur, time.Since(start))
	for i, row := range rows {
		s.logf("ClassifyAll result[%d]:\n%s", i, formatResultRowDebug(row))
	}
	return rows, nil
}

func (s *Service) rankForVector(vec []float32, originalText string, cfg Config) ResultRow {
	topK := clampTopK(cfg.TopK)
	s.logf("rankForVector start: text=%q %s", originalText, formatVectorDebug(vec))
	s.logf("rankForVector mode=%s topK=%d useNDC=%t", cfg.Mode, topK, cfg.UseNDC)
	rawSeedHits := s.seedsIdx.Search(vec, topK*3)
	s.logf("rankForVector raw seed hits (limit=%d): %s", topK*3, formatHitsDebug(rawSeedHits))
	seedHits := applySourceWeight(rawSeedHits, 1)
	s.logf("rankForVector weighted seed hits: %s", formatHitsDebug(seedHits))
	var ndcHits []Hit
	if cfg.UseNDC {
		rawNDCHits := s.ndcIdx.Search(vec, topK*3)
		s.logf("rankForVector raw NDC hits (limit=%d): %s", topK*3, formatHitsDebug(rawNDCHits))
		ndcHits = applySourceWeight(rawNDCHits, cfg.WeightNDC)
		s.logf("rankForVector weighted NDC hits: %s", formatHitsDebug(ndcHits))
	}

	var suggestions []Suggestion
	var ndcSuggestions []Suggestion

	switch cfg.Mode {
	case ModeSeeded:
		if cfg.Cluster.Enabled {
			s.logf("rankForVector clustering seed hits with threshold %.6f", cfg.Cluster.Threshold)
			seedHits = clusterHits(seedHits, cfg.Cluster.Threshold)
			s.logf("rankForVector seed hits after clustering: %s", formatHitsDebug(seedHits))
		}
		limitedSeeds := limitHits(seedHits, topK)
		s.logf("rankForVector limited seed hits (ModeSeeded): %s", formatHitsDebug(limitedSeeds))
		suggestions = hitsToSuggestions(limitedSeeds)
		s.logf("rankForVector seed suggestions (ModeSeeded): %s", formatSuggestionsDebug(suggestions))
		if cfg.UseNDC {
			limitedNDC := limitHits(ndcHits, topK)
			s.logf("rankForVector limited ndc hits (ModeSeeded): %s", formatHitsDebug(limitedNDC))
			ndcSuggestions = hitsToSuggestions(limitedNDC)
			s.logf("rankForVector ndc suggestions (ModeSeeded): %s", formatSuggestionsDebug(ndcSuggestions))
		}
	case ModeSplit:
		if cfg.Cluster.Enabled {
			s.logf("rankForVector clustering seed hits with threshold %.6f", cfg.Cluster.Threshold)
			seedHits = clusterHits(seedHits, cfg.Cluster.Threshold)
			s.logf("rankForVector seed hits after clustering: %s", formatHitsDebug(seedHits))
			if cfg.UseNDC {
				s.logf("rankForVector clustering ndc hits with threshold %.6f", cfg.Cluster.Threshold)
				ndcHits = clusterHits(ndcHits, cfg.Cluster.Threshold)
				s.logf("rankForVector ndc hits after clustering: %s", formatHitsDebug(ndcHits))
			}
		}
		limitedSeeds := limitHits(seedHits, topK)
		s.logf("rankForVector limited seed hits (ModeSplit): %s", formatHitsDebug(limitedSeeds))
		suggestions = hitsToSuggestions(limitedSeeds)
		s.logf("rankForVector seed suggestions (ModeSplit): %s", formatSuggestionsDebug(suggestions))
		if cfg.UseNDC {
			limitedNDC := limitHits(ndcHits, topK)
			s.logf("rankForVector limited ndc hits (ModeSplit): %s", formatHitsDebug(limitedNDC))
			ndcSuggestions = hitsToSuggestions(limitedNDC)
			s.logf("rankForVector ndc suggestions (ModeSplit): %s", formatSuggestionsDebug(ndcSuggestions))
		}
	case ModeMixed:
		weighted := make([]Hit, 0, len(seedHits)+len(ndcHits))
		weighted = append(weighted, seedHits...)
		if cfg.UseNDC {
			weighted = append(weighted, ndcHits...)
		}
		s.logf("rankForVector combined hits before clustering: %s", formatHitsDebug(weighted))
		if cfg.Cluster.Enabled {
			s.logf("rankForVector clustering mixed hits with threshold %.6f", cfg.Cluster.Threshold)
			weighted = clusterHits(weighted, cfg.Cluster.Threshold)
			s.logf("rankForVector mixed hits after clustering: %s", formatHitsDebug(weighted))
		}
		sort.Slice(weighted, func(i, j int) bool {
			if weighted[i].Score == weighted[j].Score {
				return weighted[i].Label < weighted[j].Label
			}
			return weighted[i].Score > weighted[j].Score
		})
		s.logf("rankForVector sorted mixed hits: %s", formatHitsDebug(weighted))
		limitedMixed := limitHits(weighted, topK)
		s.logf("rankForVector limited mixed hits (ModeMixed): %s", formatHitsDebug(limitedMixed))
		suggestions = hitsToSuggestions(limitedMixed)
		s.logf("rankForVector mixed suggestions (ModeMixed): %s", formatSuggestionsDebug(suggestions))
	default:
		if cfg.Cluster.Enabled {
			s.logf("rankForVector clustering seed hits with threshold %.6f", cfg.Cluster.Threshold)
			seedHits = clusterHits(seedHits, cfg.Cluster.Threshold)
			s.logf("rankForVector seed hits after clustering: %s", formatHitsDebug(seedHits))
		}
		limitedSeeds := limitHits(seedHits, topK)
		s.logf("rankForVector limited seed hits (default): %s", formatHitsDebug(limitedSeeds))
		suggestions = hitsToSuggestions(limitedSeeds)
		s.logf("rankForVector default suggestions: %s", formatSuggestionsDebug(suggestions))
	}

	if !cfg.UseNDC {
		s.logf("rankForVector NDC disabled")
	} else if len(ndcSuggestions) == 0 {
		s.logf("rankForVector ndc suggestions empty")
	}
	s.logf("rankForVector final suggestions: %s", formatSuggestionsDebug(suggestions))
	if len(ndcSuggestions) > 0 {
		s.logf("rankForVector final ndc suggestions: %s", formatSuggestionsDebug(ndcSuggestions))
	}
	return ResultRow{
		Text:           originalText,
		Suggestions:    suggestions,
		NDCSuggestions: ndcSuggestions,
	}
}

func limitHits(hits []Hit, k int) []Hit {
	if len(hits) <= k {
		return hits
	}
	return hits[:k]
}

const (
	biasScale    = 1e-6
	biasDivisor  = float32(1<<32 - 1)
	maxWeightVal = 1.0
)

func applySourceWeight(hits []Hit, weight float32) []Hit {
	if len(hits) == 0 {
		return hits
	}
	if weight < 0 {
		weight = 0
	}
	if weight > maxWeightVal {
		weight = maxWeightVal
	}
	for i := range hits {
		score := hits[i].Score
		if score < 0 {
			score = 0
		}
		score *= weight
		if score > maxWeightVal {
			score = maxWeightVal
		}
		score += tinyBias(hits[i].Label)
		hits[i].Score = score
	}
	return hits
}

func tinyBias(label string) float32 {
	if label == "" {
		return 0
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(label))
	return (float32(h.Sum32()) / biasDivisor) * biasScale
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

func formatVectorDebug(vec []float32) string {
	return fmt.Sprintf("vector_len=%d vector=%v", len(vec), vec)
}

func formatHitsDebug(hits []Hit) string {
	if len(hits) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteString("[\n")
	for i, hit := range hits {
		fmt.Fprintf(&b, "  [%d] label=%q score=%.6f source=%q vector_len=%d vector=%v\n", i, hit.Label, hit.Score, hit.Source, len(hit.Vector), hit.Vector)
	}
	b.WriteString("]")
	return b.String()
}

func formatSuggestionsDebug(suggestions []Suggestion) string {
	if len(suggestions) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteString("[\n")
	for i, sug := range suggestions {
		fmt.Fprintf(&b, "  [%d] label=%q score=%.6f source=%q\n", i, sug.Label, sug.Score, sug.Source)
	}
	b.WriteString("]")
	return b.String()
}

func formatResultRowDebug(row ResultRow) string {
	var b strings.Builder
	fmt.Fprintf(&b, "text=%q\n", row.Text)
	b.WriteString("suggestions=")
	b.WriteString(formatSuggestionsDebug(row.Suggestions))
	b.WriteString("\nndcSuggestions=")
	b.WriteString(formatSuggestionsDebug(row.NDCSuggestions))
	return b.String()
}
