package app

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	emb "yashubustudio/categorizer/emb"
)

type Service struct {
	mu       sync.RWMutex
	cfg      Config
	emb      *emb.Encoder
	cache    *embedCache
	userCats []string
	ndcItems []ndcItem
	candsCat []Candidate
	candsNDC []Candidate
	seedVec  map[string][]float32
	ndcVec   map[string][]float32
}

func NewService(cfg Config) (*Service, error) {
	cfg = sanitizeConfig(cfg)
	enc := &emb.Encoder{}
	if err := enc.Init(emb.Config{
		OrtDLL:        cfg.OrtDLL,
		ModelPath:     cfg.ModelPath,
		TokenizerPath: cfg.TokenizerPath,
		MaxSeqLen:     cfg.MaxSeqLen,
	}); err != nil {
		return nil, err
	}

	initialCats, fromFile, catErr := initialUserCategories(cfg.SeedFile)
	if catErr != nil {
		if errors.Is(catErr, os.ErrNotExist) {
			fmt.Printf("カテゴリシードファイルが見つかりませんでした (%s): %v\n", cfg.SeedFile, catErr)
		} else {
			fmt.Printf("カテゴリシードファイルの読み込みに失敗しました (%s): %v\n", cfg.SeedFile, catErr)
		}
	} else if fromFile {
		fmt.Printf("カテゴリシードを %s から読み込みました (%d件)\n", cfg.SeedFile, len(initialCats))
	}

	svc := &Service{
		cfg:      cfg,
		emb:      enc,
		cache:    newEmbedCache(cfg.CacheDir, filepath.Base(cfg.ModelPath)),
		userCats: initialCats,
		ndcItems: append([]ndcItem(nil), defaultNDCLabels...),
	}

	if err := svc.refreshNDCCandidates(context.Background()); err != nil {
		enc.Close()
		return nil, err
	}
	if _, err := svc.UpdateCategories(context.Background(), svc.userCats); err != nil {
		enc.Close()
		return nil, err
	}
	return svc, nil
}

func (s *Service) Close() {
	if s.emb != nil {
		s.emb.Close()
	}
}

func (s *Service) Config() Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg
}

func (s *Service) UpdateConfig(cfg Config) Config {
	cfg = sanitizeConfig(cfg)
	s.mu.Lock()
	s.cfg = cfg
	s.mu.Unlock()
	return cfg
}

func (s *Service) CandidateStats() (int, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.candsCat), len(s.candsNDC)
}

func (s *Service) refreshNDCCandidates(ctx context.Context) error {
	texts := make([]string, 0, len(s.ndcItems))
	for _, it := range s.ndcItems {
		texts = append(texts, normalize(it.Code+" "+it.Label))
	}
	cands, vecs, err := s.embedLabelSet(ctx, texts, "ndc")
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.candsNDC = cands
	s.ndcVec = vecs
	s.mu.Unlock()
	return nil
}

func (s *Service) UpdateCategories(ctx context.Context, labels []string) (int, error) {
	sanitized := uniqueNormalized(labels)
	cands, vecs, err := s.embedLabelSet(ctx, sanitized, "seed")
	if err != nil {
		return 0, err
	}
	s.mu.Lock()
	s.userCats = sanitized
	s.candsCat = cands
	s.seedVec = vecs
	s.mu.Unlock()
	return len(cands), nil
}

func (s *Service) embedLabelSet(ctx context.Context, labels []string, source string) ([]Candidate, map[string][]float32, error) {
	res := make([]Candidate, 0, len(labels))
	vecs := make(map[string][]float32, len(labels))
	seen := make(map[string]struct{})
	for _, raw := range labels {
		lab := normalize(raw)
		if lab == "" {
			continue
		}
		if _, ok := seen[lab]; ok {
			continue
		}
		seen[lab] = struct{}{}
		vec, err := s.EmbedCached(ctx, lab)
		if err != nil {
			return nil, nil, err
		}
		vecCopy := append([]float32(nil), vec...)
		res = append(res, Candidate{Label: lab, Vec: vecCopy, Source: source})
		vecs[lab] = vecCopy
	}
	return res, vecs, nil
}

func (s *Service) EmbedCached(ctx context.Context, text string) ([]float32, error) {
	key := cacheKey(text, s.cache.modelID)
	if v, ok := s.cache.get(key); ok {
		return v, nil
	}
	if v, ok, err := s.cache.load(key); err != nil {
		return nil, err
	} else if ok {
		s.cache.put(key, v)
		return v, nil
	}
	v, err := s.emb.Encode(text)
	if err != nil {
		return nil, err
	}
	s.cache.put(key, v)
	if err := s.cache.save(key, v); err != nil {
		fmt.Println("cache save error:", err)
	}
	return v, nil
}

func (s *Service) ClassifyAll(ctx context.Context, texts []string, progress func(done, total int)) ([]ResultRow, error) {
	results := make([]ResultRow, len(texts))
	total := len(texts)
	for i, t := range texts {
		row, err := s.RankOne(ctx, t)
		if err != nil {
			return nil, err
		}
		results[i] = row
		if progress != nil {
			progress(i+1, total)
		}
	}
	return results, nil
}

func (s *Service) RankOne(ctx context.Context, text string) (ResultRow, error) {
	row := ResultRow{Text: text}
	normalized := normalize(text)
	if normalized == "" {
		row.NeedReview = true
		return row, nil
	}

	vec, err := s.EmbedCached(ctx, normalized)
	if err != nil {
		return row, err
	}

	s.mu.RLock()
	cfg := s.cfg
	catCands := append([]Candidate(nil), s.candsCat...)
	ndcCands := append([]Candidate(nil), s.candsNDC...)
	seedVec := cloneVecMap(s.seedVec)
	ndcVec := cloneVecMap(s.ndcVec)
	s.mu.RUnlock()

	topK := cfg.TopK

	seeds := scoreCandidates(vec, catCands, 1.0, cfg.SeedBias)
	seeds = truncateSuggestions(seeds, topK)

	useNDC := (cfg.Mode != ModeSeeded && cfg.UseNDC) || cfg.Mode == ModeSplit
	ndc := []Suggestion{}
	if useNDC {
		ndc = scoreCandidates(vec, ndcCands, cfg.WeightNDC, 0)
		ndc = truncateSuggestions(ndc, topK)
	}

	combined := seeds
	if cfg.Mode == ModeMixed {
		combined = mergeSuggestions(seeds, ndc, topK)
	}
	if cfg.Mode == ModeSplit {
		combined = seeds
	}

	lookup := func(label string) []float32 {
		if v, ok := seedVec[label]; ok {
			return v
		}
		if v, ok := ndcVec[label]; ok {
			return v
		}
		return nil
	}
	if cfg.ClusterCfg.Enabled && cfg.ClusterCfg.Threshold > 0 {
		combined = clusterSuggestions(combined, cfg.ClusterCfg.Threshold, lookup)
		combined = truncateSuggestions(combined, topK)
	}

	row.Suggestions = combined
	row.SeedSuggestions = seeds
	row.NDCSuggestions = ndc

	ref := combined
	if len(ref) == 0 {
		if len(seeds) > 0 {
			ref = seeds
		} else {
			ref = ndc
		}
	}
	row.NeedReview = needReview(ref, cfg.Thresh)
	return row, nil
}

func cloneVecMap(src map[string][]float32) map[string][]float32 {
	if src == nil {
		return nil
	}
	dst := make(map[string][]float32, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func scoreCandidates(q []float32, cands []Candidate, weight, bias float32) []Suggestion {
	res := make([]Suggestion, 0, len(cands))
	for _, c := range cands {
		sc := cosine32(q, c.Vec)
		if sc < 0 {
			sc = 0
		}
		sc = sc*weight + bias + tinyBias(c.Label)
		res = append(res, Suggestion{Label: c.Label, Score: clamp01(sc), Source: c.Source})
	}
	sort.SliceStable(res, func(i, j int) bool { return res[i].Score > res[j].Score })
	return res
}

func truncateSuggestions(in []Suggestion, k int) []Suggestion {
	if len(in) == 0 {
		return nil
	}
	if k > len(in) {
		k = len(in)
	}
	out := make([]Suggestion, k)
	copy(out, in[:k])
	return out
}

func mergeSuggestions(a, b []Suggestion, topK int) []Suggestion {
	merged := make([]Suggestion, 0, len(a)+len(b))
	merged = append(merged, a...)
	merged = append(merged, b...)
	if len(merged) == 0 {
		return nil
	}
	sort.SliceStable(merged, func(i, j int) bool { return merged[i].Score > merged[j].Score })
	if topK > len(merged) {
		topK = len(merged)
	}
	out := make([]Suggestion, topK)
	copy(out, merged[:topK])
	return out
}

func needReview(sugs []Suggestion, thresh Threshold) bool {
	if len(sugs) == 0 {
		return true
	}
	top1 := sugs[0].Score
	top2 := float32(0)
	if len(sugs) > 1 {
		top2 = sugs[1].Score
	}
	mean := meanScore(sugs)
	return top1 < thresh.Top1 || (top1-top2) < thresh.Margin12 || mean < thresh.Mean
}

func meanScore(sugs []Suggestion) float32 {
	if len(sugs) == 0 {
		return 0
	}
	var sum float32
	for _, s := range sugs {
		sum += s.Score
	}
	return sum / float32(len(sugs))
}

func clusterSuggestions(in []Suggestion, tau float32, lookup func(string) []float32) []Suggestion {
	if len(in) <= 1 {
		return in
	}
	clusters := make([]Suggestion, 0, len(in))
	for _, sug := range in {
		vec := lookup(sug.Label)
		if vec == nil {
			clusters = append(clusters, sug)
			continue
		}
		merged := false
		for i := range clusters {
			other := lookup(clusters[i].Label)
			if other == nil {
				continue
			}
			if cosine32(vec, other) >= tau {
				clusters[i] = mergeSuggestion(clusters[i], sug)
				merged = true
				break
			}
		}
		if !merged {
			clusters = append(clusters, sug)
		}
	}
	sort.SliceStable(clusters, func(i, j int) bool { return clusters[i].Score > clusters[j].Score })
	return clusters
}

func mergeSuggestion(a, b Suggestion) Suggestion {
	label := a.Label
	score := a.Score
	if b.Score > score {
		label = b.Label
		score = b.Score
	}
	res := Suggestion{
		Label:  label,
		Score:  score,
		Source: mergeSources(a.Source, b.Source),
	}
	set := make(map[string]struct{})
	add := func(name string) {
		name = strings.TrimSpace(name)
		if name == "" || name == res.Label {
			return
		}
		if _, ok := set[name]; ok {
			return
		}
		set[name] = struct{}{}
		res.Aliases = append(res.Aliases, name)
	}
	add(a.Label)
	add(b.Label)
	for _, al := range a.Aliases {
		add(al)
	}
	for _, al := range b.Aliases {
		add(al)
	}
	return res
}

func mergeSources(values ...string) string {
	seen := make(map[string]struct{})
	order := make([]string, 0)
	for _, v := range values {
		parts := strings.Split(v, ",")
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			if _, ok := seen[p]; ok {
				continue
			}
			seen[p] = struct{}{}
			order = append(order, p)
		}
	}
	return strings.Join(order, ",")
}
