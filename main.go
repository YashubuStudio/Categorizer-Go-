package main

// Fyne GUI based categorization assistant.
// Features:
//   * Multiple ranking modes (seeded / mixed / split) with configurable Top-k.
//   * User category and NDC dictionaries embedded via ONNX encoder with in-memory
//     and disk caching.
//   * CSV / text batch classification, CSV export, progress indicator and log area.
//   * Optional similarity-based clustering of suggestions.
//   * Settings dialog for thresholds, weights, and clustering options.

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/binary"
	"encoding/csv"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"golang.org/x/text/unicode/norm"

	emb "yashubustudio/categorizer/emb"
)

const (
	ModeSeeded = "seeded"
	ModeMixed  = "mixed"
	ModeSplit  = "split"
)

var modeChoices = []struct {
	Label string
	Value string
}{
	{Label: "項目のみ", Value: ModeSeeded},
	{Label: "混合 (項目+NDC)", Value: ModeMixed},
	{Label: "別枠（項目/NDC）", Value: ModeSplit},
}

// ------------------------------
// 設定・閾値
// ------------------------------

type Threshold struct {
	Top1     float32 // 例: 0.45
	Margin12 float32 // 例: 0.03
	Mean     float32 // 例: 0.50
}

type ClusterCfg struct {
	Enabled   bool
	Threshold float32 // tau 例: 0.80
}

type Config struct {
	TopK      int
	Mode      string
	UseNDC    bool
	WeightNDC float32
	SeedBias  float32
	Thresh    Threshold

	ClusterCfg ClusterCfg

	OrtDLL        string
	ModelPath     string
	TokenizerPath string
	MaxSeqLen     int

	CacheDir string
}

func defaultConfig() Config {
	return Config{
		TopK:          3,
		Mode:          ModeMixed,
		UseNDC:        true,
		WeightNDC:     0.85,
		SeedBias:      0.03,
		Thresh:        Threshold{Top1: 0.45, Margin12: 0.03, Mean: 0.50},
		ClusterCfg:    ClusterCfg{Enabled: false, Threshold: 0.80},
		OrtDLL:        "./onnixruntime-win/lib/onnxruntime.dll",
		ModelPath:     "./models/bge-m3/model.onnx",
		TokenizerPath: "./models/bge-m3/tokenizer.json",
		MaxSeqLen:     512,
		CacheDir:      "./cache",
	}
}

func sanitizeConfig(cfg Config) Config {
	if cfg.TopK < 3 {
		cfg.TopK = 3
	}
	if cfg.TopK > 5 {
		cfg.TopK = 5
	}
	switch cfg.Mode {
	case ModeSeeded, ModeMixed, ModeSplit:
	default:
		cfg.Mode = ModeMixed
	}
	if cfg.WeightNDC < 0.5 {
		cfg.WeightNDC = 0.5
	}
	if cfg.WeightNDC > 1.2 {
		cfg.WeightNDC = 1.2
	}
	if cfg.SeedBias < 0 {
		cfg.SeedBias = 0
	}
	if cfg.SeedBias > 0.2 {
		cfg.SeedBias = 0.2
	}
	if cfg.ClusterCfg.Threshold <= 0 {
		cfg.ClusterCfg.Threshold = 0.80
	}
	if cfg.Thresh.Top1 <= 0 {
		cfg.Thresh.Top1 = 0.45
	}
	if cfg.Thresh.Margin12 < 0 {
		cfg.Thresh.Margin12 = 0.03
	}
	if cfg.Thresh.Mean <= 0 {
		cfg.Thresh.Mean = 0.50
	}
	return cfg
}

// ------------------------------
// データ構造
// ------------------------------

type Candidate struct {
	Label  string
	Vec    []float32
	Source string // "seed" or "ndc"
}

type Suggestion struct {
	Label   string
	Score   float32
	Source  string
	Aliases []string
}

type ResultRow struct {
	Text            string
	Suggestions     []Suggestion
	SeedSuggestions []Suggestion
	NDCSuggestions  []Suggestion
	NeedReview      bool
}

// ------------------------------
// 初期カテゴリ
// ------------------------------

var defaultUserCategories = []string{
	"CG・デジタルアーカイブ",
	"VR空間",
	"アバター",
	"インタラクション",
	"エージェント",
	"コミュニケーション",
	"ソーシャルVR",
	"可視化",
	"工学・サイエンスコミュニケーション",
	"応用数理",
	"感覚・知覚",
	"教育①",
	"教育②",
	"機械学習",
	"社会",
}

// ------------------------------
// 埋め込みキャッシュ（メモリ＋ディスク）
// ------------------------------

type embedCache struct {
	mu      sync.RWMutex
	m       map[string][]float32
	dir     string
	modelID string
}

func newEmbedCache(dir, modelID string) *embedCache {
	return &embedCache{m: make(map[string][]float32), dir: dir, modelID: modelID}
}

func (c *embedCache) get(key string) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.m[key]
	return v, ok
}

func (c *embedCache) put(key string, v []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[key] = v
}

func (c *embedCache) load(key string) ([]float32, bool, error) {
	if c.dir == "" {
		return nil, false, nil
	}
	path := filepath.Join(c.dir, key+".bin")
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}
	if len(data) < 4 {
		return nil, false, fmt.Errorf("cache file broken: %s", path)
	}
	length := binary.LittleEndian.Uint32(data[:4])
	need := int(length) * 4
	if len(data) < 4+need {
		return nil, false, fmt.Errorf("cache truncated: %s", path)
	}
	vec := make([]float32, int(length))
	if err := binary.Read(bytes.NewReader(data[4:4+need]), binary.LittleEndian, vec); err != nil {
		return nil, false, err
	}
	return vec, true, nil
}

func (c *embedCache) save(key string, v []float32) error {
	if c.dir == "" {
		return nil
	}
	path := filepath.Join(c.dir, key+".bin")
	buf := &bytes.Buffer{}
	_ = binary.Write(buf, binary.LittleEndian, uint32(len(v)))
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// ------------------------------
// 文字列正規化
// ------------------------------

func normalize(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Join(strings.Fields(s), " ")
	return norm.NFKC.String(s)
}

// ------------------------------
// コサイン（float32）
// ------------------------------

func cosine32(a, b []float32) float32 {
	var dot, na, nb float32
	for i := range a {
		af, bf := a[i], b[i]
		dot += af * bf
		na += af * af
		nb += bf * bf
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(na))) * float32(math.Sqrt(float64(nb))))
}

func tinyBias(label string) float32 {
	h := fnv32(label)
	return float32(h%997) * 1e-9
}

func fnv32(s string) uint32 {
	const (
		offset32 = 2166136261
		prime32  = 16777619
	)
	var h uint32 = offset32
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= prime32
	}
	return h
}

func clamp01(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// ------------------------------
// サービス
// ------------------------------

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

	svc := &Service{
		cfg:      cfg,
		emb:      enc,
		cache:    newEmbedCache(cfg.CacheDir, filepath.Base(cfg.ModelPath)),
		userCats: uniqueNormalized(defaultUserCategories),
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

func cacheKey(text, model string) string {
	h := sha1.Sum([]byte(text + "|" + model))
	return hex.EncodeToString(h[:])
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
		// keep combined as seeds, ndc displayed separately
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

func uniqueNormalized(labels []string) []string {
	seen := make(map[string]struct{})
	res := make([]string, 0, len(labels))
	for _, lab := range labels {
		normed := normalize(lab)
		if normed == "" {
			continue
		}
		if _, ok := seen[normed]; ok {
			continue
		}
		seen[normed] = struct{}{}
		res = append(res, normed)
	}
	return res
}

// ------------------------------
// GUI (Fyne)
// ------------------------------

type tableColumn struct {
	Title  string
	Width  float32
	Render func(ResultRow) string
}

type uiState struct {
	service *Service
	cfg     Config

	w             fyne.Window
	input         *widget.Entry
	log           *widget.Entry
	status        *widget.Label
	progress      *widget.ProgressBar
	configSummary *widget.Label
	resTbl        *widget.Table
	columns       []tableColumn
	rows          []ResultRow

	classifyBtn *widget.Button
	exportBtn   *widget.Button
	loadBtn     *widget.Button
	catBtn      *widget.Button
}

func buildUI(a fyne.App, svc *Service) *uiState {
	u := &uiState{service: svc}
	u.cfg = svc.Config()
	u.w = a.NewWindow("Vector Categorizer - Seeded & NDC")

	u.input = widget.NewMultiLineEntry()
	u.input.SetPlaceHolder("ここに文章を入力（1行=1件）")

	u.log = widget.NewMultiLineEntry()
	u.log.SetPlaceHolder("処理ログ")
	u.log.Disable()

	u.status = widget.NewLabel("準備完了")
	u.progress = widget.NewProgressBar()
	u.progress.Hide()
	u.configSummary = widget.NewLabel("")

	u.classifyBtn = widget.NewButtonWithIcon("分類実行", theme.ConfirmIcon(), func() { u.onClassify() })
	u.exportBtn = widget.NewButtonWithIcon("CSVエクスポート", theme.DocumentSaveIcon(), func() { u.onExport() })
	u.loadBtn = widget.NewButtonWithIcon("ファイル読込", theme.FolderOpenIcon(), func() { u.onLoadFile() })
	settingsBtn := widget.NewButtonWithIcon("設定", theme.SettingsIcon(), func() { u.openSettings() })
	u.catBtn = widget.NewButtonWithIcon("カテゴリ読込", theme.ContentAddIcon(), func() { u.onLoadCategories() })

	u.columns = u.makeColumns(u.cfg)
	u.resTbl = widget.NewTable(
		func() (int, int) {
			cols := len(u.columns)
			if cols == 0 {
				cols = 1
			}
			return len(u.rows) + 1, cols
		},
		func() fyne.CanvasObject {
			lbl := widget.NewLabel("")
			lbl.Wrapping = fyne.TextWrapWord
			return lbl
		},
		func(id widget.TableCellID, obj fyne.CanvasObject) {
			lbl := obj.(*widget.Label)
			if id.Row == 0 {
				if id.Col < len(u.columns) {
					lbl.SetText(u.columns[id.Col].Title)
				} else {
					lbl.SetText("")
				}
				lbl.Alignment = fyne.TextAlignCenter
				lbl.TextStyle = fyne.TextStyle{Bold: true}
				u.resTbl.SetRowHeight(id.Row, 32)
				return
			}
			lbl.TextStyle = fyne.TextStyle{}
			lbl.Alignment = fyne.TextAlignLeading
			lbl.Wrapping = fyne.TextWrapWord
			rowIdx := id.Row - 1
			if rowIdx >= len(u.rows) {
				lbl.SetText("")
				return
			}
			if id.Col >= len(u.columns) {
				lbl.SetText("")
				return
			}
			val := u.columns[id.Col].Render(u.rows[rowIdx])
			lbl.SetText(val)
			if id.Col == 0 {
				width := u.columns[id.Col].Width
				need := wrappedHeightFor(val, width)
				if need < 32 {
					need = 32
				}
				u.resTbl.SetRowHeight(id.Row, need)
			}
		},
	)
	u.applyColumnWidths()

	controlRow1 := container.NewGridWithColumns(3, u.classifyBtn, u.exportBtn, settingsBtn)
	controlRow2 := container.NewGridWithColumns(2, u.loadBtn, u.catBtn)
	left := container.NewVBox(
		widget.NewLabelWithStyle("入力テキスト", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewMax(u.input),
		controlRow1,
		controlRow2,
		widget.NewSeparator(),
		widget.NewLabelWithStyle("進捗", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		u.progress,
		u.status,
		widget.NewSeparator(),
		widget.NewLabelWithStyle("設定サマリ", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		u.configSummary,
		widget.NewSeparator(),
		widget.NewLabelWithStyle("ログ", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		container.NewMax(u.log),
	)

	right := container.NewBorder(nil, nil, nil, nil, u.resTbl)
	split := container.NewHSplit(left, right)
	split.Offset = 0.35

	u.w.SetContent(split)
	u.w.Resize(fyne.NewSize(1180, 760))
	u.updateConfigSummary()
	return u
}

func (u *uiState) makeColumns(cfg Config) []tableColumn {
	cols := []tableColumn{
		{Title: "本文", Width: 360, Render: func(r ResultRow) string { return r.Text }},
	}
	for i := 0; i < cfg.TopK; i++ {
		idx := i
		cols = append(cols, tableColumn{
			Title:  fmt.Sprintf("候補%d", i+1),
			Width:  190,
			Render: func(r ResultRow) string { return formatSuggestionAt(r.Suggestions, idx, true) },
		})
	}
	cols = append(cols, tableColumn{
		Title: "要確認",
		Width: 80,
		Render: func(r ResultRow) string {
			if r.NeedReview {
				return "要確認"
			}
			return ""
		},
	})
	if cfg.Mode == ModeSplit {
		for i := 0; i < cfg.TopK; i++ {
			idx := i
			cols = append(cols, tableColumn{
				Title:  fmt.Sprintf("NDC%d", i+1),
				Width:  190,
				Render: func(r ResultRow) string { return formatSuggestionAt(r.NDCSuggestions, idx, false) },
			})
		}
	} else {
		cols = append(cols, tableColumn{
			Title:  "ソース",
			Width:  120,
			Render: func(r ResultRow) string { return suggestionSources(r.Suggestions) },
		})
	}
	return cols
}

func (u *uiState) applyColumnWidths() {
	for i, col := range u.columns {
		u.resTbl.SetColumnWidth(i, col.Width)
	}
	u.resTbl.SetRowHeight(0, 32)
}

func (u *uiState) rebuildTableColumns(cfg Config) {
	u.columns = u.makeColumns(cfg)
	u.applyColumnWidths()
	u.resTbl.Refresh()
}

func (u *uiState) setBusy(b bool) {
	if b {
		u.classifyBtn.Disable()
		u.exportBtn.Disable()
		u.loadBtn.Disable()
		u.catBtn.Disable()
	} else {
		u.classifyBtn.Enable()
		u.exportBtn.Enable()
		u.loadBtn.Enable()
		u.catBtn.Enable()
	}
}

func (u *uiState) appendLog(msg string) {
	now := time.Now().Format("15:04:05")
	line := fmt.Sprintf("[%s] %s", now, msg)
	lines := append(strings.Split(u.log.Text, "\n"), line)
	if len(lines) > 1 && lines[0] == "" {
		lines = lines[1:]
	}
	if len(lines) > 200 {
		lines = lines[len(lines)-200:]
	}
	u.log.SetText(strings.Join(lines, "\n"))
}

func (u *uiState) updateConfigSummary() {
	cfg := u.cfg
	seeds, ndc := u.service.CandidateStats()
	ndcStatus := "OFF"
	if cfg.Mode == ModeSplit || cfg.UseNDC {
		ndcStatus = fmt.Sprintf("ON (w=%.2f)", cfg.WeightNDC)
	}
	clusterStatus := "OFF"
	if cfg.ClusterCfg.Enabled {
		clusterStatus = fmt.Sprintf("ON (τ=%.2f)", cfg.ClusterCfg.Threshold)
	}
	modeLabel := cfg.Mode
	for _, c := range modeChoices {
		if c.Value == cfg.Mode {
			modeLabel = c.Label
			break
		}
	}
	summary := fmt.Sprintf("モード:%s / Top-k:%d / SeedBias:%.2f / NDC:%s / クラスタ:%s / カテゴリ:%d / NDC辞書:%d",
		modeLabel, cfg.TopK, cfg.SeedBias, ndcStatus, clusterStatus, seeds, ndc)
	u.configSummary.SetText(summary)
}

func (u *uiState) onClassify() {
	lines := splitNonEmptyLines(u.input.Text)
	if len(lines) == 0 {
		dialog.ShowInformation("情報", "入力テキストが空です", u.w)
		return
	}
	total := len(lines)
	u.progress.Min = 0
	u.progress.Max = float64(total)
	u.progress.SetValue(0)
	u.progress.Refresh()
	u.progress.Show()
	u.status.SetText("処理中...")
	u.setBusy(true)
	u.appendLog(fmt.Sprintf("分類開始 (%d件)", total))
	start := time.Now()

	go func(entries []string) {
		rows, err := u.service.ClassifyAll(context.Background(), entries, func(done, total int) {
			// そのまま更新してOK（Fyneはスレッドセーフ）
			u.progress.SetValue(float64(done))
			u.status.SetText(fmt.Sprintf("処理中 %d/%d", done, total))
		})

		// ここも直接更新でOK
		u.setBusy(false)
		u.progress.Hide()
		if err != nil {
			dialog.ShowError(err, u.w)
			u.status.SetText("エラー")
			u.appendLog(fmt.Sprintf("エラー: %v", err))
			return
		}
		u.rows = rows
		u.resTbl.Refresh()
		elapsed := time.Since(start).Seconds()
		u.status.SetText(fmt.Sprintf("完了 %d件 (%.1fs)", len(rows), elapsed))
		u.appendLog(fmt.Sprintf("分類完了 %d件 (%.1fs)", len(rows), elapsed))
	}(lines)
}

func (u *uiState) onExport() {
	if len(u.rows) == 0 {
		dialog.ShowInformation("情報", "出力データがありません", u.w)
		return
	}
	cfg := u.cfg
	fd := dialog.NewFileSave(func(uc fyne.URIWriteCloser, err error) {
		if err != nil || uc == nil {
			return
		}
		defer uc.Close()
		w := csv.NewWriter(uc)
		header := []string{"text"}
		for i := 0; i < cfg.TopK; i++ {
			header = append(header,
				fmt.Sprintf("suggestion%d", i+1),
				fmt.Sprintf("score%d", i+1),
				fmt.Sprintf("source%d", i+1))
		}
		if cfg.Mode == ModeSplit {
			for i := 0; i < cfg.TopK; i++ {
				header = append(header,
					fmt.Sprintf("ndc%d", i+1),
					fmt.Sprintf("ndc_score%d", i+1))
			}
		}
		header = append(header, "need_review")
		_ = w.Write(header)
		for _, r := range u.rows {
			record := []string{r.Text}
			for i := 0; i < cfg.TopK; i++ {
				if sug, ok := suggestionAt(r.Suggestions, i); ok {
					record = append(record, suggestionLabel(sug), fmt.Sprintf("%.3f", sug.Score), sug.Source)
				} else {
					record = append(record, "", "", "")
				}
			}
			if cfg.Mode == ModeSplit {
				for i := 0; i < cfg.TopK; i++ {
					if sug, ok := suggestionAt(r.NDCSuggestions, i); ok {
						record = append(record, suggestionLabel(sug), fmt.Sprintf("%.3f", sug.Score))
					} else {
						record = append(record, "", "")
					}
				}
			}
			if r.NeedReview {
				record = append(record, "yes")
			} else {
				record = append(record, "no")
			}
			_ = w.Write(record)
		}
		w.Flush()
		u.appendLog(fmt.Sprintf("CSVエクスポート完了 (%d件)", len(u.rows)))
	}, u.w)
	fd.SetFileName("result.csv")
	fd.Show()
}

func (u *uiState) openSettings() {
	cfg := u.cfg
	topkSel := widget.NewSelect([]string{"3", "4", "5"}, nil)
	topkSel.SetSelected(strconv.Itoa(cfg.TopK))

	modeLabels := make([]string, len(modeChoices))
	modeMap := make(map[string]string, len(modeChoices))
	activeLabel := modeChoices[1].Label
	for i, c := range modeChoices {
		modeLabels[i] = c.Label
		modeMap[c.Label] = c.Value
		if c.Value == cfg.Mode {
			activeLabel = c.Label
		}
	}
	modeSel := widget.NewSelect(modeLabels, nil)
	modeSel.SetSelected(activeLabel)

	ndcCheck := widget.NewCheck("NDC を候補に含める", nil)
	ndcCheck.SetChecked(cfg.UseNDC || cfg.Mode == ModeSplit)
	weightEntry := widget.NewEntry()
	weightEntry.SetText(fmt.Sprintf("%.2f", cfg.WeightNDC))
	seedBiasEntry := widget.NewEntry()
	seedBiasEntry.SetText(fmt.Sprintf("%.2f", cfg.SeedBias))

	clusterCheck := widget.NewCheck("類似カテゴリをまとめる", nil)
	clusterCheck.SetChecked(cfg.ClusterCfg.Enabled)
	clusterTauEntry := widget.NewEntry()
	clusterTauEntry.SetText(fmt.Sprintf("%.2f", cfg.ClusterCfg.Threshold))

	top1Entry := widget.NewEntry()
	top1Entry.SetText(fmt.Sprintf("%.2f", cfg.Thresh.Top1))
	m12Entry := widget.NewEntry()
	m12Entry.SetText(fmt.Sprintf("%.2f", cfg.Thresh.Margin12))
	meanEntry := widget.NewEntry()
	meanEntry.SetText(fmt.Sprintf("%.2f", cfg.Thresh.Mean))

	updateControls := func() {
		modeVal := modeMap[modeSel.Selected]
		if modeVal == "" {
			modeVal = cfg.Mode
		}
		if modeVal == ModeSplit {
			ndcCheck.SetChecked(true)
			ndcCheck.Disable()
			weightEntry.Enable()
		} else {
			ndcCheck.Enable()
			if ndcCheck.Checked {
				weightEntry.Enable()
			} else {
				weightEntry.Disable()
			}
		}
	}
	modeSel.OnChanged = func(string) { updateControls() }
	ndcCheck.OnChanged = func(b bool) {
		modeVal := modeMap[modeSel.Selected]
		if modeVal == "" {
			modeVal = cfg.Mode
		}
		if modeVal != ModeSplit {
			if b {
				weightEntry.Enable()
			} else {
				weightEntry.Disable()
			}
		}
	}
	updateControls()

	form := &widget.Form{Items: []*widget.FormItem{
		{Text: "Top-k", Widget: topkSel},
		{Text: "ランキングモード", Widget: modeSel},
		{Text: "NDC使用", Widget: ndcCheck},
		{Text: "NDC重み", Widget: weightEntry},
		{Text: "Seedバイアス", Widget: seedBiasEntry},
		{Text: "閾値 Top1", Widget: top1Entry},
		{Text: "閾値 Top1-Top2", Widget: m12Entry},
		{Text: "閾値 平均", Widget: meanEntry},
		{Text: "クラスタリング", Widget: clusterCheck},
		{Text: "クラスタ閾値", Widget: clusterTauEntry},
	}}

	dialog.NewCustomConfirm("設定", "OK", "キャンセル", form, func(ok bool) {
		if !ok {
			return
		}
		newCfg := cfg
		if v, err := strconv.Atoi(topkSel.Selected); err == nil {
			newCfg.TopK = v
		}
		modeVal := cfg.Mode
		if sel := modeSel.Selected; sel != "" {
			if v, ok := modeMap[sel]; ok {
				modeVal = v
			}
		}
		newCfg.Mode = modeVal
		if modeVal == ModeSplit {
			newCfg.UseNDC = true
		} else {
			newCfg.UseNDC = ndcCheck.Checked
		}
		if v, err := strconv.ParseFloat(weightEntry.Text, 32); err == nil {
			newCfg.WeightNDC = float32(v)
		}
		if v, err := strconv.ParseFloat(seedBiasEntry.Text, 32); err == nil {
			newCfg.SeedBias = float32(v)
		}
		if v, err := strconv.ParseFloat(top1Entry.Text, 32); err == nil {
			newCfg.Thresh.Top1 = float32(v)
		}
		if v, err := strconv.ParseFloat(m12Entry.Text, 32); err == nil {
			newCfg.Thresh.Margin12 = float32(v)
		}
		if v, err := strconv.ParseFloat(meanEntry.Text, 32); err == nil {
			newCfg.Thresh.Mean = float32(v)
		}
		newCfg.ClusterCfg.Enabled = clusterCheck.Checked
		if v, err := strconv.ParseFloat(clusterTauEntry.Text, 32); err == nil {
			newCfg.ClusterCfg.Threshold = float32(v)
		}

		newCfg = u.service.UpdateConfig(newCfg)
		u.cfg = newCfg
		u.rebuildTableColumns(newCfg)
		u.updateConfigSummary()
		u.appendLog("設定を更新しました")
	}, u.w).Show()
}

func (u *uiState) onLoadFile() {
	fd := dialog.NewFileOpen(func(rc fyne.URIReadCloser, err error) {
		if err != nil || rc == nil {
			return
		}
		defer rc.Close()
		data, err := io.ReadAll(rc)
		if err != nil {
			dialog.ShowError(err, u.w)
			return
		}
		lines, err := parseInputFile(rc.URI(), data)
		if err != nil {
			dialog.ShowError(err, u.w)
			return
		}
		u.input.SetText(strings.Join(lines, "\n"))
		u.appendLog(fmt.Sprintf("ファイル読込: %s (%d件)", filepath.Base(rc.URI().Path()), len(lines)))
	}, u.w)
	fd.SetFilter(storage.NewExtensionFileFilter([]string{".txt", ".csv", ".tsv"}))
	fd.Show()
}

func (u *uiState) onLoadCategories() {
	fd := dialog.NewFileOpen(func(rc fyne.URIReadCloser, err error) {
		if err != nil || rc == nil {
			return
		}
		defer rc.Close()
		data, err := io.ReadAll(rc)
		if err != nil {
			dialog.ShowError(err, u.w)
			return
		}
		labels := parseCategoryText(string(data))
		if len(labels) == 0 {
			dialog.ShowInformation("情報", "カテゴリが検出できませんでした", u.w)
			return
		}
		count, err := u.service.UpdateCategories(context.Background(), labels)
		if err != nil {
			dialog.ShowError(err, u.w)
			return
		}
		u.updateConfigSummary()
		u.appendLog(fmt.Sprintf("カテゴリを更新 (%d件)", count))
	}, u.w)
	fd.SetFilter(storage.NewExtensionFileFilter([]string{".txt", ".csv"}))
	fd.Show()
}

// ------------------------------
// ヘルパ
// ------------------------------

func suggestionAt(s []Suggestion, idx int) (Suggestion, bool) {
	if idx < 0 || idx >= len(s) {
		return Suggestion{}, false
	}
	return s[idx], true
}

func suggestionLabel(s Suggestion) string {
	if len(s.Aliases) == 0 {
		return s.Label
	}
	return fmt.Sprintf("%s [%s]", s.Label, strings.Join(s.Aliases, " / "))
}

func formatSuggestionAt(list []Suggestion, idx int, showSource bool) string {
	if sug, ok := suggestionAt(list, idx); ok {
		label := suggestionLabel(sug)
		if showSource && sug.Source != "" {
			return fmt.Sprintf("%s\n%.3f (%s)", label, sug.Score, sug.Source)
		}
		return fmt.Sprintf("%s\n%.3f", label, sug.Score)
	}
	return ""
}

func suggestionSources(list []Suggestion) string {
	seen := make(map[string]struct{})
	out := make([]string, 0, len(list))
	for _, s := range list {
		for _, part := range strings.Split(s.Source, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if _, ok := seen[part]; ok {
				continue
			}
			seen[part] = struct{}{}
			out = append(out, part)
		}
	}
	return strings.Join(out, ",")
}

func wrappedHeightFor(text string, colWidth float32) float32 {
	lbl := widget.NewLabel(text)
	lbl.Wrapping = fyne.TextWrapWord
	lbl.Resize(fyne.NewSize(colWidth, 0))
	return lbl.MinSize().Height + 8
}

func splitNonEmptyLines(s string) []string {
	scanner := bufio.NewScanner(strings.NewReader(s))
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
	lines := make([]string, 0)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func parseInputFile(uri fyne.URI, data []byte) ([]string, error) {
	ext := strings.ToLower(filepath.Ext(uri.Path()))
	switch ext {
	case ".csv", ".tsv":
		delim := ','
		if ext == ".tsv" {
			delim = '\t'
		}
		return parseCSVTexts(data, delim)
	default:
		return splitNonEmptyLines(string(data)), nil
	}
}

func parseCSVTexts(data []byte, delim rune) ([]string, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.Comma = delim
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, errors.New("CSVが空です")
	}
	idx := detectTextColumn(records[0])
	start := 0
	if idx >= 0 {
		start = 1
	} else {
		idx = 0
	}
	res := make([]string, 0, len(records))
	for i := start; i < len(records); i++ {
		row := records[i]
		if idx >= len(row) {
			continue
		}
		val := strings.TrimSpace(row[idx])
		if val != "" {
			res = append(res, val)
		}
	}
	return res, nil
}

func detectTextColumn(header []string) int {
	if len(header) == 0 {
		return -1
	}
	candidates := []string{"text", "本文", "content", "body", "description", "message"}
	for idx, h := range header {
		normalized := strings.ToLower(normalize(h))
		for _, c := range candidates {
			if normalized == c {
				return idx
			}
		}
	}
	return -1
}

func parseCategoryText(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool {
		switch r {
		case '\n', '\r', ',', ';', '\t':
			return true
		default:
			return false
		}
	})
	res := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f != "" {
			res = append(res, f)
		}
	}
	return res
}

// ------------------------------
// main
// ------------------------------

func main() {
	cfg := defaultConfig()
	ensureDirs(cfg.CacheDir)

	svc, err := NewService(cfg)
	if err != nil {
		fmt.Println("初期化エラー:", err)
		fmt.Println("Config の OrtDLL / ModelPath / TokenizerPath を確認してください。")
		return
	}
	defer svc.Close()

	a := app.New()
	u := buildUI(a, svc)
	u.w.ShowAndRun()
}

func ensureDirs(p string) {
	if p == "" {
		return
	}
	_ = os.MkdirAll(filepath.Clean(p), 0o755)
}
