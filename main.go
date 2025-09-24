package main

// 最小実装: Fyne GUI で「ユーザー定義カテゴリ＋NDC（重み付き）」に対する
// ベクトル類似ランキング（Top-k）を行う。カテゴリ外の提案はしない。
// 依存:
//   go get fyne.io/fyne/v2
//   go get golang.org/x/text/unicode/norm
//   （あなたの既存エンコーダ emb パッケージに依存）
//
// 使い方:
//   1) Config のデフォルトに合わせて ONNX / モデル / トークナイザのパスを配置
//   2) 実行すると GUI が起動。左に入力（複数行=複数レコード）、右に結果表。
//   3) 「設定」で Top-k や NDC 重み、しきい値を調整可。
//   4) CSVエクスポートで結果を保存。
//
// 注意:
//   - NFKC 正規化を行うが、最小依存のため簡易。必要に応じて強化して下さい。
//   - NDC は 10 大類＋代表的細目の一部をサンプル同梱。必要に応じて拡張。

import (
	"bufio"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
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
	"fyne.io/fyne/v2/widget"
	"golang.org/x/text/unicode/norm"

	// 既存の埋め込みエンコーダ（ユーザー環境に合わせて import パスを調整）
	emb "yashubustudio/categorizer/emb"
)

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
	TopK       int
	UseNDC     bool
	WeightNDC  float32
	Thresh     Threshold
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
		UseNDC:        true,
		WeightNDC:     0.85,
		Thresh:        Threshold{Top1: 0.45, Margin12: 0.03, Mean: 0.50},
		ClusterCfg:    ClusterCfg{Enabled: false, Threshold: 0.80},
		OrtDLL:        "./onnixruntime-win/lib/onnxruntime.dll",
		ModelPath:     "./models/bge-m3/model.onnx",
		TokenizerPath: "./models/bge-m3/tokenizer.json",
		MaxSeqLen:     512,
		CacheDir:      "./cache",
	}
}

// ------------------------------
// （参考）テーブルセルの自動改行ラッパ（未使用でも可）
// ------------------------------

// newWrappedLabel はテーブル内で長文を自動改行するラベルを生成する。
func newWrappedLabel(text string, width float32) *widget.Label {
	lbl := widget.NewLabel(text)
	lbl.Wrapping = fyne.TextWrapWord
	// lbl.Truncate = false  // ← 削除
	lbl.Resize(fyne.NewSize(width, lbl.MinSize().Height))
	return lbl
}

// ------------------------------
// データ構造
// ------------------------------

type Candidate struct {
	Label  string
	Vec    []float32
	Source string // "cat" or "ndc"
}

type ScoredLabel struct {
	Label  string
	Score  float32
	Source string
}

type ResultRow struct {
	Text       string
	Top        []ScoredLabel // len <= TopK
	NeedReview bool
}

// ------------------------------
// ユーザー定義カテゴリ（初期値）
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
// NDC ラベル（最小セット）
// ------------------------------

type ndcItem struct{ Code, Label string }

// 10大類＋代表的細目サンプル
var ndcLabels = []ndcItem{
	{"000", "総記"},
	{"007", "情報科学"},
	{"100", "哲学"},
	{"200", "歴史"},
	{"300", "社会科学"},
	{"336", "経営"},
	{"657", "会計"},
	{"400", "自然科学"},
	{"491", "医学"},
	{"498", "衛生学.公衆衛生.予防医学"},
	{"500", "技術・工学"},
	{"600", "産業"},
	{"700", "芸術・美術"},
	{"800", "言語"},
	{"900", "文学"},
}

// ------------------------------
// 埋め込みキャッシュ
// ------------------------------

type embedCache struct {
	mu sync.RWMutex
	m  map[string][]float32
}

func newEmbedCache() *embedCache { return &embedCache{m: make(map[string][]float32)} }

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

// ------------------------------
// 文字列正規化
// ------------------------------

func normalize(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Join(strings.Fields(s), " ")
	// NFKC 正規化
	s = norm.NFKC.String(s)
	return s
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
	// 安定ソート用の極小バイアス。ハッシュの下位ビットから微量加算。
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

// ------------------------------
// サービス
// ------------------------------

type Service struct {
	cfg      Config
	emb      *emb.Encoder
	cache    *embedCache
	candsCat []Candidate
	candsNDC []Candidate
}

func NewService(cfg Config) (*Service, error) {
	if cfg.TopK < 1 {
		cfg.TopK = 3
	}
	// emb 初期化
	enc := &emb.Encoder{}
	if err := enc.Init(emb.Config{
		OrtDLL:        cfg.OrtDLL,
		ModelPath:     cfg.ModelPath,
		TokenizerPath: cfg.TokenizerPath,
		MaxSeqLen:     cfg.MaxSeqLen,
	}); err != nil {
		return nil, err
	}

	s := &Service{cfg: cfg, emb: enc, cache: newEmbedCache()}

	// 候補をベクトル化
	if err := s.buildCandidates(defaultUserCategories, ndcLabels); err != nil {
		enc.Close()
		return nil, err
	}
	return s, nil
}

func (s *Service) Close() {
	if s.emb != nil {
		s.emb.Close()
	}
}

func (s *Service) buildCandidates(userCats []string, ndc []ndcItem) error {
	ctx := context.Background()
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error

	// ユーザーカテゴリ
	wg.Add(1)
	go func() {
		defer wg.Done()
		local := make([]Candidate, 0, len(userCats))
		for _, label := range userCats {
			lab := normalize(label)
			vec, err := s.EmbedCached(ctx, lab)
			if err != nil {
				firstErr = err
				return
			}
			local = append(local, Candidate{Label: lab, Vec: vec, Source: "cat"})
		}
		mu.Lock()
		s.candsCat = local
		mu.Unlock()
	}()

	// NDC
	wg.Add(1)
	go func() {
		defer wg.Done()
		local := make([]Candidate, 0, len(ndc))
		for _, it := range ndc {
			lab := normalize(it.Code + " " + it.Label)
			vec, err := s.EmbedCached(ctx, lab)
			if err != nil {
				firstErr = err
				return
			}
			local = append(local, Candidate{Label: lab, Vec: vec, Source: "ndc"})
		}
		mu.Lock()
		s.candsNDC = local
		mu.Unlock()
	}()

	wg.Wait()
	return firstErr
}

func (s *Service) EmbedCached(ctx context.Context, text string) ([]float32, error) {
	key := text // 簡易キー。実運用は sha1(NFKC(text)+modelID) 推奨
	if v, ok := s.cache.get(key); ok {
		return v, nil
	}
	v, err := s.emb.Encode(text)
	if err != nil {
		return nil, err
	}
	s.cache.put(key, v)
	return v, nil
}

func (s *Service) RankOne(ctx context.Context, text string) (ResultRow, error) {
	row := ResultRow{Text: text}
	t := normalize(text)
	if t == "" {
		row.NeedReview = true
		return row, nil
	}

	q, err := s.EmbedCached(ctx, t)
	if err != nil {
		return row, err
	}

	// 候補集合
	cands := make([]Candidate, 0, len(s.candsCat)+len(s.candsNDC))
	cands = append(cands, s.candsCat...)
	if s.cfg.UseNDC {
		cands = append(cands, s.candsNDC...)
	}
	if len(cands) == 0 {
		return row, errors.New("no candidates")
	}

	// スコア計算
	scored := make([]ScoredLabel, 0, len(cands))
	for _, c := range cands {
		sc := cosine32(q, c.Vec)
		if sc < 0 {
			sc = 0
		}
		w := float32(1.0)
		if c.Source == "ndc" {
			w = s.cfg.WeightNDC
		}
		sc = clamp01(sc*w + tinyBias(c.Label))
		scored = append(scored, ScoredLabel{Label: c.Label, Score: sc, Source: c.Source})
	}

	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	k := s.cfg.TopK
	if k < 1 {
		k = 1
	}
	if k > 5 {
		k = 5
	}
	if k > len(scored) {
		k = len(scored)
	}
	row.Top = scored[:k]

	// 信頼度判定
	top1 := row.Top[0].Score
	top2 := float32(0)
	if len(row.Top) >= 2 {
		top2 = row.Top[1].Score
	}
	mean := meanScore(row.Top)
	row.NeedReview = (top1 < s.cfg.Thresh.Top1) || (top1-top2 < s.cfg.Thresh.Margin12) || (mean < s.cfg.Thresh.Mean)
	return row, nil
}

func meanScore(v []ScoredLabel) float32 {
	if len(v) == 0 {
		return 0
	}
	var s float32
	for _, x := range v {
		s += x.Score
	}
	return s / float32(len(v))
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
// GUI (Fyne)
// ------------------------------

type uiState struct {
	cfg     Config
	service *Service

	w      fyne.Window
	input  *widget.Entry // MultiLine
	resTbl *widget.Table
	rows   []ResultRow

	status *widget.Label
}

// 追加: 測定用ラベルで必要高さを見積もる
func wrappedHeightFor(text string, colWidth float32) float32 {
	lbl := widget.NewLabel(text)
	lbl.Wrapping = fyne.TextWrapWord
	// lbl.Truncate = false  // ← 削除
	lbl.Resize(fyne.NewSize(colWidth, 0))
	return lbl.MinSize().Height
}

func buildUI(a fyne.App, svc *Service, cfg Config) *uiState {
	u := &uiState{cfg: cfg, service: svc}
	u.w = a.NewWindow("Vector Categorizer - カテゴリ優先＋NDC")

	u.input = widget.NewMultiLineEntry()
	u.input.SetPlaceHolder("ここに文章を入力（複数行=複数レコード）")

	classifyBtn := widget.NewButton("分類", func() { u.onClassify() })
	exportBtn := widget.NewButton("CSVエクスポート", func() { u.onExport() })
	settingsBtn := widget.NewButton("設定", func() { u.openSettings() })

	u.status = widget.NewLabel("ready")

	u.resTbl = widget.NewTable(
		func() (int, int) { return len(u.rows), 6 },
		func() fyne.CanvasObject {
			lbl := widget.NewLabel("")
			lbl.Wrapping = fyne.TextWrapWord
			// lbl.Truncate = false  // ← 削除
			return lbl
		},
		func(id widget.TableCellID, obj fyne.CanvasObject) {
			if id.Row >= len(u.rows) {
				return
			}
			r := u.rows[id.Row]
			lbl := obj.(*widget.Label)
			// 各列のテキストを設定
			switch id.Col {
			case 0:
				// ★フルテキストを渡す（trim100しない）
				lbl.SetText(r.Text)
				// 列幅に合わせて必要高さを測定して行高を更新
				colW := float32(360) // 下の SetColumnWidth と同じ値
				needH := wrappedHeightFor(r.Text, colW)
				// 既定の最低行高（1行分）と比較して大きい方に
				minH := float32(28) // お好みで
				if needH < minH {
					needH = minH
				}
				u.resTbl.SetRowHeight(id.Row, needH)

			case 1:
				lbl.SetText(labelScore(r.Top, 0))
			case 2:
				lbl.SetText(labelScore(r.Top, 1))
			case 3:
				lbl.SetText(labelScore(r.Top, 2))
			case 4:
				lbl.SetText(ifThen(r.NeedReview, "要確認", ""))
			case 5:
				lbl.SetText(sources3(r.Top))
			}
		},
	)
	u.resTbl.SetColumnWidth(0, 360)
	u.resTbl.SetColumnWidth(1, 160)
	u.resTbl.SetColumnWidth(2, 160)
	u.resTbl.SetColumnWidth(3, 160)
	u.resTbl.SetColumnWidth(4, 80)
	u.resTbl.SetColumnWidth(5, 100)

	controls := container.NewHBox(classifyBtn, exportBtn, settingsBtn)
	content := container.NewBorder(container.NewVBox(u.input, controls), container.NewHBox(u.status), nil, nil, u.resTbl)
	u.w.SetContent(content)
	u.w.Resize(fyne.NewSize(1100, 720))
	return u
}

func (u *uiState) onClassify() {
	lines := splitNonEmptyLines(u.input.Text)
	if len(lines) == 0 {
		dialog.ShowInformation("情報", "入力テキストが空です", u.w)
		return
	}
	u.status.SetText("encoding…")
	start := time.Now()
	rows := make([]ResultRow, len(lines))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	for i := range lines {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			row, err := u.service.RankOne(context.Background(), lines[idx])
			if err != nil {
				firstErr = err
				return
			}
			mu.Lock()
			rows[idx] = row
			mu.Unlock()
		}(i)
	}
	wg.Wait()
	if firstErr != nil {
		dialog.ShowError(firstErr, u.w)
		u.status.SetText("error")
		return
	}
	u.rows = rows
	u.resTbl.Refresh()
	u.status.SetText(fmt.Sprintf("done: %d items in %v", len(rows), time.Since(start)))
}

func (u *uiState) onExport() {
	if len(u.rows) == 0 {
		dialog.ShowInformation("情報", "出力データがありません", u.w)
		return
	}
	fd := dialog.NewFileSave(func(uc fyne.URIWriteCloser, err error) {
		if err != nil || uc == nil {
			return
		}
		w := csv.NewWriter(uc)
		defer w.Flush()
		_ = w.Write([]string{"text", "label1", "score1", "label2", "score2", "label3", "score3", "need_review"})
		for _, r := range u.rows {
			l1, s1 := ls(r.Top, 0)
			l2, s2 := ls(r.Top, 1)
			l3, s3 := ls(r.Top, 2)
			_ = w.Write([]string{
				trim100(r.Text), l1, fmt.Sprintf("%.3f", s1), l2, fmt.Sprintf("%.3f", s2), l3, fmt.Sprintf("%.3f", s3),
				ifThen(r.NeedReview, "yes", "no"),
			})
		}
	}, u.w)
	fd.SetFileName("result.csv")
	fd.Show()
}

func (u *uiState) openSettings() {
	// TopK (3..5)
	topkSel := widget.NewSelect([]string{"3", "4", "5"}, nil)
	topkSel.SetSelected(strconv.Itoa(clampTopK(u.cfg.TopK)))
	// NDC ON/OFF
	ndcCheck := widget.NewCheck("NDC を候補に含める", nil)
	ndcCheck.SetChecked(u.cfg.UseNDC)
	// WeightNDC
	weightEntry := widget.NewEntry()
	weightEntry.SetText(fmt.Sprintf("%.2f", u.cfg.WeightNDC))
	// しきい値
	top1Entry := widget.NewEntry()
	top1Entry.SetText(fmt.Sprintf("%.2f", u.cfg.Thresh.Top1))
	m12Entry := widget.NewEntry()
	m12Entry.SetText(fmt.Sprintf("%.2f", u.cfg.Thresh.Margin12))
	meanEntry := widget.NewEntry()
	meanEntry.SetText(fmt.Sprintf("%.2f", u.cfg.Thresh.Mean))

	form := &widget.Form{Items: []*widget.FormItem{
		{Text: "Top-k", Widget: topkSel},
		{Text: "NDC使用", Widget: ndcCheck},
		{Text: "NDC重み(0.70-1.00)", Widget: weightEntry},
		{Text: "閾値 Top1", Widget: top1Entry},
		{Text: "閾値 Top1-Top2", Widget: m12Entry},
		{Text: "閾値 平均", Widget: meanEntry},
	}}

	dlg := dialog.NewCustomConfirm("設定", "OK", "キャンセル", form, func(ok bool) {
		if !ok {
			return
		}
		if v, err := strconv.Atoi(topkSel.Selected); err == nil {
			u.cfg.TopK = clampTopK(v)
		}
		u.cfg.UseNDC = ndcCheck.Checked
		if v, err := strconv.ParseFloat(weightEntry.Text, 32); err == nil {
			w := float32(v)
			if w < 0.7 {
				w = 0.7
			} else if w > 1.0 {
				w = 1.0
			}
			u.cfg.WeightNDC = w
		}
		if v, err := strconv.ParseFloat(top1Entry.Text, 32); err == nil {
			u.cfg.Thresh.Top1 = float32(v)
		}
		if v, err := strconv.ParseFloat(m12Entry.Text, 32); err == nil {
			u.cfg.Thresh.Margin12 = float32(v)
		}
		if v, err := strconv.ParseFloat(meanEntry.Text, 32); err == nil {
			u.cfg.Thresh.Mean = float32(v)
		}
		u.status.SetText("設定を更新しました")
	}, u.w)
	dlg.Resize(fyne.NewSize(420, 280))
	dlg.Show()
}

func clampTopK(k int) int {
	if k < 3 {
		return 3
	}
	if k > 5 {
		return 5
	}
	return k
}

// ------------------------------
// ヘルパ
// ------------------------------

func sources3(top []ScoredLabel) string {
	s := make([]string, 0, len(top))
	for i := 0; i < len(top) && i < 3; i++ {
		s = append(s, top[i].Source)
	}
	return strings.Join(s, ",")
}

func labelScore(top []ScoredLabel, idx int) string {
	if idx >= len(top) {
		return ""
	}
	return fmt.Sprintf("%s (%.3f)", top[idx].Label, top[idx].Score)
}

func ls(top []ScoredLabel, idx int) (string, float32) {
	if idx >= len(top) {
		return "", 0
	}
	return top[idx].Label, top[idx].Score
}

func ifThen[T any](cond bool, a, b T) T {
	if cond {
		return a
	}
	return b
}

func trim100(s string) string {
	s = strings.TrimSpace(s)
	if len([]rune(s)) <= 100 {
		return s
	}
	r := []rune(s)
	return string(r[:100]) + "…"
}

func splitNonEmptyLines(s string) []string {
	res := []string{}
	scanner := bufio.NewScanner(strings.NewReader(s))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			res = append(res, line)
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
	u := buildUI(a, svc, cfg)
	u.w.ShowAndRun()
}

func ensureDirs(p string) {
	if p == "" {
		return
	}
	_ = os.MkdirAll(filepath.Clean(p), 0o755)
}
