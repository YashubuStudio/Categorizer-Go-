package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/widget"

	"yashubustudio/categorizer/categorizer"
)

func main() {
	fyneApp := app.NewWithID("yashubustudio.categorizer")
	win := fyneApp.NewWindow("Categorizer (ベクトル検索支援)")
	win.Resize(fyne.NewSize(1024, 768))

	cfg, err := categorizer.LoadConfig("")
	if err != nil {
		showFatalError(win, fmt.Errorf("設定の読み込みに失敗しました: %w", err))
		return
	}

	loggerBinding := binding.NewString()
	logCapture := newLogCapture(loggerBinding, 300)
	logger := log.New(io.MultiWriter(os.Stdout, logCapture), "", log.LstdFlags)

	embedder, err := categorizer.NewOrtEmbedder(cfg.Embedder)
	if err != nil {
		logCapture.Write([]byte(fmt.Sprintf("[ERROR] %v\n", err)))
		showFatalError(win, fmt.Errorf("埋め込みエンジンの初期化に失敗しました: %w", err))
		return
	}

	ctx := context.Background()
	service, err := categorizer.NewService(ctx, embedder, cfg, logger)
	if err != nil {
		showFatalError(win, fmt.Errorf("サービス初期化に失敗しました: %w", err))
		return
	}
	defer service.Close()

	var resultRows []categorizer.ResultRow
	var resultMu sync.Mutex

	cfgMu := sync.Mutex{}

	saveConfig := func() {
		cfgMu.Lock()
		defer cfgMu.Unlock()
		if err := categorizer.SaveConfig("", cfg); err != nil {
			logger.Printf("設定の保存に失敗しました: %v", err)
		}
	}
	defer saveConfig()

	// Seed controls
	seedInput := widget.NewMultiLineEntry()
	seedInput.SetPlaceHolder("カテゴリシード（改行またはカンマ区切り）")
	seedInput.Wrapping = fyne.TextWrapWord

	seedStatus := widget.NewLabel("シード未設定")

	applySeeds := func(seeds []string) {
		seedStatus.SetText("シード更新中...")
		if fyneApp.Driver() == nil {
			if err := service.LoadSeeds(ctx, seeds); err != nil {
				showError(win, fmt.Errorf("シードの読み込みに失敗しました: %w", err))
				return
			}
			seedStatus.SetText(fmt.Sprintf("シード数: %d", service.SeedCount()))
			return
		}
		go func(list []string) {
			if err := service.LoadSeeds(ctx, list); err != nil {
				showError(win, fmt.Errorf("シードの読み込みに失敗しました: %w", err))
				return
			}
			fyneApp.Driver().CallOnMainThread(func() {
				seedStatus.SetText(fmt.Sprintf("シード数: %d", service.SeedCount()))
			})
		}(seeds)
	}

	loadSeedsFromInput := func() {
		seeds := categorizer.ParseSeeds(seedInput.Text)
		cfgMu.Lock()
		cfg.SeedsPath = ""
		cfgMu.Unlock()
		saveConfig()
		applySeeds(seeds)
	}

	loadSeedsBtn := widget.NewButton("シード反映", loadSeedsFromInput)
	loadSeedsFileBtn := widget.NewButton("シードファイル読込", func() {
		fd := dialog.NewFileOpen(func(rc fyne.URIReadCloser, err error) {
			if err != nil {
				showError(win, err)
				return
			}
			if rc == nil {
				return
			}
			defer rc.Close()
			path := rc.URI().Path()
			seeds, err := categorizer.ParseSeedFile(path)
			if err != nil {
				showError(win, err)
				return
			}
			seedInput.SetText(strings.Join(seeds, "\n"))
			cfgMu.Lock()
			cfg.SeedsPath = path
			cfgMu.Unlock()
			saveConfig()
			applySeeds(seeds)
		}, win)
		fd.SetFilter(storageFilter([]string{".txt", ".csv", ".tsv"}))
		fd.Show()
	})

	// Load seeds from config if path specified
	if cfg.SeedsPath != "" {
		if seeds, err := categorizer.ParseSeedFile(cfg.SeedsPath); err == nil {
			seedInput.SetText(strings.Join(seeds, "\n"))
			applySeeds(seeds)
		} else {
			logger.Printf("シードファイルの読み込みに失敗しました: %v", err)
		}
	}

	// Text input controls
	textInput := widget.NewMultiLineEntry()
	textInput.SetPlaceHolder("分類したい文章を1行ずつ入力してください")
	textInput.Wrapping = fyne.TextWrapWord

	statusLabel := widget.NewLabel("準備完了")

	var tableData [][]string
	resultTable := widget.NewTable(
		func() (int, int) {
			if len(tableData) == 0 {
				return 0, 0
			}
			return len(tableData), len(tableData[0])
		},
		func() fyne.CanvasObject {
			return widget.NewLabel("")
		},
		func(id widget.TableCellID, obj fyne.CanvasObject) {
			if len(tableData) == 0 || id.Row >= len(tableData) || id.Col >= len(tableData[id.Row]) {
				return
			}
			label := obj.(*widget.Label)
			label.SetText(tableData[id.Row][id.Col])
			if id.Row == 0 {
				label.TextStyle = fyne.TextStyle{Bold: true}
			} else {
				label.TextStyle = fyne.TextStyle{}
			}
		},
	)
	resultTable.OnSelected = func(id widget.TableCellID) {
		if id.Row <= 0 {
			return
		}
		resultMu.Lock()
		if id.Row-1 < len(resultRows) {
			row := resultRows[id.Row-1]
			dialog.ShowInformation("詳細", fmt.Sprintf("本文:\n%s\n\n提案:\n%s", row.Text, formatSuggestions(row)), win)
		}
		resultMu.Unlock()
	}

	updateTable := func(rows []categorizer.ResultRow) {
		resultMu.Lock()
		resultRows = rows
		resultMu.Unlock()
		localCfg := service.Config()
		includeNDC := localCfg.Mode == categorizer.ModeSplit && localCfg.UseNDC
		tableData = buildTableData(rows, localCfg.TopK, includeNDC)
		fyne.CurrentApp().Driver().CallOnMainThread(func() {
			if len(tableData) > 0 {
				for col := range tableData[0] {
					width := float32(150)
					if col == 0 {
						width = 220
					}
					resultTable.SetColumnWidth(col, width)
				}
			}
			resultTable.Refresh()
		})
	}

	classifyBtn := widget.NewButton("分類実行", func() {
		lines := parseInputTexts(textInput.Text)
		if len(lines) == 0 {
			showError(win, fmt.Errorf("入力文章がありません"))
			return
		}
		classifyBtn.Disable()
		statusLabel.SetText("推論中...")
		go func(texts []string) {
			start := time.Now()
			rows, err := service.ClassifyAll(ctx, texts)
			elapsed := time.Since(start)
			if err != nil {
				fyne.CurrentApp().Driver().CallOnMainThread(func() {
					classifyBtn.Enable()
					statusLabel.SetText("エラーが発生しました")
					showError(win, err)
				})
				return
			}
			updateTable(rows)
			fyne.CurrentApp().Driver().CallOnMainThread(func() {
				classifyBtn.Enable()
				statusLabel.SetText(fmt.Sprintf("%d件 %.2fs", len(rows), elapsed.Seconds()))
			})
		}(lines)
	})

	loadTextFileBtn := widget.NewButton("テキスト読込", func() {
		fd := dialog.NewFileOpen(func(rc fyne.URIReadCloser, err error) {
			if err != nil {
				showError(win, err)
				return
			}
			if rc == nil {
				return
			}
			defer rc.Close()
			path := rc.URI().Path()
			texts, err := categorizer.ParseTextFile(path)
			if err != nil {
				showError(win, err)
				return
			}
			textInput.SetText(strings.Join(texts, "\n"))
		}, win)
		fd.SetFilter(storageFilter([]string{".txt", ".csv", ".tsv"}))
		fd.Show()
	})

	exportBtn := widget.NewButton("結果をCSV出力", func() {
		if len(tableData) <= 1 {
			showError(win, fmt.Errorf("出力する結果がありません"))
			return
		}
		fd := dialog.NewFileSave(func(uc fyne.URIWriteCloser, err error) {
			if err != nil {
				showError(win, err)
				return
			}
			if uc == nil {
				return
			}
			defer uc.Close()
			writer := csv.NewWriter(uc)
			for _, row := range tableData {
				if err := writer.Write(row); err != nil {
					showError(win, err)
					return
				}
			}
			writer.Flush()
			if err := writer.Error(); err != nil {
				showError(win, err)
				return
			}
		}, win)
		fd.SetFileName("results.csv")
		fd.SetFilter(storageFilter([]string{".csv"}))
		fd.Show()
	})

	// Settings controls
	modeSelect := widget.NewSelect([]string{string(categorizer.ModeSeeded), string(categorizer.ModeMixed), string(categorizer.ModeSplit)}, func(val string) {
		cfgMu.Lock()
		cfg.Mode = categorizer.Mode(val)
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	})
	modeSelect.SetSelected(string(cfg.Mode))

	topKSlider := widget.NewSlider(1, 5)
	topKSlider.Step = 1
	topKSlider.SetValue(float64(cfg.TopK))
	topKLabel := widget.NewLabel(fmt.Sprintf("Top-K: %d", cfg.TopK))
	topKSlider.OnChanged = func(v float64) {
		k := int(v)
		topKLabel.SetText(fmt.Sprintf("Top-K: %d", k))
		cfgMu.Lock()
		cfg.TopK = k
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	}

	seedBiasSlider := widget.NewSlider(0, 0.2)
	seedBiasSlider.Step = 0.01
	seedBiasSlider.SetValue(float64(cfg.SeedBias))
	seedBiasLabel := widget.NewLabel(fmt.Sprintf("シードバイアス: %.2f", cfg.SeedBias))
	seedBiasSlider.OnChanged = func(v float64) {
		seedBiasLabel.SetText(fmt.Sprintf("シードバイアス: %.2f", v))
		cfgMu.Lock()
		cfg.SeedBias = float32(v)
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	}

	minScoreSlider := widget.NewSlider(0, 1)
	minScoreSlider.Step = 0.01
	minScoreSlider.SetValue(float64(cfg.MinScore))
	minScoreLabel := widget.NewLabel(fmt.Sprintf("最小スコア: %.2f", cfg.MinScore))
	minScoreSlider.OnChanged = func(v float64) {
		minScoreLabel.SetText(fmt.Sprintf("最小スコア: %.2f", v))
		cfgMu.Lock()
		cfg.MinScore = float32(v)
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	}

	clusterCheck := widget.NewCheck("類似カテゴリを束ねる", nil)
	clusterCheck.SetChecked(cfg.Cluster.Enabled)

	clusterSlider := widget.NewSlider(0.5, 0.95)
	clusterSlider.Step = 0.01
	clusterSlider.SetValue(float64(cfg.Cluster.Threshold))
	clusterLabel := widget.NewLabel(fmt.Sprintf("クラスタ閾値: %.2f", cfg.Cluster.Threshold))
	clusterSlider.OnChanged = func(v float64) {
		clusterLabel.SetText(fmt.Sprintf("クラスタ閾値: %.2f", v))
		cfgMu.Lock()
		cfg.Cluster.Threshold = float32(v)
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	}
	clusterSlider.Disable()
	if cfg.Cluster.Enabled {
		clusterSlider.Enable()
	}
	clusterCheck.OnChanged = func(checked bool) {
		if checked {
			clusterSlider.Enable()
		} else {
			clusterSlider.Disable()
		}
		cfgMu.Lock()
		cfg.Cluster.Enabled = checked
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	}

	useNDCCheck := widget.NewCheck("NDC提案を使用", func(checked bool) {
		cfgMu.Lock()
		cfg.UseNDC = checked
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
		go func() {
			if checked {
				if err := service.LoadNDCDictionary(ctx, categorizer.DefaultNDCEntries()); err != nil {
					showError(win, err)
				}
			} else {
				if err := service.LoadNDCDictionary(ctx, nil); err != nil {
					showError(win, err)
				}
			}
		}()
	})
	useNDCCheck.SetChecked(cfg.UseNDC)

	logLabel := widget.NewLabelWithData(loggerBinding)
	logLabel.Wrapping = fyne.TextWrapWord
	logContainer := container.NewVScroll(logLabel)
	logContainer.SetMinSize(fyne.NewSize(200, 120))

	controls := container.NewVBox(
		container.NewHBox(classifyBtn, loadTextFileBtn, exportBtn, statusLabel),
		container.NewVBox(widget.NewLabel("テキスト入力"), textInput),
		widget.NewSeparator(),
		container.NewVBox(
			widget.NewLabel("シードカテゴリ"),
			seedInput,
			container.NewHBox(loadSeedsBtn, loadSeedsFileBtn, seedStatus),
		),
		widget.NewSeparator(),
		container.NewVBox(
			widget.NewLabel("設定"),
			modeSelect,
			container.NewHBox(topKLabel, topKSlider),
			container.NewHBox(seedBiasLabel, seedBiasSlider),
			container.NewHBox(minScoreLabel, minScoreSlider),
			container.NewHBox(clusterCheck, clusterLabel, clusterSlider),
			useNDCCheck,
		),
		widget.NewSeparator(),
		widget.NewLabel("ログ"),
		logContainer,
	)

	root := container.NewHSplit(controls, container.NewVSplit(resultTable, widget.NewLabel("セルを選択すると詳細が表示されます")))
	root.Offset = 0.45
	win.SetContent(root)

	win.ShowAndRun()
}

func showFatalError(win fyne.Window, err error) {
	content := widget.NewLabel(err.Error())
	win.SetContent(content)
	dialog.ShowError(err, win)
	win.ShowAndRun()
}

func showError(win fyne.Window, err error) {
	if err != nil {
		dialog.ShowError(err, win)
	}
}

func storageFilter(exts []string) fyne.FileFilter {
	return storage.NewExtensionFileFilter(exts)
}

func parseInputTexts(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

func buildTableData(rows []categorizer.ResultRow, topK int, includeNDC bool) [][]string {
	header := []string{"text"}
	for i := 1; i <= topK; i++ {
		header = append(header, fmt.Sprintf("suggestion%d", i), fmt.Sprintf("score%d", i))
	}
	if includeNDC {
		for i := 1; i <= topK; i++ {
			header = append(header, fmt.Sprintf("ndc%d", i), fmt.Sprintf("ndcScore%d", i))
		}
	}
	data := make([][]string, 1, len(rows)+1)
	data[0] = header
	for _, row := range rows {
		rowData := make([]string, len(header))
		rowData[0] = truncateText(row.Text, 100)
		idx := 1
		for i := 0; i < topK; i++ {
			if i < len(row.Suggestions) {
				rowData[idx] = row.Suggestions[i].Label
				rowData[idx+1] = fmt.Sprintf("%.3f", row.Suggestions[i].Score)
			}
			idx += 2
		}
		if includeNDC {
			for i := 0; i < topK; i++ {
				if i < len(row.NDCSuggestions) {
					rowData[idx] = row.NDCSuggestions[i].Label
					rowData[idx+1] = fmt.Sprintf("%.3f", row.NDCSuggestions[i].Score)
				}
				idx += 2
			}
		}
		data = append(data, rowData)
	}
	return data
}

func truncateText(text string, max int) string {
	if len([]rune(text)) <= max {
		return text
	}
	runes := []rune(text)
	return string(runes[:max]) + "…"
}

func formatSuggestions(row categorizer.ResultRow) string {
	var b strings.Builder
	for i, s := range row.Suggestions {
		fmt.Fprintf(&b, "[%d] %s (%.3f)\n", i+1, s.Label, s.Score)
	}
	if len(row.NDCSuggestions) > 0 {
		b.WriteString("\nNDC:\n")
		for i, s := range row.NDCSuggestions {
			fmt.Fprintf(&b, "[%d] %s (%.3f)\n", i+1, s.Label, s.Score)
		}
	}
	return b.String()
}

type logCapture struct {
	mu      sync.Mutex
	lines   []string
	limit   int
	binding binding.String
}

func newLogCapture(b binding.String, limit int) *logCapture {
	return &logCapture{binding: b, limit: limit}
}

func (l *logCapture) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	text := string(p)
	text = strings.ReplaceAll(text, "\r\n", "\n")
	parts := strings.Split(text, "\n")
	for _, part := range parts {
		if part == "" {
			continue
		}
		l.lines = append(l.lines, part)
	}
	if len(l.lines) > l.limit {
		l.lines = l.lines[len(l.lines)-l.limit:]
	}
	joined := strings.Join(l.lines, "\n")
	_ = l.binding.Set(joined)
	return len(p), nil
}
