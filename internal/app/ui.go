package app

import (
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
)

const logDebounceInterval = 150 * time.Millisecond

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
	statusBind    binding.String
	logBind       binding.String
	progressBind  binding.Float
	logLines      []string
	logMu         sync.Mutex
	logUpdateCh   chan struct{}

	classifyBtn *widget.Button
	exportBtn   *widget.Button
	loadBtn     *widget.Button
	catBtn      *widget.Button
}

func buildUI(a fyne.App, svc *Service) *uiState {
	u := &uiState{service: svc}
	u.cfg = svc.Config()
	u.w = a.NewWindow("Vector Categorizer - Seeded & NDC")

	u.statusBind = binding.NewString()
	_ = u.statusBind.Set("準備完了")
	u.progressBind = binding.NewFloat()
	u.logBind = binding.NewString()
	u.startLogUpdater()

	u.input = widget.NewMultiLineEntry()
	u.input.SetPlaceHolder("ここに文章を入力（1行=1件）")

	u.log = widget.NewEntryWithData(u.logBind)
	u.log.MultiLine = true
	u.log.Wrapping = fyne.TextWrapWord
	u.log.SetPlaceHolder("処理ログ")
	u.log.Disable()

	u.status = widget.NewLabelWithData(u.statusBind)
	u.progress = widget.NewProgressBarWithData(u.progressBind)
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
	fyne.Do(func() {
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
	})
}

func (u *uiState) appendLog(msg string) {
	now := time.Now().Format("15:04:05")
	line := fmt.Sprintf("[%s] %s", now, msg)

	u.logMu.Lock()
	u.logLines = append(u.logLines, line)
	if len(u.logLines) > 200 {
		u.logLines = u.logLines[len(u.logLines)-200:]
	}
	u.logMu.Unlock()

	if u.logUpdateCh == nil {
		u.flushLog()
		return
	}
	select {
	case u.logUpdateCh <- struct{}{}:
	default:
	}
}

func (u *uiState) startLogUpdater() {
	if u.logUpdateCh != nil {
		return
	}
	u.logUpdateCh = make(chan struct{}, 1)
	go u.logUpdateLoop()
}

func (u *uiState) logUpdateLoop() {
	timer := time.NewTimer(logDebounceInterval)
	if !timer.Stop() {
		<-timer.C
	}
	for {
		select {
		case <-u.logUpdateCh:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(logDebounceInterval)
		case <-timer.C:
			u.flushLog()
		}
	}
}

func (u *uiState) flushLog() {
	u.logMu.Lock()
	text := strings.Join(u.logLines, "\n")
	u.logMu.Unlock()
	_ = u.logBind.Set(text)
}

func (u *uiState) setStatus(text string) {
	_ = u.statusBind.Set(text)
}

func (u *uiState) configureProgress(min, max float64) {
	fyne.Do(func() {
		u.progress.Min = min
		u.progress.Max = max
	})
}

func (u *uiState) setProgressValue(value float64) {
	_ = u.progressBind.Set(value)
}

func (u *uiState) showProgress() {
	fyne.Do(func() {
		u.progress.Show()
	})
}

func (u *uiState) hideProgress() {
	fyne.Do(func() {
		u.progress.Hide()
	})
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
	u.configureProgress(0, float64(total))
	u.setProgressValue(0)
	u.showProgress()
	u.setStatus("処理中...")
	u.setBusy(true)
	u.appendLog(fmt.Sprintf("分類開始 (%d件)", total))
	start := time.Now()

	go func(entries []string) {
		rows, err := u.service.ClassifyAll(context.Background(), entries, func(done, total int) {
			u.setProgressValue(float64(done))
			u.setStatus(fmt.Sprintf("処理中 %d/%d", done, total))
		})

		u.setBusy(false)
		u.hideProgress()
		if err != nil {
			fyne.Do(func() {
				dialog.ShowError(err, u.w)
			})
			u.setStatus("エラー")
			u.appendLog(fmt.Sprintf("エラー: %v", err))
			return
		}
		fyne.Do(func() {
			u.rows = rows
			u.resTbl.Refresh()
		})
		elapsed := time.Since(start).Seconds()
		u.setProgressValue(float64(len(rows)))
		u.setStatus(fmt.Sprintf("完了 %d件 (%.1fs)", len(rows), elapsed))
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
		uri := rc.URI()
		ext := strings.ToLower(filepath.Ext(uri.Path()))
		if ext == ".csv" || ext == ".tsv" {
			delim := ','
			if ext == ".tsv" {
				delim = '\t'
			}
			records, err := readCSVRecords(data, delim)
			if err != nil {
				dialog.ShowError(err, u.w)
				return
			}
			u.handleCSVRecords(uri, records)
			return
		}
		lines := splitNonEmptyLines(string(data))
		u.applyLoadedLines(uri, lines)
	}, u.w)
	fd.SetFilter(storage.NewExtensionFileFilter([]string{".txt", ".csv", ".tsv"}))
	fd.Show()
}

func (u *uiState) applyLoadedLines(uri fyne.URI, lines []string) {
	u.input.SetText(strings.Join(lines, "\n"))
	u.appendLog(fmt.Sprintf("ファイル読込: %s (%d件)", filepath.Base(uri.Path()), len(lines)))
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

func (u *uiState) handleCSVRecords(uri fyne.URI, records [][]string) {
	maxCols := 0
	for _, row := range records {
		if len(row) > maxCols {
			maxCols = len(row)
		}
	}
	if maxCols == 0 {
		dialog.ShowError(errors.New("CSVが空です"), u.w)
		return
	}
	defaultCol := detectTextColumn(records[0])
	hasHeader := false
	if defaultCol >= 0 {
		hasHeader = true
	} else {
		defaultCol = 0
	}
	if defaultCol >= maxCols {
		defaultCol = 0
	}
	if maxCols == 1 {
		lines := extractCSVColumn(records, defaultCol, hasHeader)
		u.applyLoadedLines(uri, lines)
		return
	}
	choices := buildCSVColumnChoices(records, hasHeader)
	if len(choices) == 0 {
		dialog.ShowError(errors.New("有効な列が見つかりません"), u.w)
		return
	}
	defaultChoice := 0
	for i, c := range choices {
		if c.Index == defaultCol {
			defaultChoice = i
			break
		}
	}
	options := make([]string, len(choices))
	for i, c := range choices {
		options[i] = c.Label
	}
	selectedCol := choices[defaultChoice].Index
	selectWidget := widget.NewSelect(options, func(value string) {
		for i, opt := range options {
			if opt == value {
				selectedCol = choices[i].Index
				return
			}
		}
	})
	selectWidget.SetSelected(options[defaultChoice])
	info := widget.NewLabel("読み込む列を選択してください")
	content := container.NewVBox(info, selectWidget)
	dialog.NewCustomConfirm("列の選択", "読み込む", "キャンセル", content, func(ok bool) {
		if !ok {
			return
		}
		lines := extractCSVColumn(records, selectedCol, hasHeader)
		u.applyLoadedLines(uri, lines)
	}, u.w).Show()
}

func wrappedHeightFor(text string, colWidth float32) float32 {
	lbl := widget.NewLabel(text)
	lbl.Wrapping = fyne.TextWrapWord
	lbl.Resize(fyne.NewSize(colWidth, 0))
	return lbl.MinSize().Height + 8
}
