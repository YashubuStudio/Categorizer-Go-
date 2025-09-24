package main

import (
	"context"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
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

type displayResult struct {
	Input  categorizer.InputRecord
	Result categorizer.ResultRow
}

func main() {
	batchInput := flag.String("batch-input", "", "CSV/TSV file containing texts to classify")
	batchCategories := flag.String("category-file", "", "CSV/TSV file containing category labels")
	batchOutputDir := flag.String("output-dir", "csv", "Directory where result_*.csv will be written")
	inputIndexColumn := flag.String("input-index-column", "", "Column name or #index for the presentation index column")
	inputTitleColumn := flag.String("input-title-column", "", "Column name or #index for the presentation title column")
	inputBodyColumn := flag.String("input-body-column", "", "Column name or #index for the presentation body/summary column")
	inputTextColumn := flag.String("input-text-column", "", "Column name or #index for the fallback text column")
	categoryColumn := flag.String("category-column", "", "Column name or #index for category labels")
	flag.Parse()

	if *batchInput != "" {
		inputOpts := categorizer.InputParseOptions{
			IndexColumn: strings.TrimSpace(*inputIndexColumn),
			TitleColumn: strings.TrimSpace(*inputTitleColumn),
			BodyColumn:  strings.TrimSpace(*inputBodyColumn),
			TextColumn:  strings.TrimSpace(*inputTextColumn),
		}
		if err := runBatchMode(
			*batchInput,
			*batchCategories,
			*batchOutputDir,
			inputOpts,
			strings.TrimSpace(*categoryColumn),
		); err != nil {
			log.Fatalf("batch mode: %v", err)
		}
		return
	}

	runGUIMode()
}

func runBatchMode(inputPath, categoriesPath, outputDir string, inputOpts categorizer.InputParseOptions, categoryColumn string) error {
	if categoriesPath == "" {
		return errors.New("--category-file is required when using --batch-input")
	}
	cfg, err := categorizer.LoadConfig("")
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	embedder, err := categorizer.NewOrtEmbedder(cfg.Embedder)
	if err != nil {
		return fmt.Errorf("init embedder: %w", err)
	}
	defer embedder.Close()

	ctx := context.Background()
	logger := log.New(os.Stdout, "", log.LstdFlags)
	service, err := categorizer.NewService(ctx, embedder, cfg, logger)
	if err != nil {
		return fmt.Errorf("init service: %w", err)
	}
	defer service.Close()

	categories, err := categorizer.ParseCategoryListWithOptions(categoriesPath, categorizer.CategoryParseOptions{Column: categoryColumn})
	if err != nil {
		return fmt.Errorf("read category list: %w", err)
	}
	if err := service.LoadSeeds(ctx, categories); err != nil {
		return fmt.Errorf("load categories: %w", err)
	}

	records, err := categorizer.ParseInputRecordsWithOptions(inputPath, inputOpts)
	if err != nil {
		return fmt.Errorf("read input records: %w", err)
	}
	if len(records) == 0 {
		return errors.New("input file does not contain any texts")
	}

	rows, err := classifyRecords(ctx, service, records)
	if err != nil {
		return fmt.Errorf("classify: %w", err)
	}

	outputPath, err := saveResultsCSV(outputDir, records, rows)
	if err != nil {
		return err
	}
	fmt.Printf("分類結果を %s に保存しました\n", outputPath)
	return nil
}

func runGUIMode() {
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

	var (
		displayResults    []displayResult
		displayMu         sync.Mutex
		pendingRecords    []categorizer.InputRecord
		usePendingRecords bool
		ignoreTextChange  bool
	)

	cfgMu := sync.Mutex{}
	saveConfig := func() {
		cfgMu.Lock()
		defer cfgMu.Unlock()
		if err := categorizer.SaveConfig("", cfg); err != nil {
			logger.Printf("設定の保存に失敗しました: %v", err)
		}
	}
	defer saveConfig()

	seedInput := widget.NewMultiLineEntry()
	seedInput.SetPlaceHolder("カテゴリシード（改行またはカンマ区切り）")
	seedInput.Wrapping = fyne.TextWrapWord

	seedStatus := widget.NewLabel("シード未設定")

	applySeeds := func(seeds []string) {
		seedStatus.SetText("シード更新中...")
		list := append([]string(nil), seeds...)
		go func(items []string) {
			if err := service.LoadSeeds(ctx, items); err != nil {
				fyne.Do(func() {
					seedStatus.SetText("シード更新失敗")
					showError(win, fmt.Errorf("シードの読み込みに失敗しました: %w", err))
				})
				return
			}
			fyne.Do(func() {
				seedStatus.SetText(fmt.Sprintf("シード数: %d", service.SeedCount()))
			})
		}(list)
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
			ext := strings.ToLower(filepath.Ext(path))
			applySeedsFromList := func(seeds []string) {
				seedInput.SetText(strings.Join(seeds, "\n"))
				cfgMu.Lock()
				cfg.SeedsPath = path
				cfgMu.Unlock()
				saveConfig()
				applySeeds(seeds)
			}
			if ext == ".csv" || ext == ".tsv" {
				metadata, err := categorizer.ReadCategoryFileMetadata(path)
				if err != nil {
					showError(win, err)
					return
				}
				showCategoryColumnSelector(win, path, metadata, func(column string, ok bool) {
					if !ok {
						return
					}
					seeds, err := categorizer.ParseCategoryListWithOptions(path, categorizer.CategoryParseOptions{Column: column})
					if err != nil {
						showError(win, err)
						return
					}
					applySeedsFromList(seeds)
				})
				return
			}
			seeds, err := categorizer.ParseSeedFile(path)
			if err != nil {
				showError(win, err)
				return
			}
			applySeedsFromList(seeds)
		}, win)
		fd.SetFilter(storageFilter([]string{".txt", ".csv", ".tsv"}))
		fd.Show()
	})

	if cfg.SeedsPath != "" {
		if seeds, err := categorizer.ParseCategoryListWithOptions(cfg.SeedsPath, categorizer.CategoryParseOptions{}); err == nil {
			seedInput.SetText(strings.Join(seeds, "\n"))
			applySeeds(seeds)
		} else if seeds, err := categorizer.ParseSeedFile(cfg.SeedsPath); err == nil {
			seedInput.SetText(strings.Join(seeds, "\n"))
			applySeeds(seeds)
		} else {
			logger.Printf("シードファイルの読み込みに失敗しました: %v", err)
		}
	}

	textInput := widget.NewMultiLineEntry()
	textInput.SetPlaceHolder("分類したい文章を1行ずつ入力してください")
	textInput.Wrapping = fyne.TextWrapWord
	textInput.OnChanged = func(string) {
		if ignoreTextChange {
			return
		}
		usePendingRecords = false
	}

	statusLabel := widget.NewLabel("準備完了")

	resultList := widget.NewList(
		func() int {
			displayMu.Lock()
			defer displayMu.Unlock()
			return len(displayResults)
		},
		func() fyne.CanvasObject {
			header := widget.NewLabel("")
			header.TextStyle = fyne.TextStyle{Bold: true}
			header.Wrapping = fyne.TextWrapWord
			summary := widget.NewLabel("")
			summary.Wrapping = fyne.TextWrapWord
			category := widget.NewLabel("")
			category.TextStyle = fyne.TextStyle{Bold: true}
			category.Wrapping = fyne.TextWrapWord
			score := widget.NewLabel("")
			score.Wrapping = fyne.TextWrapWord
			return container.NewVBox(header, summary, category, score)
		},
		func(i widget.ListItemID, obj fyne.CanvasObject) {
			displayMu.Lock()
			defer displayMu.Unlock()
			cont := obj.(*fyne.Container)
			header := cont.Objects[0].(*widget.Label)
			summary := cont.Objects[1].(*widget.Label)
			category := cont.Objects[2].(*widget.Label)
			score := cont.Objects[3].(*widget.Label)
			if i < 0 || i >= len(displayResults) {
				header.SetText("")
				summary.SetText("")
				summary.Hide()
				category.SetText("")
				score.SetText("")
				score.Hide()
				return
			}
			item := displayResults[i]
			header.SetText(formatDisplayHeader(item.Input))
			summaryText := formatDisplaySummary(item.Input)
			if summaryText == "" {
				summary.Hide()
				summary.SetText("")
			} else {
				summary.SetText(summaryText)
				summary.Show()
			}
			if best, ok := pickBestSuggestion(item.Result); ok {
				category.SetText(fmt.Sprintf("推定カテゴリ: %s", best.Label))
				score.SetText(fmt.Sprintf("スコア: %.3f (source: %s)", best.Score, best.Source))
				score.Show()
			} else {
				category.SetText("推定カテゴリ: (候補なし)")
				score.SetText("")
				score.Hide()
			}
		},
	)

	resultList.OnSelected = func(id widget.ListItemID) {
		displayMu.Lock()
		defer displayMu.Unlock()
		if id < 0 || id >= len(displayResults) {
			return
		}
		dialog.ShowInformation("詳細", buildDetailMessage(displayResults[id]), win)
	}

	updateResults := func(records []categorizer.InputRecord, rows []categorizer.ResultRow) {
		limit := len(records)
		if len(rows) < limit {
			limit = len(rows)
		}
		recCopy := make([]categorizer.InputRecord, limit)
		rowCopy := make([]categorizer.ResultRow, limit)
		copy(recCopy, records[:limit])
		copy(rowCopy, rows[:limit])
		fyne.Do(func() {
			displayMu.Lock()
			defer displayMu.Unlock()
			displayResults = make([]displayResult, limit)
			for i := 0; i < limit; i++ {
				displayResults[i] = displayResult{Input: recCopy[i], Result: rowCopy[i]}
			}
			resultList.Refresh()
		})
	}

	var classifyBtn *widget.Button
	runClassification := func(records []categorizer.InputRecord) {
		if len(records) == 0 {
			showError(win, fmt.Errorf("入力文章がありません"))
			return
		}
		classifyBtn.Disable()
		statusLabel.SetText("推論中...")
		localRecords := append([]categorizer.InputRecord(nil), records...)
		go func(samples []categorizer.InputRecord) {
			start := time.Now()
			rows, err := classifyRecords(ctx, service, samples)
			if err != nil {
				fyne.Do(func() {
					classifyBtn.Enable()
					statusLabel.SetText("エラーが発生しました")
					showError(win, err)
				})
				return
			}
			updateResults(samples, rows)
			fyne.Do(func() {
				classifyBtn.Enable()
				statusLabel.SetText(fmt.Sprintf("%d件 %.2fs", len(rows), time.Since(start).Seconds()))
			})
		}(localRecords)
	}

	classifyBtn = widget.NewButton("分類実行", func() {
		var records []categorizer.InputRecord
		if usePendingRecords && len(pendingRecords) > 0 {
			records = append([]categorizer.InputRecord(nil), pendingRecords...)
		} else {
			lines := parseInputTexts(textInput.Text)
			if len(lines) == 0 {
				showError(win, fmt.Errorf("入力文章がありません"))
				return
			}
			records = manualRecordsFromLines(lines)
		}
		runClassification(records)
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
			metadata, err := categorizer.ReadInputFileMetadata(path)
			if err != nil {
				showError(win, err)
				return
			}
			showInputColumnSelector(win, path, metadata, func(opts categorizer.InputParseOptions, ok bool) {
				if !ok {
					statusLabel.SetText("操作をキャンセルしました")
					return
				}
				records, err := categorizer.ParseInputRecordsWithOptions(path, opts)
				if err != nil {
					showError(win, err)
					statusLabel.SetText("入力読込エラー")
					return
				}
				if len(records) == 0 {
					showError(win, fmt.Errorf("入力が空です"))
					statusLabel.SetText("入力が空です")
					return
				}
				preview := buildPreviewText(records)
				ignoreTextChange = true
				textInput.SetText(preview)
				ignoreTextChange = false
				pendingRecords = records
				usePendingRecords = true
				statusLabel.SetText(fmt.Sprintf("%s から %d 件読み込みました", filepath.Base(path), len(records)))
			})
		}, win)
		fd.SetFilter(storageFilter([]string{".txt", ".csv", ".tsv"}))
		fd.Show()
	})

	exportBtn := widget.NewButton("結果をCSV出力", func() {
		displayMu.Lock()
		count := len(displayResults)
		displayMu.Unlock()
		if count == 0 {
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
			displayMu.Lock()
			data := buildResultRecordsFromDisplay(displayResults)
			displayMu.Unlock()
			for _, row := range data {
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

	var batchBtn *widget.Button
	batchBtn = widget.NewButton("CSV一括分類", func() {
		batchBtn.Disable()
		statusLabel.SetText("カテゴリ一覧を選択してください")
		catDialog := dialog.NewFileOpen(func(catRC fyne.URIReadCloser, err error) {
			if err != nil {
				showError(win, err)
				batchBtn.Enable()
				statusLabel.SetText("エラーが発生しました")
				return
			}
			if catRC == nil {
				batchBtn.Enable()
				statusLabel.SetText("操作をキャンセルしました")
				return
			}
			catPath := catRC.URI().Path()
			catRC.Close()
			metadata, err := categorizer.ReadCategoryFileMetadata(catPath)
			if err != nil {
				showError(win, err)
				batchBtn.Enable()
				statusLabel.SetText("カテゴリ一覧の読み込みに失敗しました")
				return
			}
			showCategoryColumnSelector(win, catPath, metadata, func(column string, ok bool) {
				if !ok {
					batchBtn.Enable()
					statusLabel.SetText("操作をキャンセルしました")
					return
				}
				statusLabel.SetText("カテゴリ一覧を読み込み中...")
				go func(catPath string, column string) {
					categories, err := categorizer.ParseCategoryListWithOptions(catPath, categorizer.CategoryParseOptions{Column: column})
					if err != nil {
						fyne.Do(func() {
							showError(win, err)
							batchBtn.Enable()
							statusLabel.SetText("カテゴリ一覧の読み込みに失敗しました")
						})
						return
					}
					if len(categories) == 0 {
						fyne.Do(func() {
							showError(win, fmt.Errorf("カテゴリが見つかりません"))
							batchBtn.Enable()
							statusLabel.SetText("カテゴリ一覧の読み込みに失敗しました")
						})
						return
					}
					fyne.Do(func() {
						statusLabel.SetText("発表CSV/TSVを選択してください")
					})
					fyne.Do(func() {
						recDialog := dialog.NewFileOpen(func(recRC fyne.URIReadCloser, err error) {
							if err != nil {
								showError(win, err)
								batchBtn.Enable()
								statusLabel.SetText("エラーが発生しました")
								return
							}
							if recRC == nil {
								batchBtn.Enable()
								statusLabel.SetText("操作をキャンセルしました")
								return
							}
							recPath := recRC.URI().Path()
							recRC.Close()
							inputMeta, err := categorizer.ReadInputFileMetadata(recPath)
							if err != nil {
								showError(win, err)
								batchBtn.Enable()
								statusLabel.SetText("入力読込エラー")
								return
							}
							showInputColumnSelector(win, recPath, inputMeta, func(inputOpts categorizer.InputParseOptions, ok bool) {
								if !ok {
									batchBtn.Enable()
									statusLabel.SetText("操作をキャンセルしました")
									return
								}
								go func(cat []string, catPath, recPath string, opts categorizer.InputParseOptions) {
									defer fyne.Do(func() { batchBtn.Enable() })
									fyne.Do(func() { statusLabel.SetText("分類の準備中...") })
									if err := service.LoadSeeds(ctx, cat); err != nil {
										fyne.Do(func() {
											statusLabel.SetText("シード読み込みエラー")
											showError(win, fmt.Errorf("シードの読み込みに失敗しました: %w", err))
										})
										return
									}
									records, err := categorizer.ParseInputRecordsWithOptions(recPath, opts)
									if err != nil {
										fyne.Do(func() {
											statusLabel.SetText("入力読込エラー")
											showError(win, err)
										})
										return
									}
									if len(records) == 0 {
										fyne.Do(func() {
											statusLabel.SetText("入力が空です")
											showError(win, fmt.Errorf("入力が空です"))
										})
										return
									}
									start := time.Now()
									rows, err := classifyRecords(ctx, service, records)
									if err != nil {
										fyne.Do(func() {
											statusLabel.SetText("分類エラー")
											showError(win, err)
										})
										return
									}
									outputPath, err := saveResultsCSV("csv", records, rows)
									if err != nil {
										fyne.Do(func() {
											statusLabel.SetText("保存エラー")
											showError(win, err)
										})
										return
									}
									preview := buildPreviewText(records)
									updateResults(records, rows)
									fyne.Do(func() {
										seedStatus.SetText(fmt.Sprintf("シード数: %d", service.SeedCount()))
										seedInput.SetText(strings.Join(cat, "\n"))
										cfgMu.Lock()
										cfg.SeedsPath = catPath
										cfgMu.Unlock()
										saveConfig()
										ignoreTextChange = true
										textInput.SetText(preview)
										ignoreTextChange = false
										pendingRecords = records
										usePendingRecords = true
										statusLabel.SetText(fmt.Sprintf("%d件 %.2fs (保存: %s)", len(rows), time.Since(start).Seconds(), filepath.Base(outputPath)))
										dialog.ShowInformation("分類完了", fmt.Sprintf("分類結果を %s に保存しました。", outputPath), win)
									})
								}(append([]string(nil), categories...), catPath, recPath, inputOpts)
							})
						}, win)
						recDialog.SetFilter(storageFilter([]string{".csv", ".tsv"}))
						recDialog.Show()
					})
				}(catPath, column)
			})
		}, win)
		catDialog.SetFilter(storageFilter([]string{".csv", ".tsv"}))
		catDialog.Show()
	})

	modeSelect := widget.NewSelect([]string{string(categorizer.ModeSeeded), string(categorizer.ModeMixed), string(categorizer.ModeSplit)}, func(val string) {
		cfgMu.Lock()
		cfg.Mode = categorizer.Mode(val)
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	})
	modeSelect.SetSelected(string(cfg.Mode))

	topKLabel := widget.NewLabel(fmt.Sprintf("Top-K: %d", cfg.TopK))
	topKSlider := widget.NewSlider(3, 5)
	topKSlider.Step = 1
	topKSlider.SetValue(float64(cfg.TopK))
	topKSlider.OnChanged = func(v float64) {
		val := int(v + 0.5)
		if val < 3 {
			val = 3
		}
		if val > 5 {
			val = 5
		}
		topKLabel.SetText(fmt.Sprintf("Top-K: %d", val))
		cfgMu.Lock()
		cfg.TopK = val
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	}

	weightNDCLabel := widget.NewLabel(fmt.Sprintf("NDC重み: %.2f", cfg.WeightNDC))
	weightNDCSlider := widget.NewSlider(0.7, 1.0)
	weightNDCSlider.Step = 0.01
	weightNDCSlider.SetValue(float64(cfg.WeightNDC))
	weightNDCSlider.OnChanged = func(v float64) {
		weightNDCLabel.SetText(fmt.Sprintf("NDC重み: %.2f", v))
		cfgMu.Lock()
		cfg.WeightNDC = float32(v)
		localCfg := cfg
		cfgMu.Unlock()
		service.UpdateConfig(localCfg)
		saveConfig()
	}
	if !cfg.UseNDC {
		weightNDCSlider.Disable()
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
		if checked {
			weightNDCSlider.Enable()
		} else {
			weightNDCSlider.Disable()
		}
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

	buttonRow := container.NewGridWithColumns(2, classifyBtn, loadTextFileBtn, exportBtn, batchBtn)

	controls := container.NewVBox(
		widget.NewLabel("テキスト入力"),
		textInput,
		buttonRow,
		statusLabel,
		widget.NewSeparator(),
		widget.NewLabel("シードカテゴリ"),
		seedInput,
		container.NewHBox(loadSeedsBtn, loadSeedsFileBtn, seedStatus),
		widget.NewSeparator(),
		widget.NewLabel("設定"),
		modeSelect,
		container.NewHBox(topKLabel, topKSlider),
		container.NewHBox(weightNDCLabel, weightNDCSlider),
		container.NewHBox(clusterCheck, clusterLabel, clusterSlider),
		useNDCCheck,
		widget.NewSeparator(),
		widget.NewLabel("ログ"),
		logContainer,
	)

	infoLabel := widget.NewLabel("項目を選択すると詳細が表示されます")
	infoLabel.Wrapping = fyne.TextWrapWord
	rightPanel := container.NewBorder(nil, infoLabel, nil, nil, resultList)

	root := container.NewHSplit(controls, rightPanel)
	root.Offset = 0.42
	win.SetContent(root)

	win.ShowAndRun()
}

type columnChoice struct {
	Display string
	Value   string
}

func buildColumnChoices(columns []string, blankLabel string) []columnChoice {
	choices := make([]columnChoice, 0, len(columns)+1)
	if blankLabel != "" {
		choices = append(choices, columnChoice{Display: blankLabel, Value: ""})
	}
	for i, col := range columns {
		displayName := col
		if displayName == "" {
			displayName = fmt.Sprintf("列%d", i+1)
		}
		display := fmt.Sprintf("%s (列%d)", displayName, i+1)
		value := col
		if value == "" {
			value = fmt.Sprintf("#%d", i+1)
		}
		choices = append(choices, columnChoice{Display: display, Value: value})
	}
	return choices
}

func choiceDisplays(choices []columnChoice) []string {
	labels := make([]string, len(choices))
	for i, c := range choices {
		labels[i] = c.Display
	}
	return labels
}

func findChoiceByValue(choices []columnChoice, value string) (string, bool) {
	value = strings.TrimSpace(value)
	for _, c := range choices {
		if c.Value == value {
			return c.Display, true
		}
	}
	return "", false
}

func choiceValue(choices []columnChoice, selected string) string {
	selected = strings.TrimSpace(selected)
	for _, c := range choices {
		if c.Display == selected {
			return strings.TrimSpace(c.Value)
		}
	}
	return ""
}

func showInputColumnSelector(win fyne.Window, path string, metadata categorizer.InputFileMetadata, onComplete func(categorizer.InputParseOptions, bool)) {
	if len(metadata.Columns) == 0 {
		onComplete(categorizer.InputParseOptions{}, true)
		return
	}
	indexChoices := buildColumnChoices(metadata.Columns, "未使用")
	titleChoices := buildColumnChoices(metadata.Columns, "未使用")
	bodyChoices := buildColumnChoices(metadata.Columns, "未使用")
	textChoices := buildColumnChoices(metadata.Columns, "自動選択")

	indexSelect := widget.NewSelect(choiceDisplays(indexChoices), nil)
	indexSelect.PlaceHolder = "未使用"
	if display, ok := findChoiceByValue(indexChoices, metadata.Suggested.IndexColumn); ok {
		indexSelect.SetSelected(display)
	}

	titleSelect := widget.NewSelect(choiceDisplays(titleChoices), nil)
	titleSelect.PlaceHolder = "未使用"
	if display, ok := findChoiceByValue(titleChoices, metadata.Suggested.TitleColumn); ok {
		titleSelect.SetSelected(display)
	}

	bodySelect := widget.NewSelect(choiceDisplays(bodyChoices), nil)
	bodySelect.PlaceHolder = "未使用"
	if display, ok := findChoiceByValue(bodyChoices, metadata.Suggested.BodyColumn); ok {
		bodySelect.SetSelected(display)
	}

	textSelect := widget.NewSelect(choiceDisplays(textChoices), nil)
	textSelect.PlaceHolder = "自動選択"
	if display, ok := findChoiceByValue(textChoices, metadata.Suggested.TextColumn); ok {
		textSelect.SetSelected(display)
	}

	description := widget.NewLabel(fmt.Sprintf("%s の列を選択してください。空欄は自動判定になります。", filepath.Base(path)))
	description.Wrapping = fyne.TextWrapWord
	form := widget.NewForm(
		widget.NewFormItem("インデックス列", indexSelect),
		widget.NewFormItem("タイトル列", titleSelect),
		widget.NewFormItem("本文列", bodySelect),
		widget.NewFormItem("テキスト列", textSelect),
	)

	content := container.NewVBox(description, widget.NewSeparator(), form)
	dialog := dialog.NewCustomConfirm("列の選択", "OK", "キャンセル", content, func(ok bool) {
		if !ok {
			onComplete(categorizer.InputParseOptions{}, false)
			return
		}
		opts := categorizer.InputParseOptions{
			IndexColumn: choiceValue(indexChoices, indexSelect.Selected),
			TitleColumn: choiceValue(titleChoices, titleSelect.Selected),
			BodyColumn:  choiceValue(bodyChoices, bodySelect.Selected),
			TextColumn:  choiceValue(textChoices, textSelect.Selected),
		}
		onComplete(opts, true)
	}, win)
	dialog.Resize(fyne.NewSize(420, 340))
	dialog.Show()
}

func showCategoryColumnSelector(win fyne.Window, path string, metadata categorizer.CategoryFileMetadata, onComplete func(string, bool)) {
	if len(metadata.Columns) == 0 {
		onComplete("", true)
		return
	}
	choices := buildColumnChoices(metadata.Columns, "自動選択")
	selectWidget := widget.NewSelect(choiceDisplays(choices), nil)
	selectWidget.PlaceHolder = "自動選択"
	if display, ok := findChoiceByValue(choices, metadata.Suggested); ok {
		selectWidget.SetSelected(display)
	}
	description := widget.NewLabel(fmt.Sprintf("%s からカテゴリ列を選択してください。", filepath.Base(path)))
	description.Wrapping = fyne.TextWrapWord
	content := container.NewVBox(description, widget.NewSeparator(), selectWidget)
	dialog := dialog.NewCustomConfirm("カテゴリ列の選択", "OK", "キャンセル", content, func(ok bool) {
		if !ok {
			onComplete("", false)
			return
		}
		onComplete(choiceValue(choices, selectWidget.Selected), true)
	}, win)
	dialog.Resize(fyne.NewSize(360, 240))
	dialog.Show()
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

func storageFilter(exts []string) storage.FileFilter {
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

func classifyRecords(ctx context.Context, service *categorizer.Service, records []categorizer.InputRecord) ([]categorizer.ResultRow, error) {
	texts := make([]string, len(records))
	for i, rec := range records {
		texts[i] = rec.Text
	}
	return service.ClassifyAll(ctx, texts)
}

func buildResultRecords(records []categorizer.InputRecord, rows []categorizer.ResultRow) [][]string {
	data := make([][]string, 0, len(records)+1)
	header := []string{"発表インデックス", "発表のタイトル", "発表の概要", "推定カテゴリ", "スコア"}
	data = append(data, header)
	for i, rec := range records {
		body := rec.Body
		if body == "" {
			body = rec.Text
		}
		label := ""
		score := ""
		if i < len(rows) {
			if best, ok := pickBestSuggestion(rows[i]); ok {
				label = best.Label
				score = fmt.Sprintf("%.3f", best.Score)
			}
		}
		data = append(data, []string{rec.Index, rec.Title, body, label, score})
	}
	return data
}

func saveResultsCSV(outputDir string, records []categorizer.InputRecord, rows []categorizer.ResultRow) (string, error) {
	if len(records) != len(rows) {
		return "", fmt.Errorf("records/results length mismatch: %d vs %d", len(records), len(rows))
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}
	filename := fmt.Sprintf("result_%s.csv", time.Now().Format("200601021504"))
	path := filepath.Join(outputDir, filename)
	f, err := os.Create(path)
	if err != nil {
		return "", fmt.Errorf("create result file: %w", err)
	}
	defer f.Close()
	writer := csv.NewWriter(f)
	data := buildResultRecords(records, rows)
	for _, row := range data {
		if err := writer.Write(row); err != nil {
			return "", fmt.Errorf("write result: %w", err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return "", fmt.Errorf("flush result: %w", err)
	}
	return path, nil
}

func pickBestSuggestion(row categorizer.ResultRow) (categorizer.Suggestion, bool) {
	if len(row.Suggestions) > 0 {
		return row.Suggestions[0], true
	}
	if len(row.NDCSuggestions) > 0 {
		return row.NDCSuggestions[0], true
	}
	return categorizer.Suggestion{}, false
}

func formatDisplayHeader(rec categorizer.InputRecord) string {
	var parts []string
	if rec.Index != "" {
		parts = append(parts, fmt.Sprintf("#%s", rec.Index))
	}
	if rec.Title != "" {
		parts = append(parts, rec.Title)
	}
	if len(parts) == 0 {
		return truncateText(rec.Text, 60)
	}
	return strings.Join(parts, " ")
}

func formatDisplaySummary(rec categorizer.InputRecord) string {
	body := rec.Body
	if body == "" {
		body = rec.Text
	}
	if rec.Title != "" && body == rec.Title {
		return ""
	}
	return body
}

func buildDetailMessage(item displayResult) string {
	var b strings.Builder
	if item.Input.Index != "" {
		fmt.Fprintf(&b, "インデックス: %s\n", item.Input.Index)
	}
	if item.Input.Title != "" {
		fmt.Fprintf(&b, "タイトル: %s\n\n", item.Input.Title)
	}
	body := formatDisplaySummary(item.Input)
	if body == "" {
		body = item.Input.Text
	}
	fmt.Fprintf(&b, "本文:\n%s\n\n", body)
	if best, ok := pickBestSuggestion(item.Result); ok {
		fmt.Fprintf(&b, "推定カテゴリ: %s (%.3f)\n", best.Label, best.Score)
		fmt.Fprintf(&b, "ソース: %s\n", best.Source)
	} else {
		b.WriteString("推定カテゴリ: 候補なし\n")
	}
	if len(item.Result.Suggestions) > 0 {
		b.WriteString("\nシード候補:\n")
		for i, s := range item.Result.Suggestions {
			fmt.Fprintf(&b, "  [%d] %s (%.3f)\n", i+1, s.Label, s.Score)
		}
	}
	if len(item.Result.NDCSuggestions) > 0 {
		b.WriteString("\nNDC候補:\n")
		for i, s := range item.Result.NDCSuggestions {
			fmt.Fprintf(&b, "  [%d] %s (%.3f)\n", i+1, s.Label, s.Score)
		}
	}
	return b.String()
}

func manualRecordsFromLines(lines []string) []categorizer.InputRecord {
	records := make([]categorizer.InputRecord, len(lines))
	for i, line := range lines {
		records[i] = categorizer.InputRecord{
			Body: line,
			Text: line,
		}
	}
	return records
}

func buildPreviewText(records []categorizer.InputRecord) string {
	parts := make([]string, 0, len(records))
	for _, rec := range records {
		var b strings.Builder
		if rec.Title != "" {
			b.WriteString(rec.Title)
		}
		body := rec.Body
		if body == "" {
			body = rec.Text
		}
		if body != "" && body != rec.Title {
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(body)
		}
		if b.Len() == 0 {
			b.WriteString(rec.Text)
		}
		parts = append(parts, b.String())
	}
	return strings.Join(parts, "\n---\n")
}

func buildResultRecordsFromDisplay(results []displayResult) [][]string {
	records := make([]categorizer.InputRecord, len(results))
	rows := make([]categorizer.ResultRow, len(results))
	for i, item := range results {
		records[i] = item.Input
		rows[i] = item.Result
	}
	return buildResultRecords(records, rows)
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
