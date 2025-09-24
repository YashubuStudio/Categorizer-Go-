package main

import (
	"context"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"yashubustudio/categorizer/categorizer"
)

type cliOptions struct {
	configPath     string
	inputPath      string
	categoriesPath string
	outputPath     string
	outputDir      string
	inputOpts      categorizer.InputParseOptions
	categoryColumn string
	stdout         bool
}

func main() {
	opts, err := parseFlags()
	if err != nil {
		log.Fatalf("categorizer-cli: %v", err)
	}
	if err := run(opts); err != nil {
		log.Fatalf("categorizer-cli: %v", err)
	}
}

func parseFlags() (cliOptions, error) {
	var opts cliOptions
	flag.StringVar(&opts.configPath, "config", "", "Path to config.json (default: ./config.json)")
	flag.StringVar(&opts.inputPath, "input", "", "CSV/TSV/text file containing texts to classify")
	flag.StringVar(&opts.categoriesPath, "categories", "", "CSV/TSV file containing category labels")
	flag.StringVar(&opts.outputPath, "output", "", "CSV file to write results (default uses --output-dir/result_*.csv)")
	flag.StringVar(&opts.outputDir, "output-dir", "csv", "Directory where result CSVs are written when --output is omitted")
	flag.StringVar(&opts.inputOpts.IndexColumn, "input-index-column", "", "Column name or #index for the presentation index column")
	flag.StringVar(&opts.inputOpts.TitleColumn, "input-title-column", "", "Column name or #index for the presentation title column")
	flag.StringVar(&opts.inputOpts.BodyColumn, "input-body-column", "", "Column name or #index for the presentation body column")
	flag.StringVar(&opts.inputOpts.TextColumn, "input-text-column", "", "Column name or #index for the fallback text column")
	flag.StringVar(&opts.categoryColumn, "category-column", "", "Column name or #index for category labels")
	flag.BoolVar(&opts.stdout, "stdout", false, "Print summary results to STDOUT")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s --input FILE --categories FILE [options]\n\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	opts.configPath = strings.TrimSpace(opts.configPath)
	opts.inputPath = strings.TrimSpace(opts.inputPath)
	opts.categoriesPath = strings.TrimSpace(opts.categoriesPath)
	opts.outputPath = strings.TrimSpace(opts.outputPath)
	opts.outputDir = strings.TrimSpace(opts.outputDir)
	opts.categoryColumn = strings.TrimSpace(opts.categoryColumn)

	if opts.inputPath == "" {
		flag.Usage()
		return opts, errors.New("missing required --input file")
	}
	if opts.categoriesPath == "" {
		flag.Usage()
		return opts, errors.New("missing required --categories file")
	}
	return opts, nil
}

func run(opts cliOptions) error {
	cfg, err := categorizer.LoadConfig(opts.configPath)
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

	categories, err := categorizer.ParseCategoryListWithOptions(opts.categoriesPath, categorizer.CategoryParseOptions{Column: opts.categoryColumn})
	if err != nil {
		return fmt.Errorf("read category list: %w", err)
	}
	if err := service.LoadSeeds(ctx, categories); err != nil {
		return fmt.Errorf("load categories: %w", err)
	}

	records, err := categorizer.ParseInputRecordsWithOptions(opts.inputPath, opts.inputOpts)
	if err != nil {
		return fmt.Errorf("read input records: %w", err)
	}
	if len(records) == 0 {
		return errors.New("input file does not contain any texts")
	}

	rows, err := classify(ctx, service, records)
	if err != nil {
		return fmt.Errorf("classify: %w", err)
	}

	outputPath, err := resolveOutputPath(opts.outputPath, opts.outputDir)
	if err != nil {
		return err
	}
	if err := writeResultCSV(outputPath, records, rows); err != nil {
		return err
	}
	fmt.Printf("分類結果を %s に保存しました\n", outputPath)

	if opts.stdout {
		printSummary(records, rows)
	}
	return nil
}

func classify(ctx context.Context, service *categorizer.Service, records []categorizer.InputRecord) ([]categorizer.ResultRow, error) {
	texts := make([]string, len(records))
	for i, rec := range records {
		texts[i] = rec.Text
	}
	return service.ClassifyAll(ctx, texts)
}

func resolveOutputPath(path, dir string) (string, error) {
	if path != "" {
		absPath, err := filepath.Abs(path)
		if err != nil {
			return "", fmt.Errorf("resolve output path: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(absPath), 0o755); err != nil {
			return "", fmt.Errorf("create output directory: %w", err)
		}
		return absPath, nil
	}
	if dir == "" {
		dir = "csv"
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return "", fmt.Errorf("resolve output dir: %w", err)
	}
	if err := os.MkdirAll(absDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}
	filename := fmt.Sprintf("result_%s.csv", time.Now().Format("20060102150405"))
	return filepath.Join(absDir, filename), nil
}

func writeResultCSV(path string, records []categorizer.InputRecord, rows []categorizer.ResultRow) error {
	if len(records) != len(rows) {
		return fmt.Errorf("records/results length mismatch: %d vs %d", len(records), len(rows))
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create result file: %w", err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	header := []string{"発表インデックス", "発表のタイトル", "発表の概要", "推定カテゴリ", "スコア"}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	for i, rec := range records {
		body := rec.Body
		if body == "" {
			body = rec.Text
		}
		label := ""
		score := ""
		if suggestion, ok := pickBestSuggestion(rows[i]); ok {
			label = suggestion.Label
			score = fmt.Sprintf("%.3f", suggestion.Score)
		}
		row := []string{rec.Index, rec.Title, body, label, score}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("write row %d: %w", i, err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush result: %w", err)
	}
	return nil
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

func printSummary(records []categorizer.InputRecord, rows []categorizer.ResultRow) {
	fmt.Println()
	fmt.Println("==== 分類結果プレビュー ====")
	for i := range records {
		rec := records[i]
		row := rows[i]
		fmt.Printf("%d. %s\n", i+1, summarizeRecord(rec))
		if len(row.Suggestions) > 0 {
			fmt.Println("    シード候補:")
			printSuggestions(row.Suggestions)
		}
		if len(row.NDCSuggestions) > 0 {
			fmt.Println("    NDC候補:")
			printSuggestions(row.NDCSuggestions)
		}
		if len(row.Suggestions) == 0 && len(row.NDCSuggestions) == 0 {
			fmt.Println("    提案なし")
		}
	}
}

func printSuggestions(suggestions []categorizer.Suggestion) {
	limit := 3
	if len(suggestions) < limit {
		limit = len(suggestions)
	}
	for i := 0; i < limit; i++ {
		suggestion := suggestions[i]
		fmt.Printf("      - %s (score=%.3f)\n", suggestion.Label, suggestion.Score)
	}
}

func summarizeRecord(rec categorizer.InputRecord) string {
	var parts []string
	if strings.TrimSpace(rec.Index) != "" {
		parts = append(parts, "#"+strings.TrimSpace(rec.Index))
	}
	if strings.TrimSpace(rec.Title) != "" {
		parts = append(parts, strings.TrimSpace(rec.Title))
	}
	if len(parts) > 0 {
		return strings.Join(parts, " ")
	}
	text := strings.TrimSpace(rec.Text)
	if text == "" {
		return "(空のテキスト)"
	}
	runeText := []rune(text)
	if len(runeText) > 60 {
		return string(runeText[:60]) + "…"
	}
	return text
}
