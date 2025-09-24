package categorizer

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// InputParseOptions allows callers to choose which CSV columns map to record fields.
type InputParseOptions struct {
	IndexColumn string
	TitleColumn string
	BodyColumn  string
	TextColumn  string
}

// InputFileMetadata provides header information and automatic column suggestions.
type InputFileMetadata struct {
	Columns   []string
	Suggested InputParseOptions
}

// CategoryParseOptions allows callers to select which column to use as category labels.
type CategoryParseOptions struct {
	Column string
}

// CategoryFileMetadata holds header data and the detected category column.
type CategoryFileMetadata struct {
	Columns   []string
	Suggested string
}

// ParseSeedFile reads the provided file and extracts seed labels using newline or comma separators.
func ParseSeedFile(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open seed file: %w", err)
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read seed file: %w", err)
	}
	return parseSeedsFromString(string(data)), nil
}

// ParseSeeds converts raw string input into normalized seed labels.
func ParseSeeds(data string) []string {
	return parseSeedsFromString(data)
}

// parseSeedsFromString splits seed definitions by comma or newline.
func parseSeedsFromString(data string) []string {
	data = strings.ReplaceAll(data, "\r\n", "\n")
	tokens := strings.FieldsFunc(data, func(r rune) bool {
		return r == '\n' || r == ',' || r == ';'
	})
	out := make([]string, 0, len(tokens))
	seen := make(map[string]struct{})
	for _, token := range tokens {
		token = strings.TrimSpace(token)
		if token == "" {
			continue
		}
		normalized := NormalizeText(token)
		if _, ok := seen[normalized]; ok {
			continue
		}
		seen[normalized] = struct{}{}
		out = append(out, token)
	}
	return out
}

// ParseInputRecords reads a file and returns rich input records combining titles and bodies when available.
func ParseInputRecords(path string) ([]InputRecord, error) {
	return ParseInputRecordsWithOptions(path, InputParseOptions{})
}

// ParseInputRecordsWithOptions allows callers to specify column mappings when reading structured files.
func ParseInputRecordsWithOptions(path string, opts InputParseOptions) ([]InputRecord, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".csv":
		return parseDelimitedRecords(path, ',', opts)
	case ".tsv":
		return parseDelimitedRecords(path, '\t', opts)
	default:
		return parsePlainTextRecords(path)
	}
}

// ParseTextFile reads a text/CSV/TSV file and extracts combined texts for backward compatibility.
func ParseTextFile(path string) ([]string, error) {
	records, err := ParseInputRecordsWithOptions(path, InputParseOptions{})
	if err != nil {
		return nil, err
	}
	out := make([]string, len(records))
	for i, record := range records {
		out[i] = record.Text
	}
	return out, nil
}

// ParseCategoryList extracts unique category labels from a CSV/TSV file.
func ParseCategoryList(path string) ([]string, error) {
	return ParseCategoryListWithOptions(path, CategoryParseOptions{})
}

// ParseCategoryListWithOptions extracts unique category labels honoring a caller provided column selection.
func ParseCategoryListWithOptions(path string, opts CategoryParseOptions) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", filepath.Base(path), err)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	if strings.EqualFold(filepath.Ext(path), ".tsv") {
		reader.Comma = '\t'
	}
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", filepath.Base(path), err)
	}
	if len(rows) == 0 {
		return nil, errors.New("empty category file")
	}
	header := make([]string, len(rows[0]))
	for i, cell := range rows[0] {
		header[i] = cleanCell(cell)
	}
	col, start, err := resolveCategoryColumn(header, opts.Column)
	if err != nil {
		return nil, err
	}
	categories := make([]string, 0, len(rows)-start)
	seen := make(map[string]struct{})
	for _, row := range rows[start:] {
		if col >= len(row) {
			continue
		}
		value := cleanCell(row[col])
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		categories = append(categories, value)
	}
	if len(categories) == 0 {
		return nil, fmt.Errorf("no categories found in %s", path)
	}
	return categories, nil
}

func parsePlainTextRecords(path string) ([]InputRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open text file: %w", err)
	}
	defer f.Close()
	var out []InputRecord
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := cleanCell(scanner.Text())
		if line == "" {
			continue
		}
		out = append(out, InputRecord{Text: line, Body: line})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan text file: %w", err)
	}
	return out, nil
}

func parseDelimitedRecords(path string, comma rune, opts InputParseOptions) ([]InputRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", filepath.Base(path), err)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	reader.Comma = comma
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", filepath.Base(path), err)
	}
	if len(rows) == 0 {
		return nil, errors.New("empty file")
	}
	header := make([]string, len(rows[0]))
	for i, cell := range rows[0] {
		header[i] = cleanCell(cell)
	}
	resolved, skipHeader, err := resolveInputColumns(header, opts)
	if err != nil {
		return nil, err
	}
	start := 0
	if skipHeader {
		start = 1
	}
	records := make([]InputRecord, 0, len(rows)-start)
	for _, row := range rows[start:] {
		rec := InputRecord{}
		if resolved.Index.Index >= 0 && resolved.Index.Index < len(row) {
			rec.Index = cleanCell(row[resolved.Index.Index])
		}
		if resolved.Title.Index >= 0 && resolved.Title.Index < len(row) {
			rec.Title = cleanCell(row[resolved.Title.Index])
		}
		var summaryVal string
		if resolved.Body.Index >= 0 && resolved.Body.Index < len(row) {
			summaryVal = cleanCell(row[resolved.Body.Index])
		}
		var textVal string
		if resolved.Text.Index >= 0 && resolved.Text.Index < len(row) {
			textVal = cleanCell(row[resolved.Text.Index])
		}
		if summaryVal == "" {
			summaryVal = textVal
		}
		rec.Body = summaryVal
		combined := combineParts(rec.Title, summaryVal)
		if combined == "" {
			combined = textVal
		}
		if combined == "" {
			continue
		}
		if rec.Body == "" {
			rec.Body = combined
		}
		rec.Text = combined
		records = append(records, rec)
	}
	return records, nil
}

func cleanCell(v string) string {
	v = strings.TrimSpace(v)
	v = strings.TrimPrefix(v, "\ufeff")
	return v
}

func findColumn(header []string, candidates []string) int {
	for i, col := range header {
		for _, cand := range candidates {
			if strings.EqualFold(col, cand) {
				return i
			}
		}
	}
	return -1
}

func combineParts(title, body string) string {
	var parts []string
	if title != "" {
		parts = append(parts, title)
	}
	if body != "" && body != title {
		parts = append(parts, body)
	}
	return strings.Join(parts, "\n")
}

type columnResult struct {
	Index      int
	FromHeader bool
	HeaderName string
}

type resolvedColumns struct {
	Index columnResult
	Title columnResult
	Body  columnResult
	Text  columnResult
}

func resolveInputColumns(header []string, opts InputParseOptions) (resolvedColumns, bool, error) {
	res := resolvedColumns{
		Index: columnResult{Index: -1},
		Title: columnResult{Index: -1},
		Body:  columnResult{Index: -1},
		Text:  columnResult{Index: -1},
	}
	var err error
	candidates := getColumnCandidates()
	if res.Index, err = pickColumn(header, opts.IndexColumn, candidates.Index); err != nil {
		return res, false, err
	}
	if res.Title, err = pickColumn(header, opts.TitleColumn, candidates.Title); err != nil {
		return res, false, err
	}
	if res.Body, err = pickColumn(header, opts.BodyColumn, candidates.Body); err != nil {
		return res, false, err
	}
	if res.Text, err = pickColumn(header, opts.TextColumn, candidates.Text); err != nil {
		return res, false, err
	}
	skipHeader := res.Index.FromHeader || res.Title.FromHeader || res.Body.FromHeader || res.Text.FromHeader
	if !skipHeader && res.Text.Index < 0 && len(header) > 0 {
		res.Text.Index = 0
		res.Text.FromHeader = false
		res.Text.HeaderName = headerNameForIndex(header, res.Text.Index, false)
	}
	res.Index.HeaderName = headerNameForIndex(header, res.Index.Index, res.Index.FromHeader)
	res.Title.HeaderName = headerNameForIndex(header, res.Title.Index, res.Title.FromHeader)
	res.Body.HeaderName = headerNameForIndex(header, res.Body.Index, res.Body.FromHeader)
	res.Text.HeaderName = headerNameForIndex(header, res.Text.Index, res.Text.FromHeader)
	return res, skipHeader, nil
}

func pickColumn(header []string, explicit string, candidates []string) (columnResult, error) {
	res := columnResult{Index: -1}
	if strings.TrimSpace(explicit) != "" {
		idx, fromHeader, err := matchExplicitColumn(header, explicit)
		if err != nil {
			return res, err
		}
		res.Index = idx
		res.FromHeader = fromHeader
		return res, nil
	}
	idx := findColumn(header, candidates)
	if idx >= 0 {
		res.Index = idx
		res.FromHeader = true
	}
	return res, nil
}

func matchExplicitColumn(header []string, explicit string) (int, bool, error) {
	trimmed := strings.TrimSpace(explicit)
	if trimmed == "" {
		return -1, false, nil
	}
	for i, col := range header {
		if strings.EqualFold(col, trimmed) {
			return i, true, nil
		}
	}
	if strings.HasPrefix(trimmed, "#") {
		idx, err := parseColumnIndex(trimmed)
		if err != nil {
			return -1, false, err
		}
		if idx >= len(header) {
			return -1, false, fmt.Errorf("column index %s is out of range", trimmed)
		}
		return idx, false, nil
	}
	return -1, false, fmt.Errorf("column %q not found", explicit)
}

func parseColumnIndex(token string) (int, error) {
	trimmed := strings.TrimSpace(strings.TrimPrefix(token, "#"))
	if trimmed == "" {
		return -1, fmt.Errorf("invalid column index %q", token)
	}
	idx, err := strconv.Atoi(trimmed)
	if err != nil {
		return -1, fmt.Errorf("invalid column index %q", token)
	}
	if idx <= 0 {
		return -1, fmt.Errorf("column indices are 1-based: %q", token)
	}
	return idx - 1, nil
}

func headerNameForIndex(header []string, idx int, fromHeader bool) string {
	if idx < 0 {
		return ""
	}
	if fromHeader && idx < len(header) {
		if name := header[idx]; name != "" {
			return name
		}
	}
	return fmt.Sprintf("#%d", idx+1)
}

func resolveCategoryColumn(header []string, explicit string) (int, int, error) {
	trimmed := strings.TrimSpace(explicit)
	if trimmed != "" {
		idx, fromHeader, err := matchExplicitColumn(header, trimmed)
		if err != nil {
			return -1, 0, err
		}
		start := 0
		if fromHeader {
			start = 1
		}
		return idx, start, nil
	}
	candidates := getColumnCandidates()
	col := findColumn(header, candidates.Category)
	start := 0
	if col >= 0 {
		start = 1
	} else if len(header) > 0 {
		col = 0
	}
	if col < 0 {
		return -1, 0, errors.New("no usable category column found")
	}
	return col, start, nil
}

// ReadInputFileMetadata returns header information and automatic suggestions for structured files.
func ReadInputFileMetadata(path string) (InputFileMetadata, error) {
	meta := InputFileMetadata{}
	ext := strings.ToLower(filepath.Ext(path))
	if ext != ".csv" && ext != ".tsv" {
		return meta, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return meta, fmt.Errorf("open %s: %w", filepath.Base(path), err)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	if ext == ".tsv" {
		reader.Comma = '\t'
	}
	reader.FieldsPerRecord = -1
	row, err := reader.Read()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return meta, nil
		}
		return meta, fmt.Errorf("read %s: %w", filepath.Base(path), err)
	}
	header := make([]string, len(row))
	for i, cell := range row {
		header[i] = cleanCell(cell)
	}
	meta.Columns = header
	resolved, _, err := resolveInputColumns(header, InputParseOptions{})
	if err == nil {
		meta.Suggested = InputParseOptions{
			IndexColumn: resolved.Index.HeaderName,
			TitleColumn: resolved.Title.HeaderName,
			BodyColumn:  resolved.Body.HeaderName,
			TextColumn:  resolved.Text.HeaderName,
		}
	}
	return meta, nil
}

// ReadCategoryFileMetadata returns header information and automatic suggestions for category files.
func ReadCategoryFileMetadata(path string) (CategoryFileMetadata, error) {
	meta := CategoryFileMetadata{}
	ext := strings.ToLower(filepath.Ext(path))
	if ext != ".csv" && ext != ".tsv" {
		return meta, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return meta, fmt.Errorf("open %s: %w", filepath.Base(path), err)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	if ext == ".tsv" {
		reader.Comma = '\t'
	}
	reader.FieldsPerRecord = -1
	row, err := reader.Read()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return meta, nil
		}
		return meta, fmt.Errorf("read %s: %w", filepath.Base(path), err)
	}
	header := make([]string, len(row))
	for i, cell := range row {
		header[i] = cleanCell(cell)
	}
	meta.Columns = header
	candidates := getColumnCandidates()
	col := findColumn(header, candidates.Category)
	if col >= 0 {
		meta.Suggested = header[col]
		if meta.Suggested == "" {
			meta.Suggested = fmt.Sprintf("#%d", col+1)
		}
	} else if len(header) > 0 {
		meta.Suggested = fmt.Sprintf("#%d", 1)
	}
	return meta, nil
}
