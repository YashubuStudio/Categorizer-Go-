package categorizer

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

var textColumnCandidates = []string{"text", "本文", "content", "body", "message"}

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

// ParseTextFile reads a text/CSV/TSV file and extracts candidate sentences.
func ParseTextFile(path string) ([]string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".csv":
		return parseDelimited(path, ',')
	case ".tsv":
		return parseDelimited(path, '\t')
	default:
		return parsePlainText(path)
	}
}

func parsePlainText(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open text file: %w", err)
	}
	defer f.Close()
	var out []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			out = append(out, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan text file: %w", err)
	}
	return out, nil
}

func parseDelimited(path string, comma rune) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", filepath.Base(path), err)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	reader.Comma = comma
	reader.ReuseRecord = true
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", filepath.Base(path), err)
	}
	if len(rows) == 0 {
		return nil, errors.New("empty file")
	}
	header := rows[0]
	colIndex := -1
	for i, col := range header {
		for _, cand := range textColumnCandidates {
			if strings.EqualFold(strings.TrimSpace(col), cand) {
				colIndex = i
				break
			}
		}
		if colIndex >= 0 {
			break
		}
	}
	if colIndex < 0 {
		return nil, fmt.Errorf("text column not found in %s", path)
	}
	out := make([]string, 0, len(rows)-1)
	for _, row := range rows[1:] {
		if colIndex >= len(row) {
			continue
		}
		value := strings.TrimSpace(row[colIndex])
		if value != "" {
			out = append(out, value)
		}
	}
	return out, nil
}
