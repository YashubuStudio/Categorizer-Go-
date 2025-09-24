package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"strings"
)

type csvColumnChoice struct {
	Index int
	Label string
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

func readCSVRecords(data []byte, delim rune) ([][]string, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.Comma = delim
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, errors.New("CSVが空です")
	}
	return records, nil
}

func extractCSVColumn(records [][]string, idx int, hasHeader bool) []string {
	start := 0
	if hasHeader {
		start = 1
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
	return res
}

func buildCSVColumnChoices(records [][]string, hasHeader bool) []csvColumnChoice {
	maxCols := 0
	for _, row := range records {
		if len(row) > maxCols {
			maxCols = len(row)
		}
	}
	choices := make([]csvColumnChoice, 0, maxCols)
	for col := 0; col < maxCols; col++ {
		header := fmt.Sprintf("列%d", col+1)
		if hasHeader && len(records) > 0 && col < len(records[0]) {
			h := strings.TrimSpace(records[0][col])
			if h != "" {
				header = h
			}
		}
		sample := csvColumnSample(records, col, hasHeader)
		label := fmt.Sprintf("[%d] %s", col+1, header)
		if sample != "" {
			label = fmt.Sprintf("%s (例: %s)", label, sample)
		}
		choices = append(choices, csvColumnChoice{Index: col, Label: label})
	}
	return choices
}

func csvColumnSample(records [][]string, col int, hasHeader bool) string {
	start := 0
	if hasHeader {
		start = 1
	}
	for i := start; i < len(records); i++ {
		row := records[i]
		if col >= len(row) {
			continue
		}
		val := strings.TrimSpace(row[col])
		if val == "" {
			continue
		}
		return truncateSampleValue(val, 20)
	}
	return ""
}

func truncateSampleValue(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "…"
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
