package categorizer

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// NormalizeText performs Unicode normalization and trims whitespace.
func NormalizeText(text string) string {
	normed := norm.NFKC.String(text)
	normed = strings.TrimSpace(normed)
	// Collapse internal control characters except newlines.
	normed = strings.Map(func(r rune) rune {
		if r == '\n' || r == '\t' {
			return r
		}
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, normed)
	return normed
}

// NormalizeAll normalizes a slice of strings in place.
func NormalizeAll(texts []string) []string {
	out := make([]string, len(texts))
	for i, t := range texts {
		out[i] = NormalizeText(t)
	}
	return out
}
