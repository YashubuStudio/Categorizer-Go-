package app

import (
	"strings"

	"golang.org/x/text/unicode/norm"
)

func normalize(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	s = norm.NFKC.String(s)
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return ""
	}
	return strings.Join(fields, " ")
}

func normalizeKey(s string) string {
	normed := normalize(s)
	if normed == "" {
		return ""
	}
	return strings.ToLower(normed)
}

func normalizeText(s string) string {
	normed := normalize(s)
	if normed == "" {
		return ""
	}
	return strings.ToLower(normed)
}

func uniqueNormalized(labels []string) []string {
	seen := make(map[string]struct{})
	res := make([]string, 0, len(labels))
	for _, lab := range labels {
		clean := normalize(lab)
		if clean == "" {
			continue
		}
		key := normalizeKey(clean)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		res = append(res, clean)
	}
	return res
}
