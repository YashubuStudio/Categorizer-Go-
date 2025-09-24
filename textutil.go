package main

import (
	"strings"

	"golang.org/x/text/unicode/norm"
)

func normalize(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Join(strings.Fields(s), " ")
	return norm.NFKC.String(s)
}

func uniqueNormalized(labels []string) []string {
	seen := make(map[string]struct{})
	res := make([]string, 0, len(labels))
	for _, lab := range labels {
		normed := normalize(lab)
		if normed == "" {
			continue
		}
		if _, ok := seen[normed]; ok {
			continue
		}
		seen[normed] = struct{}{}
		res = append(res, normed)
	}
	return res
}
