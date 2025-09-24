package app

import (
	"fmt"
	"strings"
)

func suggestionAt(s []Suggestion, idx int) (Suggestion, bool) {
	if idx < 0 || idx >= len(s) {
		return Suggestion{}, false
	}
	return s[idx], true
}

func suggestionLabel(s Suggestion) string {
	if len(s.Aliases) == 0 {
		return s.Label
	}
	return fmt.Sprintf("%s [%s]", s.Label, strings.Join(s.Aliases, " / "))
}

func formatSuggestionAt(list []Suggestion, idx int, showSource bool) string {
	if sug, ok := suggestionAt(list, idx); ok {
		label := suggestionLabel(sug)
		if showSource && sug.Source != "" {
			return fmt.Sprintf("%s\n%.3f (%s)", label, sug.Score, sug.Source)
		}
		return fmt.Sprintf("%s\n%.3f", label, sug.Score)
	}
	return ""
}

func suggestionSources(list []Suggestion) string {
	seen := make(map[string]struct{})
	out := make([]string, 0, len(list))
	for _, s := range list {
		for _, part := range strings.Split(s.Source, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if _, ok := seen[part]; ok {
				continue
			}
			seen[part] = struct{}{}
			out = append(out, part)
		}
	}
	return strings.Join(out, ",")
}
