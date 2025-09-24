package categorizer

import "sync"

// ColumnCandidates defines possible header names for auto-detecting CSV/TSV columns.
type ColumnCandidates struct {
	Text     []string `json:"text"`
	Title    []string `json:"title"`
	Body     []string `json:"body"`
	Index    []string `json:"index"`
	Category []string `json:"category"`
}

var (
	columnCandidatesMu  sync.RWMutex
	activeColumnOptions = defaultColumnCandidates()
)

func defaultColumnCandidates() ColumnCandidates {
	return ColumnCandidates{
		Text:     []string{"text", "本文", "content", "body", "message", "発表抜粋"},
		Title:    []string{"title", "タイトル", "発表のタイトル", "題名", "名称"},
		Body:     []string{"summary", "概要", "description", "発表の概要", "本文", "発表抜粋"},
		Index:    []string{"id", "index", "no", "番号", "発表インデックス"},
		Category: []string{"カテゴリ", "カテゴリー", "category"},
	}
}

// DefaultColumnCandidates returns the built-in column detection candidates.
func DefaultColumnCandidates() ColumnCandidates {
	return defaultColumnCandidates().clone()
}

// SetColumnCandidates updates the column detection candidates used during auto-detection.
// Fields left nil fall back to the built-in defaults, allowing callers to override only
// the parts they need.
func SetColumnCandidates(candidates ColumnCandidates) {
	columnCandidatesMu.Lock()
	defer columnCandidatesMu.Unlock()
	activeColumnOptions = candidates.withDefaults()
}

func getColumnCandidates() ColumnCandidates {
	columnCandidatesMu.RLock()
	defer columnCandidatesMu.RUnlock()
	return activeColumnOptions.clone()
}

func (c ColumnCandidates) withDefaults() ColumnCandidates {
	defaults := defaultColumnCandidates()
	return ColumnCandidates{
		Text:     pickStrings(c.Text, defaults.Text),
		Title:    pickStrings(c.Title, defaults.Title),
		Body:     pickStrings(c.Body, defaults.Body),
		Index:    pickStrings(c.Index, defaults.Index),
		Category: pickStrings(c.Category, defaults.Category),
	}
}

func (c ColumnCandidates) clone() ColumnCandidates {
	return ColumnCandidates{
		Text:     cloneStrings(c.Text),
		Title:    cloneStrings(c.Title),
		Body:     cloneStrings(c.Body),
		Index:    cloneStrings(c.Index),
		Category: cloneStrings(c.Category),
	}
}

func pickStrings(custom, fallback []string) []string {
	if custom == nil {
		return cloneStrings(fallback)
	}
	return cloneStrings(custom)
}

func cloneStrings(values []string) []string {
	if values == nil {
		return nil
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}
