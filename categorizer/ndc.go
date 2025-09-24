package categorizer

// NDCEntry represents a single entry in the embedded NDC dictionary.
type NDCEntry struct {
	Code  string
	Label string
}

// DefaultNDCEntries returns the minimum viable dictionary based on NDC 10 major classes
// and a few representative sub categories.
func DefaultNDCEntries() []NDCEntry {
	return []NDCEntry{
		{Code: "000", Label: "総記"},
		{Code: "100", Label: "哲学"},
		{Code: "200", Label: "歴史"},
		{Code: "300", Label: "社会科学"},
		{Code: "400", Label: "自然科学"},
		{Code: "500", Label: "技術・工学・工業"},
		{Code: "600", Label: "産業"},
		{Code: "700", Label: "芸術・美術"},
		{Code: "800", Label: "言語"},
		{Code: "900", Label: "文学"},
		// Representative finer grained entries to provide better coverage
		{Code: "007", Label: "情報科学"},
		{Code: "336", Label: "経営"},
		{Code: "657", Label: "会計"},
		{Code: "910", Label: "日本文学"},
		{Code: "913", Label: "日本小説"},
		{Code: "930", Label: "外国文学"},
		{Code: "320", Label: "法律"},
		{Code: "360", Label: "社会問題"},
		{Code: "610", Label: "農業"},
		{Code: "620", Label: "工業"},
		{Code: "830", Label: "英語"},
		{Code: "910.26", Label: "近代文学"},
	}
}
