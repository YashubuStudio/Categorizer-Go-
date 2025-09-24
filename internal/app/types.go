package app

type Candidate struct {
	Label  string
	Vec    []float32
	Source string // "seed" or "ndc"
}

type Suggestion struct {
	Label   string
	Score   float32
	Source  string
	Aliases []string
}

type ResultRow struct {
	Text            string
	Suggestions     []Suggestion
	SeedSuggestions []Suggestion
	NDCSuggestions  []Suggestion
	NeedReview      bool
}
