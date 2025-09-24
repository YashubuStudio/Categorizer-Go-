package categorizer

import "encoding/json"

// Mode represents the ranking mode for suggestions.
type Mode string

const (
	// ModeSeeded ranks only user provided seed categories.
	ModeSeeded Mode = "seeded"
	// ModeMixed ranks both seed categories and the NDC dictionary in a single list.
	ModeMixed Mode = "mixed"
	// ModeSplit keeps user seeds and NDC suggestions in separate lists.
	ModeSplit Mode = "split"
)

// Suggestion represents an individual category suggestion.
type Suggestion struct {
	Label  string  `json:"label"`
	Score  float32 `json:"score"`
	Source string  `json:"source"`
}

// ResultRow holds the suggestions for a single input text.
type ResultRow struct {
	Text           string       `json:"text"`
	Suggestions    []Suggestion `json:"suggestions"`
	NDCSuggestions []Suggestion `json:"ndcSuggestions,omitempty"`
}

// InputRecord represents a text sample optionally accompanied by metadata.
type InputRecord struct {
	Index string `json:"index,omitempty"`
	Title string `json:"title,omitempty"`
	Body  string `json:"body,omitempty"`
	Text  string `json:"text"`
}

// ClusterConfig controls optional clustering of similar categories.
type ClusterConfig struct {
	Enabled   bool    `json:"enabled"`
	Threshold float32 `json:"threshold"`
}

// EmbedderConfig wraps the configuration for the ORT embedder and cache.
type EmbedderConfig struct {
	OrtDLL        string `json:"ortDll"`
	ModelPath     string `json:"modelPath"`
	TokenizerPath string `json:"tokenizerPath"`
	MaxSeqLen     int    `json:"maxSeqLen"`
	CacheDir      string `json:"cacheDir"`
	ModelID       string `json:"modelId"`
}

// Config aggregates runtime settings persisted to config.json.
type Config struct {
	Mode      Mode           `json:"mode"`
	TopK      int            `json:"topK"`
	SeedBias  float32        `json:"seedBias"`
	MinScore  float32        `json:"minScore"`
	Cluster   ClusterConfig  `json:"cluster"`
	Embedder  EmbedderConfig `json:"embedder"`
	SeedsPath string         `json:"seedsPath"`
	UseNDC    bool           `json:"useNdc"`
}

// Clone creates a deep copy of the configuration so callers can mutate safely.
func (c Config) Clone() Config {
	buf, _ := json.Marshal(c)
	var out Config
	_ = json.Unmarshal(buf, &out)
	return out
}

// ApplyDefaults populates zero values with sensible defaults.
func (c *Config) ApplyDefaults() {
	if c.Mode == "" {
		c.Mode = ModeMixed
	}
	if c.TopK <= 0 {
		c.TopK = 3
	}
	if c.SeedBias == 0 {
		c.SeedBias = 0.03
	}
	if c.MinScore == 0 {
		c.MinScore = 0.35
	}
	if c.Cluster.Threshold == 0 {
		c.Cluster.Threshold = 0.8
	}
	if c.Embedder.MaxSeqLen == 0 {
		c.Embedder.MaxSeqLen = 512
	}
}
