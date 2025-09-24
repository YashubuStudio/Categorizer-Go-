package main

import "strings"

const (
	ModeSeeded = "seeded"
	ModeMixed  = "mixed"
	ModeSplit  = "split"

	fyneAppID       = "studio.yashubu.categorizer"
	defaultSeedFile = "config/categories_seed.txt"
)

var modeChoices = []struct {
	Label string
	Value string
}{
	{Label: "項目のみ", Value: ModeSeeded},
	{Label: "混合 (項目+NDC)", Value: ModeMixed},
	{Label: "別枠（項目/NDC）", Value: ModeSplit},
}

type Threshold struct {
	Top1     float32 // 例: 0.45
	Margin12 float32 // 例: 0.03
	Mean     float32 // 例: 0.50
}

type ClusterCfg struct {
	Enabled   bool
	Threshold float32 // tau 例: 0.80
}

type Config struct {
	TopK      int
	Mode      string
	UseNDC    bool
	WeightNDC float32
	SeedBias  float32
	Thresh    Threshold

	ClusterCfg ClusterCfg

	OrtDLL        string
	ModelPath     string
	TokenizerPath string
	MaxSeqLen     int

	CacheDir string
	SeedFile string
}

func defaultConfig() Config {
	return Config{
		TopK:          3,
		Mode:          ModeMixed,
		UseNDC:        true,
		WeightNDC:     0.85,
		SeedBias:      0.03,
		Thresh:        Threshold{Top1: 0.45, Margin12: 0.03, Mean: 0.50},
		ClusterCfg:    ClusterCfg{Enabled: false, Threshold: 0.80},
		OrtDLL:        "./onnixruntime-win/lib/onnxruntime.dll",
		ModelPath:     "./models/bge-m3/model.onnx",
		TokenizerPath: "./models/bge-m3/tokenizer.json",
		MaxSeqLen:     512,
		CacheDir:      "./cache",
		SeedFile:      defaultSeedFile,
	}
}

func sanitizeConfig(cfg Config) Config {
	if cfg.TopK < 3 {
		cfg.TopK = 3
	}
	if cfg.TopK > 5 {
		cfg.TopK = 5
	}
	switch cfg.Mode {
	case ModeSeeded, ModeMixed, ModeSplit:
	default:
		cfg.Mode = ModeMixed
	}
	if cfg.WeightNDC < 0.5 {
		cfg.WeightNDC = 0.5
	}
	if cfg.WeightNDC > 1.2 {
		cfg.WeightNDC = 1.2
	}
	if cfg.SeedBias < 0 {
		cfg.SeedBias = 0
	}
	if cfg.SeedBias > 0.2 {
		cfg.SeedBias = 0.2
	}
	if cfg.ClusterCfg.Threshold <= 0 {
		cfg.ClusterCfg.Threshold = 0.80
	}
	if cfg.Thresh.Top1 <= 0 {
		cfg.Thresh.Top1 = 0.45
	}
	if cfg.Thresh.Margin12 < 0 {
		cfg.Thresh.Margin12 = 0.03
	}
	if cfg.Thresh.Mean <= 0 {
		cfg.Thresh.Mean = 0.50
	}
	cfg.SeedFile = strings.TrimSpace(cfg.SeedFile)
	return cfg
}
