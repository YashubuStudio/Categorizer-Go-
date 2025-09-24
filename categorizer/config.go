package categorizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const defaultConfigFile = "config.json"

// LoadConfig loads configuration from the given path or the default config.json.
func LoadConfig(path string) (Config, error) {
	if path == "" {
		path = defaultConfigFile
	}
	var cfg Config
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			cfg.UseNDC = true
			cfg.ApplyDefaults()
			return cfg, nil
		}
		return cfg, fmt.Errorf("read config: %w", err)
	}
	hasUseNDC := bytes.Contains(data, []byte("\"useNdc\""))
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("decode config: %w", err)
	}
	if !hasUseNDC {
		cfg.UseNDC = true
	}
	cfg.ApplyDefaults()
	if cfg.Embedder.CacheDir != "" {
		if err := os.MkdirAll(cfg.Embedder.CacheDir, 0o755); err != nil {
			return cfg, fmt.Errorf("create cache dir: %w", err)
		}
	}
	return cfg, nil
}

// SaveConfig persists configuration to disk.
func SaveConfig(path string, cfg Config) error {
	if path == "" {
		path = defaultConfigFile
	}
	tmp := path + ".tmp"
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}
	cfg.ApplyDefaults()
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("encode config: %w", err)
	}
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write temp config: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("rename config: %w", err)
	}
	return nil
}
