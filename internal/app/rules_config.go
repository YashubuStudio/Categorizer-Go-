package app

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ensureCategoryRuleFile writes the default keyword rules to the given path
// when the file does not exist yet. This gives users a starting point for
// editing rule weights outside of the binary.
func ensureCategoryRuleFile(path string, defaults map[string]keywordRuleSet) {
	clean := strings.TrimSpace(path)
	if clean == "" {
		return
	}
	clean = filepath.Clean(clean)
	if _, err := os.Stat(clean); err == nil {
		return
	} else if !errors.Is(err, os.ErrNotExist) {
		fmt.Println("カテゴリルールファイル確認エラー:", err)
		return
	}

	dir := filepath.Dir(clean)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fmt.Println("カテゴリルールファイルディレクトリ作成エラー:", err)
			return
		}
	}

	data, err := json.MarshalIndent(defaults, "", "  ")
	if err != nil {
		fmt.Println("カテゴリルールファイル変換エラー:", err)
		return
	}
	if err := os.WriteFile(clean, append(data, '\n'), 0o644); err != nil {
		fmt.Println("カテゴリルールファイル作成エラー:", err)
	}
}

// loadCompiledCategoryRules returns the compiled keyword rules. When the path
// is empty or loading fails, the defaults defined in hybrid.go are returned.
// The boolean indicates whether a custom file was successfully loaded.
func loadCompiledCategoryRules(path string) (map[string]compiledRuleSet, bool, error) {
	defaults := defaultCompiledCategoryRules

	clean := strings.TrimSpace(path)
	if clean == "" {
		return defaults, false, nil
	}

	data, err := os.ReadFile(filepath.Clean(clean))
	if err != nil {
		return defaults, false, err
	}

	overrides := make(map[string]keywordRuleSet)
	if err := json.Unmarshal(data, &overrides); err != nil {
		return defaults, false, err
	}

	merged := mergeKeywordRuleSets(rawCategoryRules, overrides)
	compiled := compileCategoryRules(merged)
	return compiled, true, nil
}

func mergeKeywordRuleSets(base, overrides map[string]keywordRuleSet) map[string]keywordRuleSet {
	if len(overrides) == 0 {
		return cloneKeywordRuleSetMap(base)
	}
	merged := cloneKeywordRuleSetMap(base)
	for label, set := range overrides {
		merged[label] = cloneKeywordRuleSet(set)
	}
	return merged
}

func cloneKeywordRuleSetMap(src map[string]keywordRuleSet) map[string]keywordRuleSet {
	if len(src) == 0 {
		return map[string]keywordRuleSet{}
	}
	dst := make(map[string]keywordRuleSet, len(src))
	for label, set := range src {
		dst[label] = cloneKeywordRuleSet(set)
	}
	return dst
}

func cloneKeywordRuleSet(set keywordRuleSet) keywordRuleSet {
	res := keywordRuleSet{}
	if len(set.Strong) > 0 {
		res.Strong = append([]string(nil), set.Strong...)
	}
	if len(set.Weak) > 0 {
		res.Weak = append([]string(nil), set.Weak...)
	}
	if len(set.Anti) > 0 {
		res.Anti = append([]string(nil), set.Anti...)
	}
	return res
}
