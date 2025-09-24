package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var defaultUserCategories = []string{
	"CG・デジタルアーカイブ",
	"VR空間",
	"アバター",
	"インタラクション",
	"エージェント",
	"コミュニケーション",
	"ソーシャルVR",
	"可視化",
	"工学・サイエンスコミュニケーション",
	"応用数理",
	"感覚・知覚",
	"教育",
	"機械学習",
	"社会",
}

func initialUserCategories(seedFile string) ([]string, bool, error) {
	fallback := uniqueNormalized(defaultUserCategories)
	path := strings.TrimSpace(seedFile)
	if path == "" {
		return fallback, false, nil
	}
	cats, err := loadCategorySeedFile(path)
	if err != nil {
		return fallback, false, err
	}
	return cats, true, nil
}

func ensureDirs(p string) {
	if p == "" {
		return
	}
	_ = os.MkdirAll(filepath.Clean(p), 0o755)
}

func ensureSeedFile(path string, seeds []string) {
	clean := strings.TrimSpace(path)
	if clean == "" {
		return
	}
	clean = filepath.Clean(clean)
	if _, err := os.Stat(clean); err == nil {
		return
	} else if !errors.Is(err, os.ErrNotExist) {
		fmt.Println("カテゴリファイル確認エラー:", err)
		return
	}
	dir := filepath.Dir(clean)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fmt.Println("カテゴリファイルディレクトリ作成エラー:", err)
			return
		}
	}
	content := strings.Join(seeds, "\n")
	if err := os.WriteFile(clean, []byte(content+"\n"), 0o644); err != nil {
		fmt.Println("カテゴリファイル作成エラー:", err)
	}
}

func loadCategorySeedFile(path string) ([]string, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	labels := parseCategoryText(string(data))
	cats := uniqueNormalized(labels)
	if len(cats) == 0 {
		return nil, fmt.Errorf("カテゴリが見つかりません (%s)", filepath.Clean(path))
	}
	return cats, nil
}

func parseCategoryText(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool {
		switch r {
		case '\n', '\r', ',', ';', '\t':
			return true
		default:
			return false
		}
	})
	res := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f != "" {
			res = append(res, f)
		}
	}
	return res
}
