package main

import (
	"fmt"

	"fyne.io/fyne/v2/app"
)

// Fyne GUI based categorization assistant entry point.
func main() {
	cfg := defaultConfig()
	ensureDirs(cfg.CacheDir)
	ensureSeedFile(cfg.SeedFile, defaultUserCategories)

	svc, err := NewService(cfg)
	if err != nil {
		fmt.Println("初期化エラー:", err)
		fmt.Println("Config の OrtDLL / ModelPath / TokenizerPath を確認してください。")
		return
	}
	defer svc.Close()

	a := app.NewWithID(fyneAppID)
	u := buildUI(a, svc)
	u.w.ShowAndRun()
}
