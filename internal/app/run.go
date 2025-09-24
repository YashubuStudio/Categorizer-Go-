package app

import (
	fyneapp "fyne.io/fyne/v2/app"
)

// Run initializes required resources and starts the desktop UI.
func Run() error {
	cfg := defaultConfig()
	ensureDirs(cfg.CacheDir)
	ensureSeedFile(cfg.SeedFile, defaultUserCategories)
	ensureCategoryRuleFile(cfg.CategoryRuleFile, rawCategoryRules)

	svc, err := NewService(cfg)
	if err != nil {
		return err
	}
	defer svc.Close()

	a := fyneapp.NewWithID(fyneAppID)
	u := buildUI(a, svc)
	u.w.ShowAndRun()
	return nil
}
