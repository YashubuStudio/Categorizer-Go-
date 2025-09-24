package main

import (
	"fmt"

	app "yashubustudio/categorizer/internal/app"
)

func main() {
	if err := app.Run(); err != nil {
		fmt.Println("初期化エラー:", err)
		fmt.Println("Config の OrtDLL / ModelPath / TokenizerPath を確認してください。")
	}
}
