package app

// legacy main for manual encoder testing. Retained but disabled so the primary
// application entry point in cmd/categorizer can define its own main function.

/*
	func main() {
		fmt.Println("このバイナリはGUIアプリの開発用に残されたONNX Runtimeサンプルです。")
		fmt.Println("環境変数 RUN_ONNX_EXAMPLE=1 を指定すると旧来のデモを実行します。")
		if os.Getenv("RUN_ONNX_EXAMPLE") != "1" {
			return
		}
		enc := &emb.Encoder{}
		cfg := emb.Config{
			OrtDLL:        `D:\\Ollama\\projects\\csv-search\\onnixruntime-win\\lib\\onnxruntime.dll`,
			ModelPath:     `D:\\Ollama\\projects\\csv-search\\models\\bge-m3\\model.onnx`,
			TokenizerPath: `D:\\Ollama\\projects\\csv-search\\models\\bge-m3\\tokenizer.json`,
			MaxSeqLen:     512,
		}
		if err := enc.Init(cfg); err != nil {
			log.Fatal(err)
		}
		defer enc.Close()

		// 動作確認：2文の類似度
		a := "渋谷で静かなWi-Fiカフェを探しています。"
		b := "ノートPC作業に向いた落ち着いた喫茶店を知りたい。"

		va, err := enc.Encode(a)
		if err != nil {
			log.Fatal(err)
		}
		vb, err := enc.Encode(b)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("dim=%d\n", len(va))
		fmt.Printf("cosine(a,b)=%.3f\n", cosine(va, vb))
	}
*/
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		fa, fb := float64(a[i]), float64(b[i])
		dot += fa * fb
		na += fa * fa
		nb += fb * fb
	}
	if na == 0 || nb == 0 {
		return 0
	}
	// 標準ライブラリでOK（独自sqrt不要）
	return dot / (float64Sqrt(na) * float64Sqrt(nb))
}

func float64Sqrt(x float64) float64 {
	// math.Sqrt を直接使ってもOK。外部依存を増やしたくなければ簡易実装も可。
	// ここでは math を使う版:
	//   return math.Sqrt(x)
	// math を使わない簡易ニュートン法:
	z := 1.0
	for i := 0; i < 12; i++ {
		z = 0.5 * (z + x/z)
	}
	return z
}
