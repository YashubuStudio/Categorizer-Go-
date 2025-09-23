package emb

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

type Encoder struct {
	sess       *ort.DynamicAdvancedSession
	opts       *ort.SessionOptions
	tok        *tokenizer.Tokenizer
	inputNames []string
	outputName string // "last_hidden_state" を想定
	hidden     int    // 例: 1024
	maxLen     int
	mu         sync.Mutex // ORTセッションは基本スレッドセーフだが、簡易に直列化
}

type Config struct {
	// 固定パス（あなたの環境）
	OrtDLL        string // 例: D:\Ollama\projects\csv-search\onnixruntime-win\lib\onnxruntime.dll
	ModelPath     string // 例: D:\Ollama\projects\csv-search\models\bge-m3\model.onnx  (必要なら _data も同階層)
	TokenizerPath string // 例: D:\Ollama\projects\csv-search\models\bge-m3\tokenizer.json
	MaxSeqLen     int    // 例: 512
}

// Init: ORT/DLL読み込み→環境初期化→モデル/トークナイザ読み込み→セッション生成
func (e *Encoder) Init(cfg Config) error {
	if cfg.OrtDLL == "" || cfg.ModelPath == "" || cfg.TokenizerPath == "" {
		return errors.New("OrtDLL/ModelPath/TokenizerPath は必須です")
	}
	if _, err := os.Stat(cfg.OrtDLL); err != nil {
		return fmt.Errorf("onnxruntime.dll が見つかりません: %s", cfg.OrtDLL)
	}
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		return fmt.Errorf("model.onnx が見つかりません: %s", cfg.ModelPath)
	}
	if _, err := os.Stat(cfg.TokenizerPath); err != nil {
		return fmt.Errorf("tokenizer.json が見つかりません: %s", cfg.TokenizerPath)
	}

	// ORT DLL を明示ロード → 環境初期化
	ort.SetSharedLibraryPath(cfg.OrtDLL)
	if err := ort.InitializeEnvironment(ort.WithLogLevelWarning()); err != nil {
		return err
	}

	// モデルIOを確認
	inInfos, outInfos, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return err
	}
	// 入力名（input_ids / attention_mask を想定）
	e.inputNames = nil
	hasInputIDs, hasMask := false, false
	for _, ii := range inInfos {
		switch ii.Name {
		case "input_ids":
			hasInputIDs = true
			e.inputNames = append(e.inputNames, ii.Name)
		case "attention_mask":
			hasMask = true
			e.inputNames = append(e.inputNames, ii.Name)
		}
	}
	if !hasInputIDs {
		return fmt.Errorf("モデルに input_ids がありません（実IO: %+v）", inInfos)
	}
	if !hasMask {
		// attention_mask が無いモデルも存在するが、bge-m3 は通常あり
		// ここはエラーにせず自動生成も可。今回は警告扱いで対応。
		e.inputNames = []string{"input_ids"} // 最低限
	}

	// 出力名と hidden 次元を推定（"last_hidden_state": [-1 -1 hidden] を期待）
	e.outputName = ""
	e.hidden = 0
	for _, oi := range outInfos {
		if oi.Name == "last_hidden_state" {
			e.outputName = oi.Name
			dims, err := parseDimsFromShapeString(oi.String())
			if err == nil && len(dims) >= 3 {
				hidden := dims[len(dims)-1]
				if hidden > 0 {
					e.hidden = int(hidden)
				}
			}
			break
		}
	}
	if e.outputName == "" {
		return fmt.Errorf("last_hidden_state が出力に見つかりません（実IO: %+v）", outInfos)
	}
	if e.hidden == 0 {
		// 取得に失敗した場合は既定値（bge-m3は1024）
		e.hidden = 1024
	}

	// トークナイザ
	tk, err := pretrained.FromFile(cfg.TokenizerPath)
	if err != nil {
		return err
	}
	e.tok = tk

	// セッション作成
	e.opts, err = ort.NewSessionOptions()
	if err != nil {
		return err
	}
	e.sess, err = ort.NewDynamicAdvancedSession(cfg.ModelPath, e.inputNames, []string{e.outputName}, e.opts)
	if err != nil {
		return err
	}

	if cfg.MaxSeqLen <= 0 {
		cfg.MaxSeqLen = 512
	}
	e.maxLen = cfg.MaxSeqLen
	return nil
}

// Close: ORTリソースの後片付け
func (e *Encoder) Close() {
	if e.sess != nil {
		e.sess.Destroy()
		e.sess = nil
	}
	if e.opts != nil {
		e.opts.Destroy()
		e.opts = nil
	}
	// ORT環境終了
	_ = ort.DestroyEnvironment()
}

// Encode: 日本語テキスト → 句ベクトル（L2正規化済み）
// 返り値は長さ e.hidden の []float32
func (e *Encoder) Encode(text string) ([]float32, error) {
	if e.sess == nil || e.tok == nil {
		return nil, errors.New("encoder is not initialized")
	}

	// ===== トークナイズ（最大長でトリム、attentionを自動生成）=====
	if runtime.GOOS == "windows" {
		text = strings.TrimSpace(text)
	}
	enc, err := e.tok.EncodeSingle(text)
	if err != nil {
		return nil, err
	}
	ids := make([]int64, 0, len(enc.Ids))
	mask := make([]int64, 0, len(enc.Ids))
	for i, v := range enc.Ids {
		if len(ids) >= e.maxLen {
			break
		}
		ids = append(ids, int64(v))
		if len(enc.AttentionMask) > i {
			mask = append(mask, int64(enc.AttentionMask[i]))
		} else {
			mask = append(mask, 1)
		}
	}
	seqLen := int64(len(ids))
	if seqLen == 0 {
		return nil, errors.New("empty tokenized input")
	}

	// ===== 入力テンソル =====
	shape := ort.NewShape(1, seqLen)
	tIDs, err := ort.NewTensor[int64](shape, ids)
	if err != nil {
		return nil, err
	}
	defer tIDs.Destroy()

	inputs := []ort.Value{tIDs}
	if len(e.inputNames) == 2 { // attention_mask がある場合のみ
		tMask, err := ort.NewTensor[int64](shape, mask)
		if err != nil {
			return nil, err
		}
		defer tMask.Destroy()
		inputs = append(inputs, tMask)
	}

	// ===== 出力テンソル（[1, seqLen, hidden]）=====
	outShape := ort.NewShape(1, seqLen, int64(e.hidden))
	tOut, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, err
	}
	defer tOut.Destroy()

	// 実行（直列化）
	e.mu.Lock()
	err = e.sess.Run(inputs, []ort.Value{tOut})
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}

	// ===== Mean Pooling + L2 =====
	raw := tOut.GetData() // len = seqLen * hidden
	if len(raw) != int(seqLen)*e.hidden {
		// モデル側でpad/切詰めされた可能性を考慮（保険）
		if len(raw)%e.hidden != 0 {
			return nil, fmt.Errorf("unexpected output length: %d", len(raw))
		}
		seqLen = int64(len(raw) / e.hidden)
	}
	vec := meanPoolAndL2(raw, int(seqLen), e.hidden, mask)
	return vec, nil
}

// ===== ヘルパ =====

func meanPoolAndL2(lastHidden []float32, seqLen, hidden int, attn []int64) []float32 {
	out := make([]float32, hidden)
	var valid float32
	for t := 0; t < seqLen; t++ {
		if attn == nil || attn[t] != 0 {
			valid++
			base := t * hidden
			for h := 0; h < hidden; h++ {
				out[h] += lastHidden[base+h]
			}
		}
	}
	if valid > 0 {
		inv := 1.0 / valid
		for h := 0; h < hidden; h++ {
			out[h] *= float32(inv)
		}
	}
	// L2 normalize
	var s float64
	for _, v := range out {
		s += float64(v) * float64(v)
	}
	n := float32(math.Sqrt(s) + 1e-12)
	for i := range out {
		out[i] /= n
	}
	return out
}

func parseDimsFromShapeString(s string) ([]int64, error) {
	start := strings.Index(s, "[")
	end := strings.Index(s, "]")
	if start < 0 || end < 0 || end <= start {
		return nil, errors.New("shape string parse error")
	}
	inner := s[start+1 : end]
	fields := strings.Fields(inner)
	out := make([]int64, 0, len(fields))
	for _, f := range fields {
		var v int64
		if _, err := fmt.Sscan(f, &v); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// 実行ファイルのディレクトリ（使いたければどうぞ）
func exeDir() string {
	p, _ := os.Executable()
	return filepath.Dir(p)
}
