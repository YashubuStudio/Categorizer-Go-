# AI実装指示（詳細設計思考）

## 0. ゴール定義

* 入力：1件または複数件の日本語テキスト（発表概要など）
* 候補集合：ユーザー定義カテゴリ（必須）＋ NDC（任意ON/OFF、ただし常に候補集合は**固定集合**）
* 出力：各テキストに対し **Top-k=3**（設定で最大5）を返す。各候補に\*\*スコア（0–1）\*\*を付与。
* 追加出力：信頼度判定（「要確認」フラグ）
* 禁止：カテゴリ外（未知ラベル）の生成・提案

---

## 1. 前処理（Normalization）

1. 入力テキスト・カテゴリ名・NDC見出し語すべてに以下を施す：

   * Unicode **NFKC** 正規化
   * 前後空白削除（`strings.TrimSpace`）
   * 連続空白の単一化（正規表現で `\s+` → 半角スペース1つ）
2. 絵文字・URL・記号は**モデル既定**。bge-m3等の一般的日本語埋め込みは、そのまま入力で良い。
3. 文字数が極端に短い（例：<5文字）場合は低スコアになりやすい。**短文はそのまま処理**しつつ、後段の「要確認」判定が自然に働くようにする。

---

## 2. 埋め込み（Embedding）

1. 既存の ONNX Runtime 呼び出し（bge-m3 等）を `Embedder` 経由で使用。
2. **キャッシュ方針**：

   * キー：`sha1(NFKC(text) + modelID)`
   * 値：`[]float32` のベクトル（次元 `Dim()` はモデルに依存）
   * ストレージ：`cache-dir` にバイナリ保存＋起動時LRUメモリキャッシュ
3. 起動時：

   * ユーザーカテゴリ全件を埋め込み→常駐
   * NDC（有効時）も同様
4. 推奨：埋め込みベクトルは**L2正規化**して保持（後段のコサイン計算を内積で代替しやすい）。

---

## 3. 候補集合の構成（Candidate Set）

* `C_user`: ユーザー定義カテゴリのラベル集合（例：「VR空間」「感覚・知覚」…）
* `C_ndc` : NDCのラベル集合（少なくとも10大類。必要なら主要細目も追加）
* 実行モード：

  * `seeded`：`C = C_user` のみ
  * `mixed`（既定）：`C = C_user ∪ C_ndc`（**ただし重みでカテゴリ優位**）
  * `split`：表示はカテゴリランキングとNDCランキングを**別枠**で並列提示（ただしスコア計算は共通式）
* 本件要件：**カテゴリ外提案なし**のため、`C` は上記いずれでも**固定**。未知ラベルは生成禁止。

---

## 4. 類似度とスコアリング（Ranking）

1. **基本スコア**：コサイン類似度

   * 入力ベクトル `q` と候補ベクトル `v_i` のコサイン `cos(q, v_i)` を `[0,1]` に写像（負値は0扱いでもよい）
   * 実装：正規化済みなら `dot(q, v_i)` で代替可能
2. **ソース重み**（カテゴリ優先）：

   * `w_user = 1.00`
   * `w_ndc  = 0.85`
   * **最終スコア**：`score_i = clamp01( cos(q, v_i) * w_source(i) )`
3. **同点解消 bias**（ごく微小）：

   * 同じスコアが並んだ場合に**安定順序**を与える。
   * 例：`score_i += 1e-6 * StableHash(label_i)`（浮動小数の比較誤差に注意）
4. **Top-k**：既定3、最大5（設定）。
5. **低スコアの扱い**：**候補は返すが**、後述の「要確認」フラグで人手判断を促す。

---

## 5. 信頼度判定（「要確認」フラグ）

以下の**いずれか**を満たす場合に `need_review = true` とする（UIで「要確認」表示）。

* `top1 < 0.45`
* `top1 - top2 < 0.03`（1位と2位が接近）
* `mean(top1..topk) < 0.50`（既定 `k=3`）
  備考：数値は初期値。運用で適宜調整（設定から変更可）。

---

## 6. 類似カテゴリの集約（任意機能）

* 目的：似通ったラベル（表記ゆれ・近縁）を束ね、運用カテゴリ数を減らす。
* オンライン集約（ライト版）：

  * Top-k候補の **ラベルベクトル間コサイン ≥ τ**（初期 `τ = 0.80`）を**同一グループ**として表示上まとめる。
  * 代表ラベル選択：グループ内で最スコアのラベル名を代表に。
* オフライン集約（任意拡張）：

  * カテゴリ全集合に対して HAC（凝集型クラスタリング）で木構造を作り、手動でカットレベル決定→代表名登録。
* UI：設定でON/OFF、`τ` をスライダで調整。

---

## 7. バッチ処理（複数テキスト）

1. 入力は「複数行テキスト」またはCSV/TSV。
2. 行ごと（レコードごと）に 2–6 を繰り返す。
3. 逐次UI更新：Fyneでは分類スレッドとUIスレッドを分離し、行ごとにテーブルを更新。
4. 出力テーブル列：

   * `text`（先頭100文字まで表示。ホバーで全体）
   * `label1 (score1)` / `label2 (score2)` / `label3 (score3)`
   * `need_review`（Yes/No）
   * `source1/2/3`（`user` or `ndc` を薄文字で併記してもよい）

---

## 8. 例外とエッジケース

* 入力が空／記号のみ：スコア0で「要確認」
* 極短文（例：「VR酔い」1語）：スコアが不安定になりやすい → 自然に「要確認」条件にかかる
* 同義ラベル重複（例：「ソーシャルVR」「コミュニケーション」など）

  * 集約ON時はまとまる
  * OFF時でも両方がTop-kに来る可能性あり（仕様上許容）
* NDC無効：`C = C_user` のみで処理
* モデル切替：キャッシュキーに `modelID` を含め、誤利用を防止

---

## 9. UI（Fyne）指示

* 入力タブ：

  * 単発テキスト（MultiLine）＋「分類」ボタン
  * 一括：複数行 or CSV/TSV 読込
* 設定：

  * `Top-k`（3–5）、`NDC ON/OFF`、`w_ndc`（0.70–1.00範囲のスライダ）、
  * 「要確認」しきい値（`top1`, `top1-top2`, `mean`）
  * 集約ON/OFFと `τ`
* 結果テーブル：

  * 行クリックで詳細（上位5まで、NDC別枠（splitモード時））
  * CSVエクスポート（UTF-8、ヘッダ：`text,label1,score1,...,need_review`）
* ステータスバー：処理件数／経過時間／キャッシュヒット率

---

## 10. ロギング＆評価

* ログ（デフォルトINFO）：処理時間（埋め込み・ランキング分解）、Top-1スコア分布、要確認件数
* 運用評価：

  * Top-1 / Top-3 Accuracy（人手確定ラベルと比較）
  * 要確認率（低すぎる＝閾値が緩い／高すぎる＝厳しすぎ）
* 設定チューニングの推奨順：

  1. `w_ndc`（カテゴリ優位性の強弱）
  2. `top1` 閾値
  3. `top1-top2` 閾値
  4. `mean(topk)` 閾値
  5. 集約 `τ`

---

## 11. 擬似コード（中核）

```go
// 前提：カテゴリ集合（C_user, C_ndc）とそれぞれの埋め込みベクトルは起動時に用意済み。
func RankOne(text string, cfg Config) (top []ScoredLabel, needReview bool) {
    t := Normalize(text)
    q := EmbedCached(t)       // []float32 (L2 normalized)

    // 1) 候補集合の選定
    candidates := C_user
    if cfg.NDCEnabled && cfg.Mode != "seeded" {
        candidates = append(candidates, C_ndc...)
    }

    // 2) スコア計算（コサイン×ソース重み）
    scored := make([]ScoredLabel, 0, len(candidates))
    for _, c := range candidates {
        cos := CosineDot(q, c.Vec) // 正規化済みなら内積でOK
        if cos < 0 { cos = 0 }
        w := 1.0
        if c.Source == "ndc" { w = cfg.WeightNDC } // 既定 0.85
        s := clamp01(cos * w)
        s += tinyBias(c.Label) // 同点解消
        scored = append(scored, ScoredLabel{Label: c.Label, Score: s, Source: c.Source})
    }

    // 3) ソート＆Top-k
    sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
    k := clampK(cfg.TopK) // 3..5
    top = scored[:min(len(scored), k)]

    // 4) 集約（任意）
    if cfg.Cluster.Enabled {
        top = AggregateByCosine(top, cfg.Cluster.Threshold) // τ=0.80 付近
    }

    // 5) 信頼度判定
    top1 := top[0].Score
    top2 := getScore(top, 1)
    mean := meanScore(top)
    needReview = (top1 < cfg.Thresh.Top1) ||
                 (top1-top2 < cfg.Thresh.Margin12) ||
                 (mean < cfg.Thresh.Mean)
    return
}
```

---

## 12. テスト観点（AI挙動）

* **明瞭入力**（今回のVR酔い例）：`VR空間 > 感覚・知覚 > ソーシャルVR/コミュニケーション` が安定してTop-3
* **曖昧入力**：Top1とTop2が接近 → 「要確認」点灯
* **短文**：「VR酔い」単語のみ → Top1低下かつ平均低下 → 「要確認」
* **似たラベル群**（「ソーシャルVR」「コミュニケーション」）：集約ONで統合、OFFで別々
* **NDC OFF**：カテゴリのみで安定動作
* **モデル変更**：キャッシュ衝突なし（modelID含有）

---

## 13. 初期パラメータ（実装値）

* `TopK = 3`
* `WeightNDC = 0.85`
* `Thresh.Top1 = 0.45`
* `Thresh.Margin12 = 0.03`
* `Thresh.Mean = 0.50`
* `Cluster.Enabled = false`（初期OFF）
* `Cluster.Threshold = 0.80`

---

## 14. 運用チューニング手順

1. 数十〜数百件で **Top-1/Top-3** と **要確認率** を評価
2. 想定より「要確認」が多い → `Thresh.*` を緩和 or `WeightNDC` を下げる（カテゴリ更に優位）
3. ラベル分散が多い → 集約ON、`τ` を 0.80→0.78 に微調整
4. スコアが全体に低い → カテゴリ語彙の改良（単語→短いフレーズ化）を検討

---
