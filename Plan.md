# 企画書兼仕様書：ベクトル検索による文章カテゴリ付け支援アプリ

## 1. 背景・目的

* 大量のテキスト（メモ、記事、顧客問い合わせ、議事録など）に対し、素早く一貫性のあるカテゴリ付けを行いたい。
* 事前に用意されたカテゴリ項目（ユーザー追加の「項目」）に近いものを上位3～5件で提案し、人手のラベリングを省力化。
* 事前項目が未指定の場合や、既存項目に合わない文章にも柔軟に対応できるよう、日本十進分類法（NDC）を“第2の軸”として提案可能。
* 同種カテゴリが細かく分散しないよう、結果を**類似カテゴリで束ねる**最適化（任意）を提供。

## 2. ユースケース

1. **単文入力**：1つの文章を入力 → 上位3～5候補を提示。
2. **一括入力**：複数の文章（複数行 or ファイル）を投入 → 全てに対して表形式で提案（元文章／提案1～3）。
3. **項目（カテゴリシード）指定**：アプリ起動時に読み込むカテゴリ群に重みを与え、提案精度向上。
4. **項目未指定モード**：

   * **項目外同一ランキング**：項目が無くても自由提案（項目があれば“やや加点”しつつ、未登録候補も同一リストで競合）。
   * **提案分離**：登録項目ランキングと、NDCベースのランキングを**別枠**で提示。
5. **類似カテゴリの集約（任意）**：同義・近縁のカテゴリをまとめ、実用上のカテゴリ数を抑制。

## 3. 主要機能

* **ベクトル化**：ONNX Runtime（既存コードを呼び出し）で文章・項目を埋め込みベクトルに変換。
* **近傍検索**：コサイン類似度でk件（3～5）を算出。小規模は全探索、大規模はインデックス（後述）で高速化。
* **NDC提案**：NDC（日本十進分類法・10大類＋主な細目）を内蔵辞書として保持。文章→NDC見出しのベクトル類似で候補を提示。
* **一括処理**：複数行入力やCSV/TSV等からの一括分類 → 表出力（GUI表示＋CSV保存）。
* **カテゴリ集約（任意）**：カテゴリベクトル間の類似度に基づくクラスター化（しきい値 or HAC）で、提示カテゴリを縮約。
* **GUI（Fyne）**：テキスト入力、ファイル読込、設定、結果テーブル、ログ・進捗。

## 4. 入出力仕様

### 4.1 項目（カテゴリシード）ファイル

* **形式**：UTF-8 テキスト
* **区切り**：`[,]`（カンマ）または**改行**で1カテゴリ。
* **例**：

  ```
  経営, 会計, マーケティング
  データサイエンス
  自然言語処理
  ```
* **読み込み**：起動時パス指定（コマンドライン or GUI設定）
* **正規化**：`NFKC`、全角/半角統一、前後空白除去

### 4.2 一括入力

* **複数行テキスト**：各行が1件
* **CSV/TSV**：最低1列（テキスト）。列名推定（"text", "本文", "content" など）or 指定。
* **出力**：GUIテーブル＋CSVエクスポート（`text, suggestion1, score1, suggestion2, score2, suggestion3, score3`）

### 4.3 NDC辞書

* **構成**：

  * NDC 10大類（0〜9：総記, 哲学, 歴史…）の**見出し語**
  * 任意で主要細目（例：007 情報科学、910 日本文学…）
* **形式**：埋め込み時に内部でベクトル化・キャッシュ。
* **出力**：NDC提案は `「007:情報科学」` のようにコード＋名称を返す。

## 5. 分類アルゴリズム

### 5.1 ベクトル化

* 既存の **onnxruntime** 呼び出しラッパを利用（`yalue/onnxruntime_go` 等の既存方針に合流）。
* **推奨前処理**：

  * Unicode正規化（NFKC）
  * 句読点・URL・絵文字の扱いはモデル特性に合わせ設定（デフォルトは“保持”）。
* **キャッシュ**：

  * 項目ベクトルは起動時に計算しメモリ常駐＋ファイルキャッシュ（ハッシュキー：NFKC後文字列＋モデル名＋バージョン）。
  * NDC辞書も同様にキャッシュ。

### 5.2 類似度・ランク付け

* **基本**：コサイン類似度。
* **k件**：デフォルト3（設定で3〜5）。
* **重みづけ**：

  * **項目が指定されている場合**：項目候補には +α の**バイアス**（例：+0.02〜+0.05）
  * **「提案分離」モード**：

    * ランキングA：ユーザー項目のみ
    * ランキングB：NDC（項目と別枠）
  * **「項目外同一ランキング」モード**：項目・NDC・その他候補（未登録）が同一土俵で競合し、項目に微加点。
* **スコア安定化**：温度的平滑（Top-k re-ranking で±εノイズ除去）、閾値（例：0.35未満は“提案なし”扱い）。

### 5.3 類似カテゴリの集約（任意）

* **目的**：似たカテゴリ（例：「自然言語処理」「NLP」「言語モデル」）を1つに束ね、運用のカテゴリ数を削減。
* **実装**：

  * **オンライン**：上位候補のラベルベクトル間で**しきい値**（cos ≥ τ）により同一クラスタとみなす（推奨 τ=0.75〜0.85）。
  * **オフライン**：カテゴリ全体を HAC（凝集型）でクラスター化し、代表名を自動提案（初回学習/手動承認制）。
* **UI**：設定でON/OFF、閾値調整スライダ。

### 5.4 インデックス（将来拡張）

* **小規模（\~数千）**：全探索で十分（SIMD最適化／Goルーチン並列）。
* **大規模（\~10万〜）**：HNSW/IVF-PQ等のANN導入を**拡張ポイント**として用意（プラガブルな`VectorIndex`インターフェース）。

## 6. アーキテクチャ

```
+---------------------+        +-----------------+
| Fyne GUI            | <----> | Application     |
| - 入力/結果表示     |        | Service Layer   |
| - 設定/ログ         |        | (UseCases)      |
+---------------------+        +--------+--------+
                                      |
                                      v
                          +-----------+-----------+
                          | Domain / Core        |
                          | - Embedder(IF)       |
                          | - VectorIndex(IF)    |
                          | - Ranker             |
                          | - Clusterer          |
                          | - NDC Dictionary     |
                          +-----------+----------+
                                      |
                                      v
                          +-----------+-----------+
                          | Infrastructure       |
                          | - ONNX Runtime call  |
                          | - Cache(embeddings)  |
                          | - Storage (Bolt/FS)  |
                          +----------------------+
```

### 6.1 主要インターフェース（抜粋・擬似コード）

```go
// 埋め込み器（既存ONNX呼び出しをここに注入）
type Embedder interface {
    EmbedText(ctx context.Context, texts []string) ([][]float32, error)
    Dim() int
    ModelID() string
}

// ベクトル索引（小規模はInMemで全探索）
type VectorIndex interface {
    Upsert(id string, vec []float32, meta map[string]string) error
    Search(vec []float32, k int) ([]Hit, error) // Hit{ID, Score, Meta}
    Size() int
}

// ランカー（重み付け・再ランキング）
type Ranker interface {
    Rank(candidates []Hit, bias map[string]float32, k int) []Hit
}

// クラスタラ（カテゴリ集約）
type Clusterer interface {
    Group(labels []LabelVec, tau float32) [][]LabelVec
}
```

### 6.2 類似度（コサイン）

```go
func Cosine(a, b []float32) float32 {
    var dot, na, nb float32
    for i := range a {
        dot += a[i] * b[i]
        na  += a[i] * a[i]
        nb  += b[i] * b[i]
    }
    if na == 0 || nb == 0 { return 0 }
    return dot / (float32(math.Sqrt(float64(na))) * float32(math.Sqrt(float64(nb))))
}
```

### 6.3 全探索 + 並列

* Goルーチンで `workers = min(GOMAXPROCS, shards)`、SIMD利用はGo標準 + `unsafe`回避で安全運用。

## 7. 設定仕様

### 7.1 起動オプション（CLI）

* `--items <path>`：カテゴリシードファイル（\[,] or 改行区切り）
* `--mode <seeded|mixed|split>`

  * `seeded`：登録項目のみ
  * `mixed`：登録項目に微加点しつつ未登録候補も同一ランキング
  * `split`：登録項目ランキングとNDCランキングを**別枠**で表示
* `--topk <3..5>`：上位件数
* `--cluster <on|off>`、`--cluster-thres <0.75..0.90>`
* `--ndc <on|off>`：NDC提案有効化（`split`時は常に有効）
* `--cache-dir <path>`：埋め込みキャッシュ
* `--model-id <string>`：埋め込みモデル識別子（キャッシュキーに使用）

### 7.2 アプリ設定（GUI）

* 同上を**Settings**ダイアログで操作（保存：`config.json`）。
* モデル変更や閾値は再起動なしで反映（再ベクトル化が必要な場合は進捗バー表示）。

## 8. GUI仕様（Fyne）

### 8.1 画面構成

* **上段**：

  * 入力タブ

    * **単文**：大きめのMultiLineエリア＋「分類」ボタン
    * **一括**：テキスト複数行 or CSV/TSVファイル選択＋「分類」
  * **設定**ボタン（モード、Top-k、NDC、クラスタリング、閾値）
* **中段（結果）**：

  * **単文**：

    * 左：元文章（省略表示／ホバーツールチップで全体）
    * 右：表形式（`提案1/score1, 提案2/score2, 提案3/score3`）
    * `split`モード時は**2つのテーブル**（項目ランキング／NDCランキング）
  * **一括**：

    * 表（`text, 提案1, score1, 提案2, score2, 提案3, score3`）
    * **エクスポート**（CSV）ボタン
* **下段**：ログ／ステータスバー（ベクトル化中、キャッシュヒット率、処理時間）

### 8.2 Fyneコードスケッチ（抜粋）

```go
func buildUI(app fyne.App, svc *Service) fyne.Window {
    w := app.NewWindow("Vector Categorizer")
    input := widget.NewMultiLineEntry()
    input.SetPlaceHolder("ここに文章を入力（複数行可）")

    topk := binding.NewInt()
    topk.Set(3)

    result := widget.NewTable(
        func() (int, int) { return svc.ResultRows(), 7 },
        func() fyne.CanvasObject { return widget.NewLabel("cell") },
        func(id widget.TableCellID, obj fyne.CanvasObject) {
            obj.(*widget.Label).SetText(svc.CellText(id.Row, id.Col))
        },
    )

    classifyBtn := widget.NewButton("分類", func() {
        go func() {
            svc.ClassifyText(input.Text) // 内部で単文/複数行を判定
            result.Refresh()
        }()
    })

    content := container.NewBorder(
        container.NewVBox(input, classifyBtn),
        nil, nil, nil, result,
    )
    w.SetContent(content)
    w.Resize(fyne.NewSize(1000, 700))
    return w
}
```

## 9. 内部データとキャッシュ

* **埋め込みキャッシュ**：

  * `cache-dir`以下に `sha1(NFKC(text) + model-id).bin` で保存（`[]float32`）。
  * LRUでメモリキャッシュ（サイズ上限MB設定可）。
* **設定**：`config.json`（GUI/CLI双方から更新・読み書き）。
* **ログ**：`app.log` ローテーション（サイズ上限）。

## 10. 性能要件（目安）

* 文長〜512トークン程度：1文当たり**数十ms〜数百ms**（CPU/GPUやモデルに依存）。
* 1000文一括：進捗バー表示、逐次テーブル反映（UIスレッド分離）。
* 初回起動時：項目＋NDCの埋め込み計算→キャッシュ化。

## 11. 品質・評価

* **人手評価**：Top-1/Top-3 Accuracy、MRR。
* **運用ログ**：採用されたカテゴリの分布、未採用提案の傾向分析。
* **A/B**：クラスタリングON/OFFでの運用負荷・再分類率比較。

## 12. エラーハンドリング

* モデル読み込み失敗 → ダイアログ＋再試行／モデル変更。
* ベクトル化失敗（空文字・記号のみ） → 「提案なし」。
* CSV列未検出 → 列指定ダイアログ。
* NDC辞書未ロード → NDC提案を一時無効＋通知。

## 13. セキュリティ・プライバシ

* すべてローカル処理（インターネット送信なし）。
* ログに入力本文を保存しない（オプションで匿名要約）。
* キャッシュ削除機能（GUIからワンクリック）。

## 14. 拡張計画（任意）

* **ANNインデックス**バックエンド差し替え（HNSW等）。
* **ユーザー辞書学習**：受理されたラベルを強化学習的に重み更新。
* **用語同義語辞書**：表記ゆれの統合（例：LLM/大規模言語モデル）。
* **NDCの深層細目**展開とハイブリッド提案（上位類→細目ナビ）。

## 15. 例：サービス層フロー（擬似コード）

```go
func (s *Service) ClassifyAll(texts []string) []ResultRow {
    // ベクトル化（キャッシュ利用）
    vecs := s.embedder.EmbedText(ctx, normalizeAll(texts))

    // 検索候補の準備
    seedHits := s.indexSeeds(vecs)         // ユーザー項目
    ndcHits  := s.indexNDC(vecs)           // NDC辞書

    // モード別ランキング
    switch s.cfg.Mode {
    case "seeded":
        rows = rankOnly(seedHits, k)
    case "mixed":
        rows = rankMixed(seedHits, ndcHits, bias=+0.03, k)
    case "split":
        rows = rankSplit(seedHits, ndcHits, k) // 表を2つ or 列分け
    }

    // 類似カテゴリ集約（任意）
    if s.cfg.Cluster.Enabled {
        rows = clusterRows(rows, tau=s.cfg.Cluster.Threshold)
    }
    return rows
}
```

## 16. 例：結果テーブル列定義（デフォルト）

* `text`（先頭100文字表示、ホバーで全体）
* `suggestion1` / `score1`
* `suggestion2` / `score2`
* `suggestion3` / `score3`
* `...`（設定でTop-5まで拡張）

## 17. 開発タスク（実装順）

1. **Embedder**（既存ONNX呼び出しのIF化＋キャッシュ）
2. **InMem VectorIndex**（全探索＋並列）
3. **ランカー**（モード・バイアス・閾値）
4. **NDC辞書**（10大類＋主要細目、埋め込み＆キャッシュ）
5. **Fyne GUI**（単文→一括→CSV出力）
6. **クラスタリング**（オンラインしきい値版）
7. 設定保存／ログ／例外処理
8. ベンチ・評価ログ
9. （任意）ANNバックエンド導入ポイント整備

## 18. テスト観点

* 入力フォーマット差（改行/カンマ混在、空行、重複）。
* 日本語特有：全角/半角、絵文字、機種依存文字。
* 項目数0/極端に多い場合の応答性。
* NDC有効/無効、モード切替、Top-k切替。
* クラスタリングしきい値の安定性。

---

### 付録A：NDC（10大類）サンプル（内蔵辞書の最小セット）

* 0 総記
* 1 哲学
* 2 歴史
* 3 社会科学
* 4 自然科学
* 5 技術・工学・工業
* 6 産業
* 7 芸術・美術
* 8 言語
* 9 文学
  （必要に応じて代表的細目も同梱：例 007 情報科学、336 経営、657 会計、910 日本文学 等）

### 付録B：簡易データ構造（Go）

```go
type Suggestion struct {
    Label string  `json:"label"`
    Score float32 `json:"score"`
    Source string `json:"source"` // "seed", "ndc", "other"
}
type ResultRow struct {
    Text        string        `json:"text"`
    Suggestions []Suggestion  `json:"suggestions"` // 上位k
}
```

---