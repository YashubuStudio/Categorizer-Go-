# Categorizer (ベクトル検索支援)

## 概要
Categorizer は、ユーザーが指定したカテゴリシードと日本十進分類法 (NDC) 辞書を用いて、日本語テキストに最適なカテゴリ候補を提示するデスクトップ向け支援ツールです。Fyne 製の GUI から複数文章をまとめて推論し、シードの有無に応じたランキングや NDC からのサジェストを同時に確認できます。推論に必要な文埋め込みは ONNX Runtime ベースのエンコーダーで生成し、結果は GUI 上のテーブル表示と CSV 出力の両方に対応しています。【F:main.go†L1-L343】【F:categorizer/service.go†L1-L222】

## 主な機能
- **シードカテゴリの管理**: テキスト入力欄またはファイルからカテゴリシードを読み込み、重複排除と正規化を行った上でベクトル化して保持します。【F:main.go†L50-L121】【F:categorizer/service.go†L82-L120】
- **NDC 辞書連携**: 組み込みの NDC 辞書を自動ベクトル化し、ユーザー指定のモードに応じてランキングへ加味します。【F:categorizer/service.go†L39-L108】【F:categorizer/ndc.go†L1-L26】
- **一括分類**: 複数行テキストや CSV/TSV から抽出した文章をまとめて推論し、スコア付きの候補リストを返します。【F:main.go†L174-L239】【F:categorizer/io.go†L34-L109】
- **結果の表示・保存**: 推論結果をテーブルで可視化し、詳細ダイアログや CSV エクスポートで共有できます。【F:main.go†L136-L239】
- **柔軟な設定**: ランキングモード、Top-K（最大5件）、NDC重み、クラスタリングの閾値、NDC 使用可否などを GUI から即時反映し、自動で `config.json` に保存します。【F:main.go†L639-L773】【F:categorizer/config.go†L12-L69】

## 動作環境
- Go 1.24 以降。【F:go.mod†L1-L4】
- CGO を有効にした ONNX Runtime ランタイム (動的ライブラリ) と日本語対応の ONNX モデル、および対応するトークナイザー。【F:emb/emb.go†L19-L130】
- Fyne を含む Go モジュール依存関係は `go mod tidy` 実行時に自動取得されます。【F:main.go†L14-L23】

## セットアップ
1. リポジトリを取得し、依存関係をインストールします。
   ```bash
   git clone <your fork or clone URL>
   cd Categorizer-Go-
   go mod tidy
   ```
2. ONNX Runtime の動的ライブラリと使用する埋め込みモデル (例: bge-m3) をダウンロードし、パスを控えておきます。【F:emb/emb.go†L33-L131】
3. アプリと同じディレクトリに `config.json` を作成し、下記のように埋め込み設定や初期値を記述します。未指定の値は起動時に自動補完されます。【F:categorizer/config.go†L12-L69】【F:categorizer/types.go†L60-L99】

### config.json の例
```json
{
  "mode": "mixed",
  "topK": 3,
  "weightNdc": 0.85,
  "cluster": {
    "enabled": false,
    "threshold": 0.8
  },
  "embedder": {
    "ortDll": "./onnixruntime-win/lib/onnxruntime.dll",
    "modelPath": "./models/bge-m3/model.onnx",
    "tokenizerPath": "./models/bge-m3/tokenizer.json",
    "maxSeqLen": 512,
    "cacheDir": "./cache",
    "modelId": "bge-m3"
  },
  "seedsPath": "./seeds.txt",
  "useNdc": true
}
```
- `embedder.cacheDir` を設定すると、ベクトルをバイナリでディスクキャッシュし再利用できます。【F:categorizer/embedder.go†L41-L129】
- `seedsPath` を指定すると、起動時にシードファイルを読み込んで GUI に反映します。【F:main.go†L102-L121】

### シードファイルの形式
- UTF-8 テキストで、改行・カンマ・セミコロン区切りのいずれかでカテゴリを列挙します。【F:categorizer/io.go†L16-L64】
- 同じカテゴリは正規化後に自動的に重複排除されます。【F:categorizer/service.go†L82-L118】

### 入力テキストファイルの形式
- `.txt` は各行を 1 サンプルとして読み込みます。【F:categorizer/io.go†L66-L94】
- `.csv`/`.tsv` はヘッダー行から `text`・`本文`・`content` などの列名を推定し、その列を対象に読み込みます。【F:categorizer/io.go†L34-L109】

## 使い方
1. アプリケーションを起動します。
   ```bash
   go run .
   ```
2. **シードカテゴリの準備**
   - テキストボックスへ直接入力するか、「シードファイル読込」から外部ファイルを選択します。【F:main.go†L62-L112】
   - 「シード反映」で現在の入力内容を再ベクトル化し、シード数がステータスに表示されます。【F:main.go†L50-L101】
3. **分類対象テキストの入力**
   - 「テキスト入力」に 1 行ごとに文章を入力するか、「テキスト読込」でファイルから読み込みます。【F:main.go†L123-L207】
4. **推論の実行**
   - 「分類実行」を押すとバックグラウンドで推論が開始され、完了すると件数と処理時間が表示されます。【F:main.go†L174-L223】
5. **結果の確認**
   - テーブルに `text` と候補がスコア付きで表示されます。セルを選択すると、原文と候補の詳細ダイアログが開きます。【F:main.go†L129-L173】
   - NDC 分離モード時は NDC 列も追加されます。【F:main.go†L154-L173】
6. **結果の保存**
   - 「結果をCSV出力」で任意のパスに保存できます。ヘッダー付き CSV として書き出され、他システムへの連携に利用できます。【F:main.go†L207-L239】
7. **設定の調整**
   - 画面右側の設定パネルでモードや閾値を調整すると、即座にサービスへ反映され `config.json` に保存されます。【F:main.go†L241-L342】

### CLI での一括分類
- GUI を使わずに分類を完結させたい場合は、`--batch-input` と `--category-file` を指定して CLI モードで起動します。【F:main.go†L60-L145】
- 入力ファイルの列名が標準の `text` や `body` 以外の場合は、`--input-text-column` や `--input-body-column` などのオプションで列を明示すると確実です。【F:main.go†L32-L47】【F:categorizer/io.go†L34-L109】
- 実行例:
  ```bash
  go run . \
    --batch-input ./inputs.tsv \
    --category-file ./seeds.csv \
    --input-text-column "#3" \
    --output-dir ./csv
  ```
- 分類が完了すると `分類結果を <出力パス> に保存しました` が表示され、結果 CSV が `--output-dir`（未指定時は `csv/`）に生成されます。【F:main.go†L139-L144】

## CLI デバッグモード
- GUI を経由せずにシードの読み込みやテキスト分類を一通り再現したい場合は、`--debug-` 系フラグを付けて CLI モードを起動できます。【F:main.go†L36-L104】【F:main.go†L126-L229】
- 例: シード CSV を読み込み、正規化済みシードを `seeds_normalized.txt` に書き出した上でテキストファイルを分類し、ログをすべて標準出力に流すには次のコマンドを実行します。
  ```bash
  go run . --debug-seed-cli \
    --debug-seed-file ./seeds.csv \
    --debug-text-file ./inputs.tsv \
    --debug-save-results \
    --debug-seed-output ./seeds_normalized.txt
  ```
- `--debug-seed-text` を指定すると CLI から直接入力したシード文字列を利用できます。`--debug-disable-ndc` で NDC 辞書のロードを抑止し、`--output-dir` は `--debug-save-results` と併用したときの CSV 出力先になります。【F:main.go†L36-L104】【F:main.go†L126-L229】
- デバッグ実行中はサービスが取り込んだ正規化済みシード一覧もログへ出力されるため、GUI フリーズ時にどこまで処理が進んでいるかを CLI で追跡できます。【F:main.go†L155-L196】【F:categorizer/service.go†L92-L134】

## 設定項目の詳細
- **Mode** (`seeded` / `mixed` / `split`): シードのみ、シード+NDC 混合、シードと NDC を別リストで提示するモードを選択します。【F:categorizer/types.go†L5-L23】【F:main.go†L241-L254】
- **Top-K**: 候補数。スライダで 3〜5 の範囲を指定します。【F:main.go†L649-L668】
- **NDC重み**: NDC 候補のスコアに掛ける重み。0.70〜1.00 の範囲で調整できます。【F:main.go†L670-L685】【F:categorizer/service.go†L160-L215】
- **類似カテゴリを束ねる**: ON にすると、類似スコアが閾値以上の候補をまとめて表示します。【F:main.go†L302-L331】【F:categorizer/cluster.go†L1-L62】
- **NDC 提案を使用**: NDC 辞書のロード／解除を切り替えます。【F:main.go†L332-L342】【F:categorizer/service.go†L39-L108】

## キャッシュとログ
- 埋め込み結果はメモリとディスク双方にキャッシュされ、同一テキスト・同一モデルでの再推論を高速化します。【F:categorizer/embedder.go†L41-L129】
- 画面下部のログ領域には処理状況やエラーが逐次追記されます。一定行数を超えると古い行から削除されます。【F:main.go†L343-L511】

## トラブルシューティング
- ONNX Runtime やモデルファイルのパスが誤っている場合、初期化時にエラーになります。`config.json` のパス設定を再確認してください。【F:emb/emb.go†L33-L87】
- テキスト列が検出できない CSV/TSV を読み込んだ場合はエラーになります。ヘッダー名を `text` などの認識可能な名称に変更してください。【F:categorizer/io.go†L34-L109】

## 開発者向けメモ
- GUI のエントリポイントは `main.go`、ドメインロジックは `categorizer` パッケージにまとまっています。【F:main.go†L1-L511】【F:categorizer/service.go†L1-L222】
- 動作確認には `go test ./...` でユニットテスト (存在する場合) を実行してください。

