package app

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

type keywordRuleSet struct {
	Strong []string
	Weak   []string
	Anti   []string
}

type compiledRuleSet struct {
	strong []string
	weak   []string
	anti   []string
}

var rawCategoryRules = map[string]keywordRuleSet{
	"CG・デジタルアーカイブ": {
		Strong: []string{
			"デジタルアーカイブ",
			"文化財",
			"文化遺産",
			"博物館資料",
			"フォトグラメトリ",
			"写真測量",
			"三次元復元",
			"3D復元",
			"点群",
			"LiDAR",
			"レーザースキャン",
			"スキャンデータ",
			"メッシュ再構成",
			"SfM",
			"Structure from Motion",
		},
		Weak: []string{
			"モデリング",
			"リトポロジー",
			"リトポ",
			"UV展開",
			"テクスチャ",
			"ベイク",
			"レンダリング",
			"GLTF",
			"GLB",
			"OBJ",
			"PLY",
			"点群処理",
			"データ保存",
			"アノテーション",
		},
		Anti: nil,
	},
	"VR空間": {
		Strong: []string{
			"VR空間",
			"仮想空間",
			"仮想環境",
			"仮想世界",
			"メタバース",
			"VRChat",
			"cluster",
			"Neos",
			"Spatial",
			"HMD",
			"ヘッドマウントディスプレイ",
			"Oculus",
			"Meta Quest",
			"Quest2",
			"Quest3",
			"Vive",
			"Index",
			"OpenXR",
		},
		Weak: []string{
			"インスタンス",
			"ワールド",
			"ルームスケール",
			"トラッキング",
			"フルトラ",
			"IK",
			"アイトラ",
			"ハンドトラッキング",
			"アイトラッキング",
		},
		Anti: nil,
	},
	"アバター": {
		Strong: []string{
			"アバター",
			"アバター生成",
			"アバター編集",
			"フェイシャル",
			"表情認識",
			"モーションキャプチャ",
			"モーキャプ",
			"リギング",
			"スキニング",
			"ボーン",
			"ブレンドシェイプ",
			"VRCアバター",
			"VRM",
			"Humanoid",
		},
		Weak: []string{
			"衣装",
			"衣装替え",
			"髪物理",
			"揺れもの",
			"体型調整",
			"顔トラッキング",
			"表情制御",
		},
		Anti: nil,
	},
	"インタラクション": {
		Strong: []string{
			"インタラクション",
			"UI",
			"UX",
			"操作手法",
			"選択操作",
			"ポインティング",
			"ジェスチャ",
			"姿勢推定",
			"身体性",
			"ハプティクス",
			"触覚提示",
			"力覚提示",
			"モーダル切替",
			"メニュー操作",
		},
		Weak: []string{
			"没入感",
			"プレゼンス",
			"使い勝手",
			"フィードバック",
			"提示手法",
			"視線入力",
			"手入力",
			"音声入力",
		},
		Anti: nil,
	},
	"エージェント": {
		Strong: []string{
			"エージェント",
			"会話エージェント",
			"対話エージェント",
			"自律エージェント",
			"LLMエージェント",
			"NPC",
			"行動計画",
			"プランニング",
			"強化学習",
			"RL",
			"Reinforcement Learning",
		},
		Weak: []string{
			"アシスタント",
			"ナビゲーション",
			"案内",
			"対話支援",
			"ルールベース",
		},
		Anti: nil,
	},
	"コミュニケーション": {
		Strong: []string{
			"コミュニケーション",
			"会話",
			"対話",
			"交流",
			"ソーシャルサポート",
			"関係構築",
			"協調作業",
			"コラボレーション",
		},
		Weak: []string{
			"雑談",
			"アイスブレイク",
			"発話",
			"感情",
			"感情推定",
			"同席感",
		},
		Anti: nil,
	},
	"ソーシャルVR": {
		Strong: []string{
			"ソーシャルVR",
			"VRChat",
			"cluster",
			"Neos",
			"メタバースプラットフォーム",
			"インスタンス制御",
			"フレンド機能",
			"イベント開催",
		},
		Weak: []string{
			"アバターマーケット",
			"ワールドアップロード",
			"コミュニティ運営",
			"Booth",
			"配信イベント",
		},
		Anti: nil,
	},
	"可視化": {
		Strong: []string{
			"可視化",
			"視覚化",
			"可視化手法",
			"可視化技術",
			"可視化結果",
			"ボリュームレンダリング",
			"等値面",
			"流線",
			"点群可視化",
		},
		Weak: []string{
			"3D可視化",
			"VR可視化",
			"インタラクティブ可視化",
			"可視化ツール",
		},
		Anti: nil,
	},
	"工学・サイエンスコミュニケーション": {
		Strong: []string{
			"科学コミュニケーション",
			"サイエンスコミュニケーション",
			"アウトリーチ",
			"展示解説",
			"科学館",
			"博物館",
			"ミュージアム",
			"STEAM",
		},
		Weak: []string{
			"市民参加",
			"ワークショップ",
			"普及啓発",
			"体験学習",
		},
		Anti: nil,
	},
	"応用数理": {
		Strong: []string{
			"最適化",
			"数値解析",
			"数理モデル",
			"シミュレーション",
			"微分方程式",
			"ベイズ推定",
			"確率過程",
			"離散化",
			"有限要素法",
			"FEM",
		},
		Weak: []string{
			"近似",
			"数理的",
			"解析的",
			"再現実験",
		},
		Anti: nil,
	},
	"感覚・知覚": {
		Strong: []string{
			"知覚",
			"感覚",
			"多感覚",
			"触覚",
			"前庭",
			"視覚認知",
			"錯視",
			"VR酔い",
			"サッカード",
			"順応",
			"感度",
		},
		Weak: []string{
			"感性",
			"疲労",
			"快不快",
			"主観評価",
			"SSQ",
			"SUS",
		},
		Anti: nil,
	},
	"教育": {
		Strong: []string{
			"教育",
			"授業",
			"教材",
			"学習",
			"訓練",
			"トレーニング",
			"指導",
			"評価",
			"ルーブリック",
			"授業設計",
		},
		Weak: []string{
			"eラーニング",
			"学習効果",
			"学習支援",
			"教育実践",
			"実証授業",
		},
		Anti: nil,
	},
	"機械学習": {
		Strong: []string{
			"機械学習",
			"ディープラーニング",
			"深層学習",
			"ニューラルネットワーク",
			"Transformer",
			"BERT",
			"学習モデル",
			"分類器",
			"回帰",
		},
		Weak: []string{
			"特徴量",
			"埋め込み",
			"ベクトル",
			"クラスタリング",
			"次元削減",
		},
		Anti: nil,
	},
	"社会": {
		Strong: []string{
			"社会",
			"倫理",
			"ガバナンス",
			"プライバシー",
			"規範",
			"制度",
			"アクセシビリティ",
			"包摂",
			"障害当事者",
		},
		Weak: []string{
			"普及",
			"受容",
			"合意形成",
			"文化",
			"コミュニティ規約",
		},
		Anti: nil,
	},
}

var compiledCategoryRules = compileCategoryRules(rawCategoryRules)

var vrCategoryKeySet = buildVRCategoryKeySet()

var dampCategoryKeys = []string{normalizeKey("教育"), normalizeKey("可視化")}

const (
	strongWeight  float32 = 1.0
	weakWeight    float32 = 0.25
	antiWeight    float32 = 1.0
	strongCap             = 3
	weakCap               = 5
	bonusCapValue float32 = 4.0
	alphaWeight   float32 = 0.80
	betaWeight    float32 = 0.20
	floorForced   float32 = 0.60
	dampValue     float32 = 0.03
)

func computeBaseScores(vec []float32, cands []Candidate) map[string]float32 {
	scores := make(map[string]float32, len(cands))
	for _, c := range cands {
		sc := cosine32(vec, c.Vec)
		if sc < 0 {
			sc = 0
		}
		scores[c.Label] = clamp01(sc)
	}
	return scores
}

func applyHybridScoring(text string, cands []Candidate, baseScores map[string]float32, seedBias float32) ([]Suggestion, map[string]float32, map[string]float32) {
	ruleBonus := make(map[string]float32, len(cands))
	finalScores := make(map[string]float32, len(cands))

	hasVRSignal := false
	for _, c := range cands {
		base := baseScores[c.Label]
		rules, ok := compiledCategoryRules[c.Key]
		if !ok {
			rules = compiledRuleSet{}
		}
		strongHits, weakHits, antiHits := countRuleHits(text, rules)
		bonus := computeRuleBonus(strongHits, weakHits, antiHits)
		ruleBonus[c.Label] = bonus

		final := alphaWeight * base
		if bonus > 0 {
			final += betaWeight * (bonus / bonusCapValue)
		}
		if strongHits > 0 && final < floorForced {
			final = floorForced
		}
		final += seedBias
		final += tinyBias(c.Key)
		final = clamp01(final)
		finalScores[c.Label] = final

		if !hasVRSignal {
			if _, ok := vrCategoryKeySet[c.Key]; ok && strongHits > 0 {
				hasVRSignal = true
			}
		}
	}

	if hasVRSignal {
		for _, targetKey := range dampCategoryKeys {
			if targetKey == "" {
				continue
			}
			for _, c := range cands {
				if c.Key != targetKey {
					continue
				}
				if score, ok := finalScores[c.Label]; ok {
					adjusted := score - dampValue
					if adjusted < 0 {
						adjusted = 0
					}
					finalScores[c.Label] = adjusted
				}
				break
			}
		}
	}

	suggestions := make([]Suggestion, 0, len(finalScores))
	for _, c := range cands {
		if score, ok := finalScores[c.Label]; ok {
			suggestions = append(suggestions, Suggestion{
				Label:  c.Label,
				Score:  score,
				Source: "hybrid",
			})
		}
	}
	sort.SliceStable(suggestions, func(i, j int) bool {
		if suggestions[i].Score == suggestions[j].Score {
			return suggestions[i].Label < suggestions[j].Label
		}
		return suggestions[i].Score > suggestions[j].Score
	})
	return suggestions, ruleBonus, finalScores
}

func compileCategoryRules(raw map[string]keywordRuleSet) map[string]compiledRuleSet {
	compiled := make(map[string]compiledRuleSet, len(raw))
	for label, set := range raw {
		key := normalizeKey(label)
		if key == "" {
			continue
		}
		compiled[key] = compiledRuleSet{
			strong: normalizeKeywordList(set.Strong),
			weak:   normalizeKeywordList(set.Weak),
			anti:   normalizeKeywordList(set.Anti),
		}
	}
	return compiled
}

func buildVRCategoryKeySet() map[string]struct{} {
	keys := make(map[string]struct{})
	for _, label := range []string{"VR空間", "インタラクション", "アバター"} {
		key := normalizeKey(label)
		if key == "" {
			continue
		}
		keys[key] = struct{}{}
	}
	return keys
}

func normalizeKeywordList(words []string) []string {
	if len(words) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(words))
	res := make([]string, 0, len(words))
	for _, w := range words {
		normed := normalizeText(w)
		if normed == "" {
			continue
		}
		if _, ok := seen[normed]; ok {
			continue
		}
		seen[normed] = struct{}{}
		res = append(res, normed)
	}
	return res
}

func countRuleHits(text string, set compiledRuleSet) (int, int, int) {
	strong := countKeywordHits(text, set.strong)
	weak := countKeywordHits(text, set.weak)
	anti := countKeywordHits(text, set.anti)
	return strong, weak, anti
}

func countKeywordHits(text string, keywords []string) int {
	if len(keywords) == 0 {
		return 0
	}
	hits := 0
	for _, kw := range keywords {
		if containsKeyword(text, kw) {
			hits++
		}
	}
	return hits
}

func containsKeyword(text, kw string) bool {
	if kw == "" {
		return false
	}
	if useWordBoundary(kw) {
		return containsAsWord(text, kw)
	}
	return strings.Contains(text, kw)
}

func useWordBoundary(kw string) bool {
	if kw == "" {
		return false
	}
	count := 0
	for _, r := range kw {
		if r > unicode.MaxASCII {
			return false
		}
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return false
		}
		count++
		if count > 3 {
			return false
		}
	}
	return count > 0
}

func containsAsWord(text, word string) bool {
	start := 0
	for start < len(text) {
		idx := strings.Index(text[start:], word)
		if idx < 0 {
			return false
		}
		idx += start
		var before rune
		if idx > 0 {
			before, _ = utf8.DecodeLastRuneInString(text[:idx])
		}
		var after rune
		if end := idx + len(word); end < len(text) {
			after, _ = utf8.DecodeRuneInString(text[end:])
		}
		if !isAlphaNumRune(before) && !isAlphaNumRune(after) {
			return true
		}
		start = idx + len(word)
	}
	return false
}

func isAlphaNumRune(r rune) bool {
	if r == 0 || r == utf8.RuneError {
		return false
	}
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

func computeRuleBonus(strongHits, weakHits, antiHits int) float32 {
	s := strongHits
	if s > strongCap {
		s = strongCap
	}
	w := weakHits
	if w > weakCap {
		w = weakCap
	}
	bonus := strongWeight*float32(s) + weakWeight*float32(w)
	if antiHits > 0 {
		bonus -= antiWeight * float32(antiHits)
	}
	if bonus < 0 {
		bonus = 0
	}
	if bonus > bonusCapValue {
		bonus = bonusCapValue
	}
	return bonus
}
