package categorizer

import (
	"fmt"
	"sort"
	"strings"
)

type cluster struct {
	repr    []float32
	best    Hit
	members []Hit
}

func clusterHits(hits []Hit, threshold float32) []Hit {
	if len(hits) <= 1 || threshold <= 0 {
		return hits
	}
	clusters := make([]cluster, 0, len(hits))
	for _, h := range hits {
		if len(h.Vector) == 0 {
			clusters = append(clusters, cluster{repr: nil, best: h, members: []Hit{h}})
			continue
		}
		assigned := false
		for i := range clusters {
			if len(clusters[i].repr) == 0 {
				continue
			}
			if cosineSimilarity(h.Vector, clusters[i].repr) >= threshold {
				clusters[i].members = append(clusters[i].members, h)
				if h.Score > clusters[i].best.Score {
					clusters[i].best = h
					clusters[i].repr = cloneVector(h.Vector)
				}
				assigned = true
				break
			}
		}
		if !assigned {
			clusters = append(clusters, cluster{
				repr:    cloneVector(h.Vector),
				best:    h,
				members: []Hit{h},
			})
		}
	}
	out := make([]Hit, 0, len(clusters))
	for _, c := range clusters {
		if len(c.members) == 0 {
			continue
		}
		label := c.best.Label
		if len(c.members) > 1 {
			extras := make([]string, 0, len(c.members)-1)
			seen := map[string]struct{}{c.best.Label: {}}
			for _, m := range c.members {
				if m.Label == c.best.Label {
					continue
				}
				if _, ok := seen[m.Label]; ok {
					continue
				}
				seen[m.Label] = struct{}{}
				extras = append(extras, m.Label)
			}
			if len(extras) > 0 {
				label = fmt.Sprintf("%s（類似: %s）", label, strings.Join(extras, ", "))
			}
		}
		out = append(out, Hit{
			Label:  label,
			Score:  c.best.Score,
			Source: c.best.Source,
			Vector: c.best.Vector,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].Label < out[j].Label
		}
		return out[i].Score > out[j].Score
	})
	return out
}
