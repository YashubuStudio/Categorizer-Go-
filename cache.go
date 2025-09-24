package main

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

type embedCache struct {
	mu      sync.RWMutex
	m       map[string][]float32
	dir     string
	modelID string
}

func newEmbedCache(dir, modelID string) *embedCache {
	return &embedCache{m: make(map[string][]float32), dir: dir, modelID: modelID}
}

func (c *embedCache) get(key string) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.m[key]
	return v, ok
}

func (c *embedCache) put(key string, v []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[key] = v
}

func (c *embedCache) load(key string) ([]float32, bool, error) {
	if c.dir == "" {
		return nil, false, nil
	}
	path := filepath.Join(c.dir, key+".bin")
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}
	if len(data) < 4 {
		return nil, false, fmt.Errorf("cache file broken: %s", path)
	}
	length := binary.LittleEndian.Uint32(data[:4])
	need := int(length) * 4
	if len(data) < 4+need {
		return nil, false, fmt.Errorf("cache truncated: %s", path)
	}
	vec := make([]float32, int(length))
	if err := binary.Read(bytes.NewReader(data[4:4+need]), binary.LittleEndian, vec); err != nil {
		return nil, false, err
	}
	return vec, true, nil
}

func (c *embedCache) save(key string, v []float32) error {
	if c.dir == "" {
		return nil
	}
	path := filepath.Join(c.dir, key+".bin")
	buf := &bytes.Buffer{}
	_ = binary.Write(buf, binary.LittleEndian, uint32(len(v)))
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func cacheKey(text, model string) string {
	h := sha1.Sum([]byte(text + "|" + model))
	return hex.EncodeToString(h[:])
}
