package categorizer

import (
	"context"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"

	"yashubustudio/categorizer/emb"
)

// Embedder exposes the minimal surface required by the service layer.
type Embedder interface {
	EmbedText(ctx context.Context, text string) ([]float32, error)
	EmbedTexts(ctx context.Context, texts []string) ([][]float32, error)
	Close() error
	ModelID() string
}

// OrtEmbedder is a thin wrapper over emb.Encoder with caching.
type OrtEmbedder struct {
	enc      *emb.Encoder
	cfg      EmbedderConfig
	memCache map[string][]float32
	mu       sync.RWMutex
}

// NewOrtEmbedder initializes the encoder and prepares cache directories.
func NewOrtEmbedder(cfg EmbedderConfig) (*OrtEmbedder, error) {
	if cfg.ModelID == "" && cfg.ModelPath != "" {
		cfg.ModelID = filepath.Base(cfg.ModelPath)
	}
	if cfg.CacheDir != "" {
		if err := os.MkdirAll(cfg.CacheDir, 0o755); err != nil {
			return nil, fmt.Errorf("create cache dir: %w", err)
		}
	}
	encoder := &emb.Encoder{}
	if err := encoder.Init(emb.Config{
		OrtDLL:        cfg.OrtDLL,
		ModelPath:     cfg.ModelPath,
		TokenizerPath: cfg.TokenizerPath,
		MaxSeqLen:     cfg.MaxSeqLen,
	}); err != nil {
		return nil, err
	}
	return &OrtEmbedder{
		enc:      encoder,
		cfg:      cfg,
		memCache: make(map[string][]float32),
	}, nil
}

// Close releases ORT resources.
func (o *OrtEmbedder) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.enc != nil {
		o.enc.Close()
		o.enc = nil
	}
	o.memCache = nil
	return nil
}

// ModelID returns the identifier used for cache keys.
func (o *OrtEmbedder) ModelID() string {
	return o.cfg.ModelID
}

// EmbedText embeds a single string with caching.
func (o *OrtEmbedder) EmbedText(_ context.Context, text string) ([]float32, error) {
	if o == nil || o.enc == nil {
		return nil, errors.New("embedder is not initialized")
	}
	normalized := NormalizeText(text)
	key := o.cacheKey(normalized)
	if vec := o.getFromCache(key); vec != nil {
		return vec, nil
	}
	if vec, err := o.loadFromDisk(key); err == nil {
		o.storeInMemory(key, vec)
		return cloneVector(vec), nil
	}
	vec, err := o.enc.Encode(normalized)
	if err != nil {
		return nil, err
	}
	o.storeInMemory(key, vec)
	_ = o.saveToDisk(key, vec)
	return cloneVector(vec), nil
}

// EmbedTexts embeds a slice of strings sequentially.
func (o *OrtEmbedder) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		vec, err := o.EmbedText(ctx, t)
		if err != nil {
			return nil, err
		}
		out[i] = vec
	}
	return out, nil
}

func (o *OrtEmbedder) cacheKey(text string) string {
	h := sha1.New()
	_, _ = io.WriteString(h, o.cfg.ModelID)
	_, _ = io.WriteString(h, "|")
	_, _ = io.WriteString(h, text)
	return hex.EncodeToString(h.Sum(nil))
}

func (o *OrtEmbedder) getFromCache(key string) []float32 {
	o.mu.RLock()
	defer o.mu.RUnlock()
	if vec, ok := o.memCache[key]; ok {
		return cloneVector(vec)
	}
	return nil
}

func (o *OrtEmbedder) storeInMemory(key string, vec []float32) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.memCache[key] = cloneVector(vec)
}

func (o *OrtEmbedder) loadFromDisk(key string) ([]float32, error) {
	if o.cfg.CacheDir == "" {
		return nil, os.ErrNotExist
	}
	path := filepath.Join(o.cfg.CacheDir, key+".bin")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 4 {
		return nil, fmt.Errorf("cache file too small: %s", path)
	}
	length := int(binary.LittleEndian.Uint32(data[:4]))
	data = data[4:]
	if len(data) != length*4 {
		return nil, fmt.Errorf("cache length mismatch: %s", path)
	}
	vec := make([]float32, length)
	for i := 0; i < length; i++ {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : (i+1)*4]))
	}
	return vec, nil
}

func (o *OrtEmbedder) saveToDisk(key string, vec []float32) error {
	if o.cfg.CacheDir == "" {
		return nil
	}
	path := filepath.Join(o.cfg.CacheDir, key+".bin")
	tmp := path + ".tmp"
	buf := make([]byte, 4+len(vec)*4)
	binary.LittleEndian.PutUint32(buf[:4], uint32(len(vec)))
	off := 4
	for _, v := range vec {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(v))
		off += 4
	}
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func cloneVector(vec []float32) []float32 {
	out := make([]float32, len(vec))
	copy(out, vec)
	return out
}
