package embed

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/Zinoki12/rag-ai-system/backoff"
	"github.com/Zinoki12/rag-ai-system/internal/config"
)

// Config selects and tunes an embedding provider.
type Config struct {
	Provider  string // google | ollama | fake
	Model     string // provider default when empty
	Dim       int    // provider default when zero
	APIKey    string // google only
	OllamaURL string // ollama only
	RPS       float64
	Burst     int
	Retry     backoff.Policy
}

// defaults per provider. The dimension is deliberately 768 across the board so
// the local and hosted providers are directly comparable during development.
var providerDefaults = map[string]struct {
	model string
	dim   int
	rps   float64
}{
	// The Gemini free tier allows roughly 100 requests per minute; 1.5/s leaves
	// headroom so a burst does not spend the run's budget on 429s.
	ProviderGoogle: {model: "gemini-embedding-001", dim: 768, rps: 1.5},
	// Local: no quota to respect, and the model itself is the bottleneck.
	ProviderOllama: {model: "nomic-embed-text", dim: 768, rps: 0},
	ProviderFake:   {model: "hashing-trick", dim: 768, rps: 0},
}

// ConfigFromEnv reads the connector's settings.
//
//	EMBED_PROVIDER   google | ollama | fake      (default: ollama)
//	EMBED_MODEL      model name                  (provider default)
//	EMBED_DIM        vector dimension            (provider default)
//	EMBED_RPS        max provider calls/second   (provider default)
//	EMBED_BURST      rate limiter burst          (default: 1)
//	GOOGLE_API_KEY   required when provider=google
//	OLLAMA_URL       default http://127.0.0.1:11434
func ConfigFromEnv() (Config, error) {
	provider := strings.ToLower(config.String("EMBED_PROVIDER", ProviderOllama))
	def, known := providerDefaults[provider]
	if !known {
		return Config{}, fmt.Errorf("EMBED_PROVIDER=%q: want one of %s, %s, %s",
			provider, ProviderGoogle, ProviderOllama, ProviderFake)
	}

	dim, err := config.Int("EMBED_DIM", def.dim)
	if err != nil {
		return Config{}, err
	}
	rps, err := config.Float("EMBED_RPS", def.rps)
	if err != nil {
		return Config{}, err
	}
	burst, err := config.Int("EMBED_BURST", 1)
	if err != nil {
		return Config{}, err
	}

	return Config{
		Provider:  provider,
		Model:     config.String("EMBED_MODEL", def.model),
		Dim:       dim,
		APIKey:    os.Getenv("GOOGLE_API_KEY"),
		OllamaURL: config.String("OLLAMA_URL", DefaultOllamaURL),
		RPS:       rps,
		Burst:     burst,
		Retry:     backoff.Default,
	}, nil
}

// New builds the configured provider already wrapped in the shared retry and
// rate-limit behaviour.
//
// This is the only place in the program that knows which vendors exist. Adding
// a third is a case in this switch plus one file; nothing downstream changes.
func New(ctx context.Context, cfg Config) (Provider, error) {
	def := providerDefaults[strings.ToLower(cfg.Provider)]
	if cfg.Model == "" {
		cfg.Model = def.model
	}
	if cfg.Dim == 0 {
		cfg.Dim = def.dim
	}

	var (
		p   Provider
		err error
	)
	switch strings.ToLower(cfg.Provider) {
	case ProviderGoogle:
		p, err = NewGoogle(ctx, cfg.APIKey, cfg.Model, cfg.Dim)
	case ProviderOllama:
		p, err = NewOllama(cfg.OllamaURL, cfg.Model, cfg.Dim)
	case ProviderFake:
		p, err = NewFake(cfg.Dim)
	default:
		return nil, fmt.Errorf("unknown embedding provider %q", cfg.Provider)
	}
	if err != nil {
		return nil, err
	}

	// Order matters: the limiter is outermost so a retry storm also waits its
	// turn. Wrapped the other way round, retries would bypass the quota that
	// caused them.
	return WithRateLimit(WithRetry(p, cfg.Retry), cfg.RPS, cfg.Burst), nil
}
