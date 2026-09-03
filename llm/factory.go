package llm

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/Zinoki12/rag-ai-system/backoff"
	"github.com/Zinoki12/rag-ai-system/embed"
	"github.com/Zinoki12/rag-ai-system/internal/config"
)

// Config selects and tunes a generation provider.
type Config struct {
	Provider  string // google | ollama | stub
	Model     string
	APIKey    string
	OllamaURL string
	RPS       float64
	Burst     int
	Retry     backoff.Policy
}

var providerDefaults = map[string]struct {
	model string
	rps   float64
}{
	// Verified live on 2026-09-03. Gemini model names retire: the API answers a
	// withdrawn one with a 404 that names the replacement, so if generation
	// starts failing with "no longer available to new users", read the message
	// and update this line.
	ProviderGoogle: {model: "gemini-3.6-flash", rps: 1},
	// llama3.2 is the smallest model that answers coherently in Russian and
	// still fits comfortably on a laptop.
	ProviderOllama: {model: "llama3.2", rps: 0},
	ProviderStub:   {model: "stub", rps: 0},
}

// ConfigFromEnv reads the generator's settings.
//
//	LLM_PROVIDER   google | ollama | stub   (default: stub)
//	LLM_MODEL      model name               (provider default)
//	LLM_RPS        max calls/second         (provider default)
//	LLM_BURST      rate limiter burst       (default: 1)
//	GOOGLE_API_KEY required when provider=google
//	OLLAMA_URL     default http://127.0.0.1:11434
//
// The default is deliberately the stub: a fresh clone with no API key and no
// model downloaded still runs end to end and shows retrieved sources, instead
// of failing at startup on a missing credential.
func ConfigFromEnv() (Config, error) {
	provider := strings.ToLower(config.String("LLM_PROVIDER", ProviderStub))
	def, known := providerDefaults[provider]
	if !known {
		return Config{}, fmt.Errorf("LLM_PROVIDER=%q: want one of %s, %s, %s",
			provider, ProviderGoogle, ProviderOllama, ProviderStub)
	}

	rps, err := config.Float("LLM_RPS", def.rps)
	if err != nil {
		return Config{}, err
	}
	burst, err := config.Int("LLM_BURST", 1)
	if err != nil {
		return Config{}, err
	}

	return Config{
		Provider:  provider,
		Model:     config.String("LLM_MODEL", def.model),
		APIKey:    os.Getenv("GOOGLE_API_KEY"),
		OllamaURL: config.String("OLLAMA_URL", embed.DefaultOllamaURL),
		RPS:       rps,
		Burst:     burst,
		Retry:     backoff.Default,
	}, nil
}

// New builds the configured provider wrapped in the shared retry and
// rate-limit behaviour.
func New(ctx context.Context, cfg Config) (Provider, error) {
	if cfg.Model == "" {
		cfg.Model = providerDefaults[strings.ToLower(cfg.Provider)].model
	}

	var (
		p   Provider
		err error
	)
	switch strings.ToLower(cfg.Provider) {
	case ProviderGoogle:
		p, err = NewGoogle(ctx, cfg.APIKey, cfg.Model)
	case ProviderOllama:
		p, err = NewOllama(cfg.OllamaURL, cfg.Model)
	case ProviderStub:
		p = NewStub()
	default:
		return nil, fmt.Errorf("unknown generation provider %q", cfg.Provider)
	}
	if err != nil {
		return nil, err
	}

	// The limiter sits outside the retries so a retry storm also waits its
	// turn; the other order would let retries bypass the quota that caused them.
	return WithRateLimit(WithRetry(p, cfg.Retry), cfg.RPS, cfg.Burst), nil
}
