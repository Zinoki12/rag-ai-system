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
	Provider  string // google | openai | openrouter | ollama | stub
	Model     string
	APIKey    string // google, openai and openrouter
	BaseURL   string // openai only: endpoint before /chat/completions
	OllamaURL string

	// AppURL and AppName identify the calling application to a broker that
	// attributes traffic — today only OpenRouter, which reads them as
	// HTTP-Referer and X-Title. Both are optional and neither changes an answer.
	AppURL  string
	AppName string
	RPS     float64
	Burst   int
	Retry   backoff.Policy
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
	// No default model: an OpenAI-compatible endpoint may be any service, and a
	// guessed name comes back as a 404 that reads like a bug in this code.
	ProviderOpenAI: {model: "", rps: 1},
	// Also no default model, for the same reason, and one more: OpenRouter's
	// free tier is rate limited per account, so the sensible rate is low.
	ProviderOpenRouter: {model: "", rps: 1},
	ProviderOllama:     {model: "llama3.2", rps: 0},
	ProviderStub:       {model: "stub", rps: 0},
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
		return Config{}, fmt.Errorf("LLM_PROVIDER=%q: want one of %s, %s, %s, %s, %s",
			provider, ProviderGoogle, ProviderOpenAI, ProviderOpenRouter,
			ProviderOllama, ProviderStub)
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
		APIKey:    llmAPIKey(provider),
		BaseURL:   os.Getenv("LLM_BASE_URL"),
		OllamaURL: config.String("OLLAMA_URL", embed.DefaultOllamaURL),
		AppURL:    config.String("OPENROUTER_REFERER", defaultAppURL),
		AppName:   config.String("OPENROUTER_TITLE", defaultAppName),
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
	case ProviderOpenAI:
		p, err = NewOpenAI(cfg.BaseURL, cfg.APIKey, cfg.Model)
	case ProviderOpenRouter:
		p, err = NewOpenRouter(cfg.APIKey, cfg.Model, cfg.AppURL, cfg.AppName)
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

// llmAPIKey picks the variable belonging to the selected provider, for the same
// reason embedAPIKey does: never send one vendor's credential to another's host.
func llmAPIKey(provider string) string {
	switch provider {
	case ProviderOpenAI:
		return os.Getenv("LLM_API_KEY")
	case ProviderOpenRouter:
		// OPENROUTER_API_KEY first, because a machine that talks to OpenRouter
		// and to some other OpenAI-compatible host needs to keep the two keys
		// apart. LLM_API_KEY still works for the single-provider case.
		if k := os.Getenv("OPENROUTER_API_KEY"); k != "" {
			return k
		}
		return os.Getenv("LLM_API_KEY")
	default:
		return os.Getenv("GOOGLE_API_KEY")
	}
}

// Defaults for the attribution headers. They name this library, not whatever
// links it: an application that wants its own name in OpenRouter's dashboard
// sets OPENROUTER_TITLE, and the library has no way to guess it.
const (
	defaultAppURL  = "https://github.com/Zinoki12/rag-ai-system"
	defaultAppName = "rag-ai-system"
)
