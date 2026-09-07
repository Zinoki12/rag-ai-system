package llm

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

func TestConfigFromEnv(t *testing.T) {
	t.Run("по умолчанию — заглушка", func(t *testing.T) {
		cfg, err := ConfigFromEnv()
		if err != nil {
			t.Fatalf("ConfigFromEnv: %v", err)
		}
		// A fresh clone with no key and no downloaded model must still run end
		// to end rather than fail at startup on a missing credential.
		if cfg.Provider != ProviderStub {
			t.Errorf("Provider = %q, want %q", cfg.Provider, ProviderStub)
		}
	})

	t.Run("неизвестный провайдер называет допустимые", func(t *testing.T) {
		t.Setenv("LLM_PROVIDER", "no-such-vendor")
		_, err := ConfigFromEnv()
		if err == nil {
			t.Fatal("ConfigFromEnv accepted an unknown provider")
		}
		for _, want := range []string{ProviderGoogle, ProviderOpenAI, ProviderOllama, ProviderStub} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not mention %q", err, want)
			}
		}
	})

	t.Run("модель по умолчанию берётся от провайдера", func(t *testing.T) {
		t.Setenv("LLM_PROVIDER", "ollama")
		cfg, err := ConfigFromEnv()
		if err != nil {
			t.Fatalf("ConfigFromEnv: %v", err)
		}
		if cfg.Model != providerDefaults[ProviderOllama].model {
			t.Errorf("Model = %q, want the ollama default", cfg.Model)
		}
	})

	t.Run("нечисловой LLM_RPS падает с именем переменной", func(t *testing.T) {
		t.Setenv("LLM_RPS", "быстро")
		_, err := ConfigFromEnv()
		if err == nil || !strings.Contains(err.Error(), "LLM_RPS") {
			t.Errorf("error = %v, want it to name LLM_RPS", err)
		}
	})
}

func TestNewRejectsGoogleWithoutKey(t *testing.T) {
	_, err := New(context.Background(), Config{Provider: ProviderGoogle, Model: "gemini-3.6-flash"})
	if err == nil {
		t.Fatal("New built a Google provider with no API key")
	}
	if !strings.Contains(err.Error(), "GOOGLE_API_KEY") {
		t.Errorf("error = %q, want it to name the variable to set", err)
	}
}

func TestNewUnknownProvider(t *testing.T) {
	if _, err := New(context.Background(), Config{Provider: "no-such-vendor"}); err == nil {
		t.Error("New accepted an unknown provider")
	}
}

func TestNewFillsInTheDefaultModel(t *testing.T) {
	p, err := New(context.Background(), Config{Provider: ProviderOllama})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p.Model() != providerDefaults[ProviderOllama].model {
		t.Errorf("Model() = %q, want the provider default", p.Model())
	}
}

func TestStubAnswersWithoutAModel(t *testing.T) {
	p := NewStub()
	out, err := p.Generate(context.Background(), "система", "контекст и вопрос")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// The stub must announce itself. An answer that reads like a model's would
	// let a stub run pass for a working one.
	if !strings.Contains(out, "stub") {
		t.Errorf("stub answer %q does not say it is a stub", out)
	}
	if !strings.Contains(out, "контекст и вопрос") {
		t.Error("stub answer dropped the retrieved context")
	}
}

func TestRateLimitDelaysTheSecondCall(t *testing.T) {
	// 20 rps with burst 1: the first call goes straight through, the second
	// waits about 50ms. Without the limiter both return instantly.
	p := WithRateLimit(NewStub(), 20, 1)

	start := time.Now()
	for range 2 {
		if _, err := p.Generate(context.Background(), "", "x"); err != nil {
			t.Fatalf("Generate: %v", err)
		}
	}
	if elapsed := time.Since(start); elapsed < 25*time.Millisecond {
		t.Errorf("two calls took %v, want the limiter to have delayed the second", elapsed)
	}
}

func TestRateLimitOfZeroIsNotWrapped(t *testing.T) {
	stub := NewStub()
	if got := WithRateLimit(stub, 0, 1); got != stub {
		t.Error("WithRateLimit(0) wrapped the provider; it should be a no-op")
	}
}

// classifyGoogleErr decides what gets retried. Getting this wrong is expensive
// in both directions: retrying a 400 burns quota against a request that can
// never succeed, and not retrying a 503 turns a hiccup into a failed answer.
func TestClassifyGoogleErr(t *testing.T) {
	cases := []struct {
		name      string
		code      int
		wantRetry bool
	}{
		{"429 повторяется", http.StatusTooManyRequests, true},
		{"500 повторяется", http.StatusInternalServerError, true},
		{"503 повторяется", http.StatusServiceUnavailable, true},
		{"400 не повторяется", http.StatusBadRequest, false},
		{"401 не повторяется", http.StatusUnauthorized, false},
		// The model-retired case: 404 with a message naming the replacement.
		// Retrying it would hide that message behind three more failures.
		{"404 не повторяется", http.StatusNotFound, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := classifyGoogleErr(genai.APIError{Code: c.code, Message: "нет"})

			var retryable *backoff.Retryable
			if got := errors.As(err, &retryable); got != c.wantRetry {
				t.Errorf("retryable = %v, want %v (error: %v)", got, c.wantRetry, err)
			}
			// The upstream message must survive: it is often the only thing
			// that says what to change.
			if !strings.Contains(err.Error(), "нет") {
				t.Errorf("error %q dropped the upstream message", err)
			}
		})
	}
}
