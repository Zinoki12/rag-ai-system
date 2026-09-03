package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Zinoki12/rag-ai-system/internal/backoff"
)

// respondWith serves one canned reply and captures the request body.
func respondWith(t *testing.T, status int, body string, captured *ollamaEmbedRequest) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if captured != nil {
			if err := json.NewDecoder(r.Body).Decode(captured); err != nil {
				t.Errorf("decode request: %v", err)
			}
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func vectorsJSON(vecs ...[]float32) string {
	b, _ := json.Marshal(ollamaEmbedResponse{Embeddings: vecs})
	return string(b)
}

func TestOllamaEmbedSuccess(t *testing.T) {
	var got ollamaEmbedRequest
	srv := respondWith(t, http.StatusOK, vectorsJSON([]float32{1, 2, 3}, []float32{4, 5, 6}), &got)

	p, err := NewOllama(srv.URL, "nomic-embed-text", 3)
	if err != nil {
		t.Fatalf("NewOllama: %v", err)
	}

	vecs, err := p.Embed(context.Background(), []string{"один", "два"}, KindDocument)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 3 {
		t.Fatalf("got %d vectors of width %d, want 2x3", len(vecs), len(vecs[0]))
	}
	if got.Model != "nomic-embed-text" {
		t.Errorf("request model = %q, want nomic-embed-text", got.Model)
	}
}

// nomic models are trained to read the task from a prefix in the text itself.
// Sending a query without its prefix returns perfectly valid vectors that rank
// badly — there is no error to catch it, so a test has to.
func TestOllamaAppliesTaskPrefix(t *testing.T) {
	tests := []struct {
		name       string
		model      string
		kind       Kind
		wantPrefix string
	}{
		{"nomic, документ", "nomic-embed-text", KindDocument, "search_document: "},
		{"nomic, запрос", "nomic-embed-text", KindQuery, "search_query: "},
		{"nomic с тегом версии", "nomic-embed-text:latest", KindQuery, "search_query: "},
		{"mxbai, запрос", "mxbai-embed-large", KindQuery, "Represent this sentence for searching relevant passages: "},
		{"mxbai, документ — без префикса", "mxbai-embed-large", KindDocument, ""},
		{"неизвестная модель — без префикса", "some-other-model", KindQuery, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got ollamaEmbedRequest
			srv := respondWith(t, http.StatusOK, vectorsJSON([]float32{1, 2}), &got)

			p, err := NewOllama(srv.URL, tt.model, 2)
			if err != nil {
				t.Fatalf("NewOllama: %v", err)
			}
			if _, err := p.Embed(context.Background(), []string{"текст"}, tt.kind); err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if want := tt.wantPrefix + "текст"; got.Input[0] != want {
				t.Errorf("sent %q, want %q", got.Input[0], want)
			}
		})
	}
}

func TestOllamaErrorClassification(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		body          string
		wantRetryable bool
		wantContains  string
	}{
		{"429 — ретраим", http.StatusTooManyRequests, "slow down", true, ""},
		{"500 — ретраим", http.StatusInternalServerError, "boom", true, ""},
		{"503 — ретраим", http.StatusServiceUnavailable, "loading", true, ""},
		{"400 — не ретраим, запрос сам виноват", http.StatusBadRequest, "bad input", false, ""},
		{"401 — не ретраим", http.StatusUnauthorized, "nope", false, ""},
		{"404 — подсказываем про ollama pull", http.StatusNotFound, "model not found", false, "ollama pull"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := respondWith(t, tt.status, tt.body, nil)
			p, err := NewOllama(srv.URL, "nomic-embed-text", 3)
			if err != nil {
				t.Fatalf("NewOllama: %v", err)
			}

			_, err = p.Embed(context.Background(), []string{"текст"}, KindDocument)
			if err == nil {
				t.Fatal("Embed() returned no error")
			}

			var r *backoff.Retryable
			if got := errors.As(err, &r); got != tt.wantRetryable {
				t.Errorf("retryable = %v, want %v (err: %v)", got, tt.wantRetryable, err)
			}
			if tt.wantContains != "" && !strings.Contains(err.Error(), tt.wantContains) {
				t.Errorf("error %q does not mention %q", err, tt.wantContains)
			}
		})
	}
}

// A model that quietly returns a different width than the column expects would
// fail on insert at best, and at worst be inserted into a fresh table nobody
// meant to create.
func TestOllamaRejectsWrongDimension(t *testing.T) {
	srv := respondWith(t, http.StatusOK, vectorsJSON([]float32{1, 2, 3, 4}), nil)

	p, err := NewOllama(srv.URL, "nomic-embed-text", 768)
	if err != nil {
		t.Fatalf("NewOllama: %v", err)
	}
	if _, err := p.Embed(context.Background(), []string{"текст"}, KindDocument); err == nil {
		t.Fatal("Embed() accepted a 4-dimension vector for a 768-dimension space")
	}
}

func TestOllamaRejectsShortBatch(t *testing.T) {
	srv := respondWith(t, http.StatusOK, vectorsJSON([]float32{1, 2}), nil)

	p, err := NewOllama(srv.URL, "nomic-embed-text", 2)
	if err != nil {
		t.Fatalf("NewOllama: %v", err)
	}
	if _, err := p.Embed(context.Background(), []string{"один", "два"}, KindDocument); err == nil {
		t.Fatal("Embed() accepted 1 vector for 2 inputs")
	}
}

func TestOllamaUnreachableServerIsRetryable(t *testing.T) {
	// A port nothing listens on: connection refused is a transport error, the
	// classic case where a second attempt often just works.
	p, err := NewOllama("http://127.0.0.1:1", "nomic-embed-text", 3)
	if err != nil {
		t.Fatalf("NewOllama: %v", err)
	}

	_, err = p.Embed(context.Background(), []string{"текст"}, KindDocument)
	if err == nil {
		t.Fatal("Embed() against a dead port returned no error")
	}
	var r *backoff.Retryable
	if !errors.As(err, &r) {
		t.Errorf("connection failure was not marked retryable: %v", err)
	}
}

func TestOllamaRejectsOversizedBatch(t *testing.T) {
	p, err := NewOllama("http://127.0.0.1:1", "nomic-embed-text", 3)
	if err != nil {
		t.Fatalf("NewOllama: %v", err)
	}
	texts := make([]string, ollamaMaxBatch+1)
	if _, err := p.Embed(context.Background(), texts, KindDocument); err == nil {
		t.Fatal("Embed() accepted a batch above the declared limit")
	}
}
