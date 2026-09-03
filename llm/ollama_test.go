package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

// fastRetry keeps the retrying tests quick without removing the waiting they
// exist to exercise.
var fastRetry = backoff.Policy{MaxAttempts: 3, Base: time.Millisecond, Max: 5 * time.Millisecond}

func chatReply(t *testing.T, text string) string {
	t.Helper()
	body, err := json.Marshal(ollamaChatResponse{Message: ollamaMessage{Role: "assistant", Content: text}})
	if err != nil {
		t.Fatalf("marshal reply: %v", err)
	}
	return string(body)
}

func TestOllamaGenerate(t *testing.T) {
	var got ollamaChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("path = %q, want /api/chat", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		io.WriteString(w, chatReply(t, "  Ответ модели.  "))
	}))
	defer srv.Close()

	p, err := NewOllama(srv.URL, "llama3.2")
	if err != nil {
		t.Fatalf("NewOllama: %v", err)
	}

	answer, err := p.Generate(context.Background(), "системная инструкция", "вопрос")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if answer != "Ответ модели." {
		t.Errorf("answer = %q, want it trimmed", answer)
	}

	// stream:false is load-bearing. With streaming on, Ollama replies with a
	// sequence of JSON objects and a single Decode reads only the first token —
	// an answer that looks plausible and is one word long.
	if got.Stream {
		t.Error("request had stream:true; the response decoder only reads one object")
	}
	if got.Model != "llama3.2" {
		t.Errorf("model = %q, want llama3.2", got.Model)
	}
	if len(got.Messages) != 2 ||
		got.Messages[0].Role != "system" || got.Messages[0].Content != "системная инструкция" ||
		got.Messages[1].Role != "user" || got.Messages[1].Content != "вопрос" {
		t.Errorf("messages = %+v, want system then user", got.Messages)
	}
}

func TestOllamaOmitsEmptySystemPrompt(t *testing.T) {
	var got ollamaChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		io.WriteString(w, chatReply(t, "ok"))
	}))
	defer srv.Close()

	p, _ := NewOllama(srv.URL, "llama3.2")
	if _, err := p.Generate(context.Background(), "", "вопрос"); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Errorf("messages = %+v, want just the user message", got.Messages)
	}
}

// An empty answer is a failure, not a valid result: a caller printing "" has no
// way to tell a silent model from a working one.
func TestOllamaRejectsEmptyAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, chatReply(t, "   "))
	}))
	defer srv.Close()

	p, _ := NewOllama(srv.URL, "llama3.2")
	if _, err := p.Generate(context.Background(), "", "вопрос"); err == nil {
		t.Fatal("Generate accepted an empty answer")
	}
}

// The classification, not the retrying, is what matters here: retrying a bad
// request burns quota and delays the error the caller needs to see, while not
// retrying a 503 turns a hiccup into a failure.
func TestOllamaRetryClassification(t *testing.T) {
	cases := []struct {
		name         string
		status       int
		wantAttempts int32
	}{
		{"429 повторяется", http.StatusTooManyRequests, 3},
		{"500 повторяется", http.StatusInternalServerError, 3},
		{"503 повторяется", http.StatusServiceUnavailable, 3},
		{"400 не повторяется", http.StatusBadRequest, 1},
		{"404 не повторяется", http.StatusNotFound, 1},
		{"422 не повторяется", http.StatusUnprocessableEntity, 1},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.WriteHeader(c.status)
				io.WriteString(w, `{"error":"нет"}`)
			}))
			defer srv.Close()

			base, _ := NewOllama(srv.URL, "llama3.2")
			p := WithRetry(base, fastRetry)

			if _, err := p.Generate(context.Background(), "", "вопрос"); err == nil {
				t.Fatal("Generate returned no error")
			}
			if got := calls.Load(); got != c.wantAttempts {
				t.Errorf("server saw %d attempts, want %d", got, c.wantAttempts)
			}
		})
	}
}

// A 404 from Ollama almost always means the model was never pulled. The error
// has to say that, because "http 404" sends people to look at the wrong thing.
func TestOllamaNotFoundNamesTheLikelyCause(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	p, _ := NewOllama(srv.URL, "llama3.2")
	_, err := p.Generate(context.Background(), "", "вопрос")
	if err == nil {
		t.Fatal("no error")
	}
	if !strings.Contains(err.Error(), "pull llama3.2") {
		t.Errorf("error = %q, want it to suggest pulling the model", err)
	}
}

// A refused connection is transient: the server may be starting up.
func TestOllamaTransportErrorIsRetried(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close() // nothing is listening now

	base, _ := NewOllama(url, "llama3.2")
	_, err := base.Generate(context.Background(), "", "вопрос")
	if err == nil {
		t.Fatal("no error from a closed server")
	}
	var retryable *backoff.Retryable
	if !errors.As(err, &retryable) {
		t.Errorf("error %v was not marked retryable", err)
	}
}

// A cancelled context must not be retried: the caller already gave up.
func TestOllamaHonoursCancelledContext(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		io.WriteString(w, chatReply(t, "ok"))
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	base, _ := NewOllama(srv.URL, "llama3.2")
	if _, err := WithRetry(base, fastRetry).Generate(ctx, "", "вопрос"); err == nil {
		t.Fatal("Generate ignored a cancelled context")
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("server saw %d requests, want 0", got)
	}
}

func TestNewOllamaValidatesModel(t *testing.T) {
	if _, err := NewOllama("http://127.0.0.1:11434", "  "); err == nil {
		t.Error("NewOllama accepted an empty model name")
	}
}
