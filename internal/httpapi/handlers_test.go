package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Zinoki12/rag-ai-system/internal/app"
	"github.com/Zinoki12/rag-ai-system/internal/rag"
	"github.com/Zinoki12/rag-ai-system/internal/storage"
)

// stubService stands in for the whole application stack.
type stubService struct {
	hits    []storage.Hit
	answer  app.Answer
	stats   storage.EmbeddingStats
	err     error
	lastK   int
	lastQ   string
	panicOn string // path prefix that should panic
}

func (s *stubService) Stats(context.Context) (storage.EmbeddingStats, error) {
	return s.stats, s.err
}

func (s *stubService) SpaceName() string { return "stub/test@4" }

func (s *stubService) Search(_ context.Context, query string, topK int) ([]storage.Hit, error) {
	s.lastQ, s.lastK = query, topK
	if s.panicOn == "search" {
		panic("boom")
	}
	return s.hits, s.err
}

func (s *stubService) Ask(_ context.Context, question string, topK int) (app.Answer, error) {
	s.lastQ, s.lastK = question, topK
	return s.answer, s.err
}

func newTestHandler(svc Service) http.Handler {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return Handler(svc, log, Config{RequestTimeout: 5 * time.Second})
}

func do(t *testing.T, h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	var r *http.Request
	if body == "" {
		r = httptest.NewRequest(method, path, nil)
	} else {
		r = httptest.NewRequest(method, path, strings.NewReader(body))
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestHealth(t *testing.T) {
	t.Run("здоровый сервис отдаёт покрытие индекса", func(t *testing.T) {
		svc := &stubService{stats: storage.EmbeddingStats{Chunks: 16, Embedded: 16}}
		w := do(t, newTestHandler(svc), http.MethodGet, "/health", "")

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		var got healthResponse
		if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if got.Chunks != 16 || got.Vector != 16 || got.Space != "stub/test@4" {
			t.Errorf("body = %+v, want 16/16 in stub/test@4", got)
		}
	})

	// A health check that returns 200 while the database is unreachable is
	// worse than none: it tells a load balancer to keep sending traffic.
	t.Run("недоступная база даёт 503", func(t *testing.T) {
		svc := &stubService{err: errors.New("connection refused")}
		w := do(t, newTestHandler(svc), http.MethodGet, "/health", "")

		if w.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503", w.Code)
		}
	})
}

func TestSearchValidation(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		body       string
		wantStatus int
	}{
		{"нормальный запрос", http.MethodPost, `{"query":"чанкинг","k":3}`, http.StatusOK},
		{"k не задан — берётся умолчание", http.MethodPost, `{"query":"чанкинг"}`, http.StatusOK},
		{"пустой query", http.MethodPost, `{"query":""}`, http.StatusBadRequest},
		{"query из пробелов", http.MethodPost, `{"query":"   "}`, http.StatusBadRequest},
		{"сломанный JSON", http.MethodPost, `{"query":`, http.StatusBadRequest},
		{"опечатка в имени поля", http.MethodPost, `{"quary":"чанкинг"}`, http.StatusBadRequest},
		{"пустое тело", http.MethodPost, ``, http.StatusBadRequest},
		{"GET вместо POST", http.MethodGet, ``, http.StatusMethodNotAllowed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := &stubService{hits: []storage.Hit{{ChunkID: 1, Text: "t", NotePath: "a.md", Score: 0.5}}}
			w := do(t, newTestHandler(svc), tt.method, "/search", tt.body)

			if w.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d (body: %s)", w.Code, tt.wantStatus, w.Body.String())
			}
		})
	}
}

// An unbounded k would let one request pull the entire table into a prompt.
func TestSearchClampsK(t *testing.T) {
	tests := []struct {
		name  string
		body  string
		wantK int
	}{
		{"ноль превращается в умолчание", `{"query":"q","k":0}`, 5},
		{"отрицательное превращается в умолчание", `{"query":"q","k":-3}`, 5},
		{"разумное значение проходит", `{"query":"q","k":7}`, 7},
		{"слишком большое обрезается", `{"query":"q","k":9999}`, 25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := &stubService{}
			do(t, newTestHandler(svc), http.MethodPost, "/search", tt.body)
			if svc.lastK != tt.wantK {
				t.Errorf("topK = %d, want %d", svc.lastK, tt.wantK)
			}
		})
	}
}

func TestSearchEmptyResultIsEmptyArray(t *testing.T) {
	// null would make a client iterating the field crash; [] would not.
	svc := &stubService{}
	w := do(t, newTestHandler(svc), http.MethodPost, "/search", `{"query":"нет такого"}`)

	if !strings.Contains(w.Body.String(), `"hits":[]`) {
		t.Errorf("body = %s, want an empty hits array", w.Body.String())
	}
}

// An upstream error message can carry an API key, a hostname or quota details.
// The client gets a category; the log gets the cause.
func TestUpstreamErrorIsNotLeakedToClient(t *testing.T) {
	secret := "api key AIzaSyTOTALLYSECRET rejected by embeddings.googleapis.com"
	svc := &stubService{err: errors.New(secret)}

	w := do(t, newTestHandler(svc), http.MethodPost, "/search", `{"query":"q"}`)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", w.Code)
	}
	if strings.Contains(w.Body.String(), "AIzaSy") || strings.Contains(w.Body.String(), "googleapis") {
		t.Errorf("response leaked upstream detail: %s", w.Body.String())
	}
}

func TestAsk(t *testing.T) {
	t.Run("успешный ответ несёт источники и модель", func(t *testing.T) {
		svc := &stubService{answer: app.Answer{
			Text:    "Чанкинг — это разрезание документа.",
			Model:   "llama3.2",
			Sources: []string{"rag/chunking.md"},
			Passages: []rag.Passage{
				{Source: "rag/chunking.md", Index: 0, Text: "..."},
			},
		}}
		w := do(t, newTestHandler(svc), http.MethodPost, "/ask", `{"question":"что такое чанкинг"}`)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
		}
		var got askResponse
		if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if got.Model != "llama3.2" || len(got.Sources) != 1 {
			t.Errorf("body = %+v, want model llama3.2 and one source", got)
		}
	})

	t.Run("пустой вопрос отвергается", func(t *testing.T) {
		w := do(t, newTestHandler(&stubService{}), http.MethodPost, "/ask", `{"question":"  "}`)
		if w.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", w.Code)
		}
	})
}

// A body larger than the limit must be refused rather than read into memory.
func TestOversizedBodyIsRejected(t *testing.T) {
	huge := `{"query":"` + strings.Repeat("я", maxRequestBody) + `"}`
	w := do(t, newTestHandler(&stubService{}), http.MethodPost, "/search", huge)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for an oversized body", w.Code)
	}
}

// net/http recovers panics per connection, but silently: the client just sees
// the connection die. The middleware must turn one into an answer.
func TestPanicBecomesFiveHundred(t *testing.T) {
	svc := &stubService{panicOn: "search"}
	w := do(t, newTestHandler(svc), http.MethodPost, "/search", `{"query":"q"}`)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", w.Code)
	}
	if strings.Contains(w.Body.String(), "boom") {
		t.Errorf("panic value leaked to the client: %s", w.Body.String())
	}
}

func TestRequestIDIsAlwaysPresent(t *testing.T) {
	t.Run("генерируется, если клиент не прислал", func(t *testing.T) {
		w := do(t, newTestHandler(&stubService{}), http.MethodGet, "/health", "")
		if w.Header().Get("X-Request-Id") == "" {
			t.Error("no X-Request-Id in the response")
		}
	})

	t.Run("клиентский идентификатор сохраняется сквозным", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/health", nil)
		r.Header.Set("X-Request-Id", "trace-me")
		w := httptest.NewRecorder()
		newTestHandler(&stubService{}).ServeHTTP(w, r)

		if got := w.Header().Get("X-Request-Id"); got != "trace-me" {
			t.Errorf("X-Request-Id = %q, want the client's own id", got)
		}
	})
}
