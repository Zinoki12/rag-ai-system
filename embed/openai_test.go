package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

// serveEmbeddings stands in for any OpenAI-compatible endpoint and records what
// the client actually sent, so the test checks the wire format rather than the
// client's opinion of it.
func serveEmbeddings(t *testing.T, dim int, capture *openAIEmbedRequest) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("запрос ушёл на %s, ожидался /v1/embeddings", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("заголовок авторизации = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(capture); err != nil {
			t.Fatalf("тело запроса не разбирается: %v", err)
		}

		// Answer out of order on purpose: the spec promises input order, the
		// client must not depend on the promise.
		type item struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		items := make([]item, len(capture.Input))
		for i := range capture.Input {
			vec := make([]float32, dim)
			vec[0] = float32(i) // marks which input this vector belongs to
			items[len(capture.Input)-1-i] = item{Index: i, Embedding: vec}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": items})
	}))
}

func TestOpenAIEmbedSendsAndOrders(t *testing.T) {
	var got openAIEmbedRequest
	srv := serveEmbeddings(t, 4, &got)
	defer srv.Close()

	p, err := NewOpenAI(srv.URL+"/v1", "test-key", "some-model", 4)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}

	vectors, err := p.Embed(context.Background(), []string{"первый", "второй", "третий"}, KindDocument)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != 3 {
		t.Fatalf("получено %d векторов, ожидалось 3", len(vectors))
	}
	if got.Model != "some-model" {
		t.Errorf("в запросе model = %q", got.Model)
	}

	// The server answered in reverse; each vector carries its input index in
	// element 0. If the client trusted arrival order these would be 2,1,0.
	for i, v := range vectors {
		if v[0] != float32(i) {
			t.Errorf("вектор %d принадлежит входу %v — порядок не восстановлен", i, v[0])
		}
	}
}

// TestOpenAIEmbedAppliesTaskPrefix: the OpenAI embeddings API has no task_type,
// so for models trained to read the task from the text, dropping Kind is a
// silent quality regression — exactly what Kind exists to prevent.
func TestOpenAIEmbedAppliesTaskPrefix(t *testing.T) {
	for _, tc := range []struct {
		kind       Kind
		wantPrefix string
	}{
		{KindDocument, "passage: "},
		{KindQuery, "query: "},
	} {
		var got openAIEmbedRequest
		srv := serveEmbeddings(t, 4, &got)

		p, err := NewOpenAI(srv.URL+"/v1", "test-key", "multilingual-e5-large", 4)
		if err != nil {
			t.Fatalf("NewOpenAI: %v", err)
		}
		if _, err := p.Embed(context.Background(), []string{"текст"}, tc.kind); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		srv.Close()

		if len(got.Input) != 1 || !strings.HasPrefix(got.Input[0], tc.wantPrefix) {
			t.Errorf("kind=%v: отправлено %q, ожидался префикс %q", tc.kind, got.Input, tc.wantPrefix)
		}
	}
}

// TestOpenAIEmbedRejectsWrongDimension: a model that returns 1024 floats when
// the space is declared as 768 must fail loudly. Silently accepting would write
// vectors the database rejects later, far from the cause.
func TestOpenAIEmbedRejectsWrongDimension(t *testing.T) {
	var got openAIEmbedRequest
	srv := serveEmbeddings(t, 1024, &got)
	defer srv.Close()

	p, err := NewOpenAI(srv.URL+"/v1", "test-key", "some-model", 768)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}

	_, err = p.Embed(context.Background(), []string{"текст"}, KindDocument)
	if err == nil {
		t.Fatal("несовпадение размерности принято молча")
	}
	// The message has to name the number to set, or the operator is left
	// guessing which of the two values is the right one.
	if !strings.Contains(err.Error(), "EMBED_DIM=1024") {
		t.Errorf("ошибка не подсказывает нужное значение: %v", err)
	}
}

func TestOpenAIEmbedNeedsBaseURLAndModel(t *testing.T) {
	if _, err := NewOpenAI("", "k", "m", 768); err == nil {
		t.Error("пустой base URL принят")
	}
	if _, err := NewOpenAI("http://x/v1", "k", "", 768); err == nil {
		t.Error("пустое имя модели принято")
	}
}

// TestOpenAIEmbedMarksRateLimitRetryable: free tiers answer 429 constantly, and
// a 429 that is not marked retryable turns a pause into a failed run.
func TestOpenAIEmbedMarksRateLimitRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "2")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit"}}`))
	}))
	defer srv.Close()

	p, err := NewOpenAI(srv.URL+"/v1", "k", "m", 4)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	_, err = p.Embed(context.Background(), []string{"t"}, KindDocument)
	if err == nil {
		t.Fatal("429 не привёл к ошибке")
	}

	var marked *backoff.Retryable
	if !errors.As(err, &marked) {
		t.Fatalf("429 не помечен как повторяемый: %v", err)
	}
	if marked.After != 2*time.Second {
		t.Errorf("подсказка Retry-After не прочитана: %v", marked.After)
	}
}
