package llm

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

func TestOpenAIGenerateSendsSystemAndUser(t *testing.T) {
	var got openAIChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("запрос ушёл на %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer k" {
			t.Errorf("заголовок авторизации = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("тело не разбирается: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"  ответ  "}}]}`))
	}))
	defer srv.Close()

	p, err := NewOpenAI(srv.URL+"/v1", "k", "some-model")
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}

	answer, err := p.Generate(context.Background(), "ты помощник", "вопрос")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if answer != "ответ" {
		t.Errorf("ответ = %q, ожидался обрезанный по краям", answer)
	}
	if len(got.Messages) != 2 ||
		got.Messages[0].Role != "system" || got.Messages[0].Content != "ты помощник" ||
		got.Messages[1].Role != "user" || got.Messages[1].Content != "вопрос" {
		t.Errorf("сообщения отправлены неверно: %+v", got.Messages)
	}
	if got.Stream {
		t.Error("stream=true — ответ пришёл бы кусками, а клиент их не собирает")
	}
}

// TestOpenAIGenerateOmitsEmptySystem: some endpoints reject an empty system
// message outright, so it must not be sent at all rather than sent blank.
func TestOpenAIGenerateOmitsEmptySystem(t *testing.T) {
	var got openAIChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	p, _ := NewOpenAI(srv.URL+"/v1", "k", "m")
	if _, err := p.Generate(context.Background(), "   ", "вопрос"); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Errorf("пустая системная роль всё-таки отправлена: %+v", got.Messages)
	}
}

// TestOpenAIGenerateReportsEmptyAnswer: a blank reply with a stop reason is a
// fact the operator needs, not a blank page to puzzle over.
func TestOpenAIGenerateReportsEmptyAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""},"finish_reason":"length"}]}`))
	}))
	defer srv.Close()

	p, _ := NewOpenAI(srv.URL+"/v1", "k", "m")
	_, err := p.Generate(context.Background(), "s", "u")
	if err == nil {
		t.Fatal("пустой ответ принят как успех")
	}
	if !strings.Contains(err.Error(), "length") {
		t.Errorf("причина остановки не названа: %v", err)
	}
}

// TestOpenAIGenerateSurfacesProviderError: an endpoint that answers 200 with an
// error object must not look like success.
func TestOpenAIGenerateSurfacesProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"error":{"message":"model not found","type":"invalid_request_error"}}`))
	}))
	defer srv.Close()

	p, _ := NewOpenAI(srv.URL+"/v1", "k", "m")
	_, err := p.Generate(context.Background(), "s", "u")
	if err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Errorf("ошибка провайдера не доведена до вызывающего: %v", err)
	}
}

func TestOpenAIGenerateMarksRateLimitRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "3")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	p, _ := NewOpenAI(srv.URL+"/v1", "k", "m")
	_, err := p.Generate(context.Background(), "s", "u")

	var marked *backoff.Retryable
	if !errors.As(err, &marked) {
		t.Fatalf("429 не помечен как повторяемый: %v", err)
	}
	if marked.After != 3*time.Second {
		t.Errorf("Retry-After не прочитан: %v", marked.After)
	}
}

// A 400 is the operator's mistake and retrying it only wastes the quota.
func TestOpenAIGenerateDoesNotRetryBadRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"unknown model"}}`))
	}))
	defer srv.Close()

	p, _ := NewOpenAI(srv.URL+"/v1", "k", "m")
	_, err := p.Generate(context.Background(), "s", "u")

	var marked *backoff.Retryable
	if errors.As(err, &marked) {
		t.Errorf("400 помечен как повторяемый: %v", err)
	}
	if err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Errorf("сообщение сервера не дошло до оператора: %v", err)
	}
}
