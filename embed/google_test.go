package embed

import (
	"errors"
	"testing"
	"time"

	"google.golang.org/genai"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

func TestClassifyGoogleErr(t *testing.T) {
	retryInfo := map[string]any{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "42s"}
	quota := func(id string) map[string]any {
		return map[string]any{
			"@type":      "type.googleapis.com/google.rpc.QuotaFailure",
			"violations": []any{map[string]any{"quotaId": id}},
		}
	}

	tests := []struct {
		name      string
		err       error
		wantRetry bool
		wantAfter time.Duration
	}{
		{"429 минутной квоты ждёт, сколько сказал сервер",
			genai.APIError{Code: 429, Details: []map[string]any{quota("EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier"), retryInfo}},
			true, 42 * time.Second},
		{"429 суточной квоты не повторяется",
			genai.APIError{Code: 429, Details: []map[string]any{quota("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"), retryInfo}},
			false, 0},
		{"429 без деталей повторяется по политике", genai.APIError{Code: 429}, true, 0},
		{"503 повторяется", genai.APIError{Code: 503}, true, 0},
		{"400 не повторяется", genai.APIError{Code: 400}, false, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := classifyGoogleErr(tt.err)

			var r *backoff.Retryable
			if got := errors.As(err, &r); got != tt.wantRetry {
				t.Fatalf("retryable = %v, want %v (err: %v)", got, tt.wantRetry, err)
			}
			if tt.wantRetry && r.After != tt.wantAfter {
				t.Errorf("After = %v, want %v", r.After, tt.wantAfter)
			}
			var apiErr genai.APIError
			if !errors.As(err, &apiErr) {
				t.Errorf("the API error is no longer reachable through errors.As: %v", err)
			}
		})
	}
}
