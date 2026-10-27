package llm

import (
	"errors"
	"testing"
	"time"

	"google.golang.org/genai"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

// Generation reads the same hints as embeddings: the free tier counts both per
// minute and per day.
func TestClassifyGoogleErrReadsTheQuotaHints(t *testing.T) {
	minute := genai.APIError{Code: 429, Details: []map[string]any{
		{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"},
	}}
	var r *backoff.Retryable
	if err := classifyGoogleErr(minute); !errors.As(err, &r) || r.After != 17*time.Second {
		t.Fatalf("минутная квота: %v, want retryable after 17s", err)
	}

	daily := genai.APIError{Code: 429, Details: []map[string]any{{
		"@type":      "type.googleapis.com/google.rpc.QuotaFailure",
		"violations": []any{map[string]any{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}},
	}}}
	if err := classifyGoogleErr(daily); errors.As(err, &r) {
		t.Fatalf("суточная квота: %v, want it not retried", err)
	}
}
