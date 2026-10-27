package backoff

import (
	"testing"
	"time"
)

// The shape is what the Gemini API actually returns with a 429: a QuotaFailure
// naming the quota, a Help link, and a RetryInfo with the delay.
func geminiDetails(quotaID, delay string) []map[string]any {
	return []map[string]any{
		{
			"@type": "type.googleapis.com/google.rpc.QuotaFailure",
			"violations": []any{map[string]any{
				"quotaMetric": "generativelanguage.googleapis.com/embed_content_free_tier_requests",
				"quotaId":     quotaID,
				"quotaValue":  "100",
			}},
		},
		{"@type": "type.googleapis.com/google.rpc.Help", "links": []any{}},
		{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
	}
}

func TestGoogleHint(t *testing.T) {
	tests := []struct {
		name      string
		details   []map[string]any
		wantAfter time.Duration
		wantDaily bool
	}{
		{"минутная квота: задержка из RetryInfo",
			geminiDetails("EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier", "36s"), 36 * time.Second, false},
		{"дробные секунды",
			geminiDetails("EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier", "1.5s"), 1500 * time.Millisecond, false},
		{"суточная квота распознаётся",
			geminiDetails("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier", "36s"), 36 * time.Second, true},
		{"без деталей — ноль", nil, 0, false},
		{"нечитаемая задержка — ноль",
			[]map[string]any{{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "скоро"}}, 0, false},
		{"задержка не строкой — ноль",
			[]map[string]any{{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": 36}}, 0, false},
		{"отрицательная задержка — ноль",
			[]map[string]any{{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "-5s"}}, 0, false},
		{"кривые violations не роняют разбор",
			[]map[string]any{{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": "PerDay"}}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			after, daily := GoogleHint(tt.details)
			if after != tt.wantAfter || daily != tt.wantDaily {
				t.Fatalf("GoogleHint() = %v, %v; want %v, %v", after, daily, tt.wantAfter, tt.wantDaily)
			}
		})
	}
}

// The default policy must wait out a minute-long quota window when the server
// asks for it; capping the delay at 15 s made every such retry fail again.
func TestDefaultHonoursAMinuteLongHint(t *testing.T) {
	for _, hint := range []time.Duration{36 * time.Second, 59 * time.Second} {
		if got := Default.delay(1, hint); got != hint {
			t.Errorf("Default.delay(1, %v) = %v, want the hint itself", hint, got)
		}
	}
	if got := Default.delay(1, time.Hour); got != Default.Max {
		t.Errorf("Default.delay(1, 1h) = %v, want the ceiling %v", got, Default.Max)
	}
}
