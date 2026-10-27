package backoff

import (
	"strings"
	"time"
)

// GoogleHint reads what a Google API error says about retrying it, from the
// structured details its JSON error body carries next to the status code.
//
// Two entries matter. google.rpc.RetryInfo is that API family's Retry-After:
// the server's own figure for when the quota refills, which on the Gemini free
// tier is the rest of the current minute — well past anything an exponential
// backoff would guess, so without it every retry of a 429 is refused again.
// google.rpc.QuotaFailure names the quota that ran out, and a per-day one does
// not come back within any retry worth making: retrying it only holds the
// caller for minutes before the same refusal.
//
// Both are best effort. Absent or unreadable details give zero and false,
// which leaves the decision to the status code and the policy, as before.
func GoogleHint(details []map[string]any) (after time.Duration, daily bool) {
	for _, d := range details {
		switch typ, _ := d["@type"].(string); {
		case strings.HasSuffix(typ, "google.rpc.RetryInfo"):
			// A protobuf Duration in JSON: "36s", "1.5s".
			if s, ok := d["retryDelay"].(string); ok {
				if v, err := time.ParseDuration(s); err == nil && v > 0 {
					after = v
				}
			}
		case strings.HasSuffix(typ, "google.rpc.QuotaFailure"):
			violations, _ := d["violations"].([]any)
			for _, v := range violations {
				m, _ := v.(map[string]any)
				if id, _ := m["quotaId"].(string); strings.Contains(id, "PerDay") {
					daily = true
				}
			}
		}
	}
	return after, daily
}
