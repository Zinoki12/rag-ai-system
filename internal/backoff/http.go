package backoff

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"time"
)

// IsTransientStatus reports whether an HTTP status is worth another attempt.
//
// 429 and 5xx are the server saying "later" or "my fault". Everything else in
// the 4xx range is the request itself being wrong — a bad key, a malformed
// body, a text over the token limit — and retrying it only burns quota and
// delays the error the caller needs to see.
func IsTransientStatus(code int) bool {
	return code == http.StatusTooManyRequests ||
		code == http.StatusRequestTimeout ||
		code >= 500
}

// ParseRetryAfter reads a Retry-After header in either of its two forms:
// delay-seconds, or an HTTP date. Returns zero when absent or unparseable.
func ParseRetryAfter(h string) time.Duration {
	if h == "" {
		return 0
	}
	if secs, err := strconv.Atoi(h); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(h); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// MarkTransport marks a transport-level failure as retryable.
//
// A refused connection or a dropped socket is the classic case where a second
// attempt often just works. Context cancellation is deliberately excluded: it
// means the caller gave up, and retrying would ignore that.
func MarkTransport(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return Mark(err, 0)
	}
	return err
}
