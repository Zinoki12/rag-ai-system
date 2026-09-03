package llm

import (
	"context"
	"fmt"

	"golang.org/x/time/rate"

	"github.com/Zinoki12/rag-ai-system/internal/backoff"
)

type retryProvider struct {
	next   Provider
	policy backoff.Policy
}

// WithRetry retries transient failures with exponential backoff and jitter.
// What counts as transient is decided by the wrapped provider via backoff.Mark.
func WithRetry(p Provider, policy backoff.Policy) Provider {
	return &retryProvider{next: p, policy: policy}
}

func (r *retryProvider) Generate(ctx context.Context, system, user string) (string, error) {
	var out string
	err := r.policy.Do(ctx, func(ctx context.Context) error {
		var err error
		out, err = r.next.Generate(ctx, system, user)
		return err
	})
	if err != nil {
		return "", err
	}
	return out, nil
}

func (r *retryProvider) Model() string { return r.next.Model() }

type limitedProvider struct {
	next Provider
	lim  *rate.Limiter
}

// WithRateLimit caps how often the wrapped provider is called.
func WithRateLimit(p Provider, rps float64, burst int) Provider {
	if rps <= 0 {
		return p
	}
	if burst < 1 {
		burst = 1
	}
	return &limitedProvider{next: p, lim: rate.NewLimiter(rate.Limit(rps), burst)}
}

func (l *limitedProvider) Generate(ctx context.Context, system, user string) (string, error) {
	if err := l.lim.Wait(ctx); err != nil {
		return "", fmt.Errorf("rate limit wait: %w", err)
	}
	return l.next.Generate(ctx, system, user)
}

func (l *limitedProvider) Model() string { return l.next.Model() }
