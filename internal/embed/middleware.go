package embed

import (
	"context"
	"fmt"

	"golang.org/x/time/rate"

	"github.com/Zinoki12/rag-ai-system/internal/backoff"
)

// The decorators below are the part of this package that makes it a connector
// rather than two API clients. Retries and rate limiting are written once,
// against the interface, and every provider — present or future — inherits
// them.

type retryProvider struct {
	next   Provider
	policy backoff.Policy
}

// WithRetry retries transient failures with exponential backoff and jitter.
// Which failures count as transient is decided by the wrapped provider, which
// marks them with backoff.Mark; a bad API key or a malformed request is not
// retried.
func WithRetry(p Provider, policy backoff.Policy) Provider {
	return &retryProvider{next: p, policy: policy}
}

func (r *retryProvider) Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error) {
	var out []Vector
	err := r.policy.Do(ctx, func(ctx context.Context) error {
		var err error
		out, err = r.next.Embed(ctx, texts, kind)
		return err
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (r *retryProvider) Space() Space  { return r.next.Space() }
func (r *retryProvider) MaxBatch() int { return r.next.MaxBatch() }

type limitedProvider struct {
	next Provider
	lim  *rate.Limiter
}

// WithRateLimit caps how often the wrapped provider is called.
//
// A token bucket rather than a sleep between calls: a sleep paces a single
// sequential loop and nothing else, while a limiter also holds when several
// goroutines share one provider, and it releases immediately when the bucket
// has spare capacity instead of always paying the full delay.
func WithRateLimit(p Provider, rps float64, burst int) Provider {
	if rps <= 0 {
		return p
	}
	if burst < 1 {
		burst = 1
	}
	return &limitedProvider{next: p, lim: rate.NewLimiter(rate.Limit(rps), burst)}
}

func (l *limitedProvider) Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if err := l.lim.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit wait: %w", err)
	}
	return l.next.Embed(ctx, texts, kind)
}

func (l *limitedProvider) Space() Space  { return l.next.Space() }
func (l *limitedProvider) MaxBatch() int { return l.next.MaxBatch() }

// Batched embeds any number of texts by splitting them across as many provider
// calls as MaxBatch requires, preserving input order.
//
// Callers should always go through this instead of calling Embed directly with
// a slice of unknown length: exceeding a provider's batch limit is a runtime
// error from the API, and the limit differs per provider.
func Batched(ctx context.Context, p Provider, texts []string, kind Kind) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	size := p.MaxBatch()
	if size < 1 {
		size = 1
	}

	out := make([]Vector, 0, len(texts))
	for start := 0; start < len(texts); start += size {
		end := min(start+size, len(texts))

		part, err := p.Embed(ctx, texts[start:end], kind)
		if err != nil {
			return nil, fmt.Errorf("embed batch [%d:%d) of %d: %w", start, end, len(texts), err)
		}
		if len(part) != end-start {
			return nil, fmt.Errorf("embed batch [%d:%d): got %d vectors for %d inputs", start, end, len(part), end-start)
		}
		out = append(out, part...)
	}
	return out, nil
}
