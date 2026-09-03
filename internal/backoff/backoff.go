// Package backoff holds the retry policy shared by every outbound provider
// call — embeddings and generation alike.
//
// The design choice worth knowing: an error is retried only if it was
// explicitly marked with Mark. There is no heuristic that inspects error
// strings or guesses from types. A caller that forgets to mark a retryable
// failure gets one attempt instead of a silent retry storm, which is the safer
// direction to fail in.
package backoff

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"time"
)

// Retryable marks an error as worth another attempt.
type Retryable struct {
	Err error
	// After carries a server-supplied hint (an HTTP Retry-After header). Zero
	// means "no hint, use the policy's own backoff".
	After time.Duration
}

func (e *Retryable) Error() string { return e.Err.Error() }
func (e *Retryable) Unwrap() error { return e.Err }

// Mark wraps err so Policy.Do will retry it. after may be zero.
func Mark(err error, after time.Duration) error {
	if err == nil {
		return nil
	}
	return &Retryable{Err: err, After: after}
}

// Policy describes how many attempts to make and how long to wait between them.
type Policy struct {
	MaxAttempts int           // total attempts, not retries; must be >= 1
	Base        time.Duration // delay before the first retry
	Max         time.Duration // ceiling for a single delay
}

// Default is a conservative policy suitable for third-party HTTP APIs.
var Default = Policy{MaxAttempts: 4, Base: 500 * time.Millisecond, Max: 15 * time.Second}

func (p Policy) normalized() Policy {
	if p.MaxAttempts < 1 {
		p.MaxAttempts = 1
	}
	if p.Base <= 0 {
		p.Base = Default.Base
	}
	if p.Max <= 0 {
		p.Max = Default.Max
	}
	return p
}

// Do runs op until it succeeds, returns an unmarked error, or the attempts run
// out. The context is honoured while waiting, so a cancelled request stops
// immediately instead of sleeping out the remaining backoff.
func (p Policy) Do(ctx context.Context, op func(context.Context) error) error {
	p = p.normalized()

	var lastErr error
	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			if lastErr != nil {
				return fmt.Errorf("after %d attempt(s): %w", attempt-1, errors.Join(lastErr, err))
			}
			return err
		}

		err := op(ctx)
		if err == nil {
			return nil
		}
		lastErr = err

		var r *Retryable
		if !errors.As(err, &r) {
			return err // not retryable: surface it as-is
		}
		if attempt >= p.MaxAttempts {
			return fmt.Errorf("giving up after %d attempt(s): %w", attempt, err)
		}

		if err := sleep(ctx, p.delay(attempt, r.After)); err != nil {
			return fmt.Errorf("while backing off after %d attempt(s): %w", attempt, errors.Join(lastErr, err))
		}
	}
}

// delay implements full jitter: a uniform draw from [0, window). Spreading
// retries across the whole window is what stops a batch of callers that failed
// together from marching back in lockstep and failing together again.
func (p Policy) delay(attempt int, hint time.Duration) time.Duration {
	if hint > 0 {
		return min(hint, p.Max)
	}
	window := float64(p.Base) * math.Pow(2, float64(attempt-1))
	if window > float64(p.Max) {
		window = float64(p.Max)
	}
	return time.Duration(rand.Float64() * window)
}

func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
