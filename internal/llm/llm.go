// Package llm is the connector for text generation models.
//
// It mirrors internal/embed deliberately: one interface, several vendors behind
// it, and the retry and rate-limit behaviour written once against the interface
// rather than once per client. The two packages stay separate because their
// call shapes have nothing in common, but they share internal/backoff and the
// same configuration idiom, so learning one teaches the other.
package llm

import "context"

// Provider generates text. Implementations must be safe for concurrent use.
type Provider interface {
	// Generate answers a user prompt under a system instruction.
	Generate(ctx context.Context, system, user string) (string, error)

	// Model reports which model answers, for logging and provenance.
	Model() string
}
