package llm

import (
	"context"
	"fmt"
)

// ProviderStub is the value of LLM_PROVIDER that selects the offline provider.
const ProviderStub = "stub"

// stubProvider answers without a model, by quoting the prompt it was given.
//
// It exists so the full ask pipeline — retrieval, prompt assembly, output —
// can be exercised in tests and offline with no API key and no multi-gigabyte
// download. It is honest about what it is: the answer says so, so nobody
// mistakes a stub run for a working model.
type stubProvider struct{}

// NewStub builds the offline provider.
func NewStub() Provider { return stubProvider{} }

func (stubProvider) Model() string { return "stub" }

func (stubProvider) Generate(ctx context.Context, _, user string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return fmt.Sprintf(
		"[генератор не подключён: LLM_PROVIDER=stub]\n\nНиже — найденный контекст без пересказа моделью.\n\n%s",
		user), nil
}
