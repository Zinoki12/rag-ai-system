package embed

import (
	"context"
	"hash/fnv"
	"strings"
	"unicode"
)

// ProviderFake is the value of EMBED_PROVIDER that selects this client.
const ProviderFake = "fake"

// fakeProvider embeds text locally with the hashing trick: every token is
// hashed to a coordinate and a sign, and the vector is the signed sum,
// normalised.
//
// It exists so the whole pipeline — ingest, search, prompt assembly — can be
// exercised in tests and offline with no API key and no model download. It is
// not a semantic model: it captures word overlap and nothing else. That is
// still enough for an end-to-end test to distinguish a relevant chunk from an
// unrelated one, which is what such a test is actually asserting.
type fakeProvider struct {
	space Space
}

// NewFake builds a deterministic offline provider.
func NewFake(dim int) (Provider, error) {
	space := Space{Provider: ProviderFake, Model: "hashing-trick", Dim: dim}
	if err := space.Validate(); err != nil {
		return nil, err
	}
	return &fakeProvider{space: space}, nil
}

func (f *fakeProvider) Space() Space  { return f.space }
func (f *fakeProvider) MaxBatch() int { return 1024 }

// Embed ignores kind: a bag of words has no notion of a document/query
// asymmetry to model.
func (f *fakeProvider) Embed(ctx context.Context, texts []string, _ Kind) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	out := make([]Vector, len(texts))
	for i, text := range texts {
		v := make(Vector, f.space.Dim)
		for _, token := range tokenize(text) {
			h := fnv.New32a()
			_, _ = h.Write([]byte(token))
			sum := h.Sum32()

			idx := int(sum % uint32(f.space.Dim))
			// The top bit picks the sign. Signed contributions stop unrelated
			// tokens that collide on one coordinate from always reinforcing
			// each other, which is what makes the trick usable at small
			// dimensions.
			if sum&0x8000_0000 != 0 {
				v[idx]--
			} else {
				v[idx]++
			}
		}
		out[i] = normalize(v)
	}
	return out, nil
}

func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
