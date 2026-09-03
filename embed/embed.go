// Package embed is the connector for embedding models.
//
// Everything downstream — ingestion, search, the HTTP handlers — depends only
// on the Provider interface, never on a concrete vendor. Swapping a hosted API
// for a model running on the machine next door is an environment variable, not
// a code change.
package embed

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// Vector is a single embedding. Its length is always Space.Dim.
type Vector []float32

// Kind tells a provider what the text is going to be used for.
//
// This is a parameter rather than client configuration on purpose. Google (and
// most hosted models) embed a document and a search query into deliberately
// different corners of the space; using the document task type for a query
// degrades result quality without producing a single error or log line. Putting
// it in the signature means the compiler asks the question at every call site.
type Kind int

const (
	// KindDocument is for text being stored and later searched over.
	KindDocument Kind = iota
	// KindQuery is for a user's search string.
	KindQuery
)

func (k Kind) String() string {
	switch k {
	case KindDocument:
		return "document"
	case KindQuery:
		return "query"
	default:
		return fmt.Sprintf("Kind(%d)", int(k))
	}
}

// Space identifies a vector space: one (provider, model, dimension) triple.
//
// Vectors from two different spaces are not comparable. Cosine distance between
// them computes perfectly happily and the number it produces is meaningless, so
// a space is the unit at which vectors get stored and searched.
type Space struct {
	Provider string
	Model    string
	Dim      int
}

func (s Space) String() string {
	return fmt.Sprintf("%s/%s@%d", s.Provider, s.Model, s.Dim)
}

// MarshalText renders the space as "provider/model@dim", so it appears in JSON
// as the same one-line identity people read in logs rather than as a nested
// object.
func (s Space) MarshalText() ([]byte, error) { return []byte(s.String()), nil }

// Validate rejects a space that could not be stored or searched.
func (s Space) Validate() error {
	switch {
	case strings.TrimSpace(s.Provider) == "":
		return errors.New("embedding space: provider is empty")
	case strings.TrimSpace(s.Model) == "":
		return errors.New("embedding space: model is empty")
	case s.Dim < 1 || s.Dim > 16000:
		return fmt.Errorf("embedding space: dimension %d out of range 1..16000", s.Dim)
	}
	return nil
}

// maxSlugLen keeps the generated table name inside Postgres's 63-byte
// identifier limit with room to spare for the "_hnsw" index name suffix.
const maxSlugLen = 54

// digestLen is the length of the identity suffix every slug carries.
const digestLen = 8

// Slug renders the space as a safe SQL identifier fragment.
//
// The output is restricted to [a-z0-9_] by construction, which is the first of
// two defences around the one place in this project that builds an identifier
// at runtime (internal/storage.EnsureSpace also runs it through
// pgx.Identifier.Sanitize).
//
// Every slug ends in a digest of the exact space identity, and that suffix is
// not decoration. Sanitising is lossy in several ways at once: a name written
// in a non-Latin script reduces to nothing, "-" and "_" and "." all fold to the
// same character, and a long name gets truncated. Any of those can map two
// genuinely different models onto one table name — which would file two
// incompatible vector spaces into a single index and quietly ruin every search
// that touched it. The readable prefix is for humans reading \dt; the digest is
// what actually guarantees distinctness.
func (s Space) Slug() string {
	sum := sha256.Sum256([]byte(s.String()))
	suffix := "_" + hex.EncodeToString(sum[:])[:digestLen]

	full := fmt.Sprintf("emb_%s__%s__%d", sanitize(s.Provider), sanitize(s.Model), s.Dim)
	if len(full)+len(suffix) > maxSlugLen {
		full = full[:maxSlugLen-len(suffix)]
	}
	return full + suffix
}

func sanitize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	lastUnderscore := false
	for _, r := range strings.ToLower(s) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			lastUnderscore = false
		case !lastUnderscore:
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	return strings.Trim(b.String(), "_")
}

// Provider produces embeddings. Implementations must be safe for concurrent use.
type Provider interface {
	// Embed returns one vector per input text, in the same order. A nil or
	// empty input returns an empty result and no error.
	Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error)

	// Space reports which vector space this provider writes into.
	Space() Space

	// MaxBatch is the largest number of texts accepted by one Embed call.
	MaxBatch() int
}

// checkResult enforces the Provider contract on a concrete implementation's
// output.
//
// This is not paranoia about our own code: a provider that quietly returns
// 3072-dimension vectors when the column is 768, or drops an input from a
// batch, would corrupt the index in a way no later query reports as an error.
// Better to fail the ingest run loudly.
func checkResult(space Space, texts []string, got []Vector) error {
	if len(got) != len(texts) {
		return fmt.Errorf("%s: got %d vectors for %d inputs", space, len(got), len(texts))
	}
	for i, v := range got {
		if len(v) != space.Dim {
			return fmt.Errorf("%s: vector %d has dimension %d, want %d", space, i, len(v), space.Dim)
		}
	}
	return nil
}
