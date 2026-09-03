package embed

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"time"

	"google.golang.org/genai"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

// ProviderGoogle is the value of EMBED_PROVIDER that selects this client.
const ProviderGoogle = "google"

// googleMaxBatch is the documented ceiling on inputs per embedContent call.
const googleMaxBatch = 100

// nativeDims records the full, untruncated output width of the models we know
// about. It is only used to decide whether a vector had to be truncated and
// therefore needs re-normalising; an unknown model is assumed to be exact.
var nativeDims = map[string]int{
	"gemini-embedding-001": 3072,
	"text-embedding-004":   768,
}

type googleProvider struct {
	client *genai.Client
	space  Space
}

// NewGoogle builds an embedding provider backed by the Gemini API.
func NewGoogle(ctx context.Context, apiKey, model string, dim int) (Provider, error) {
	if apiKey == "" {
		return nil, errors.New("google embeddings: API key is empty (set GOOGLE_API_KEY)")
	}
	space := Space{Provider: ProviderGoogle, Model: model, Dim: dim}
	if err := space.Validate(); err != nil {
		return nil, err
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
		// Never http.DefaultClient: it has no timeout, so a server that accepts
		// the connection and then stalls hangs the ingest run forever. This
		// timeout is the outer bound on one call; the context deadline can and
		// usually does cut in earlier.
		HTTPClient: &http.Client{Timeout: 60 * time.Second},
	})
	if err != nil {
		return nil, fmt.Errorf("google embeddings: init client: %w", err)
	}
	return &googleProvider{client: client, space: space}, nil
}

func (g *googleProvider) Space() Space  { return g.space }
func (g *googleProvider) MaxBatch() int { return googleMaxBatch }

func (g *googleProvider) Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if len(texts) > googleMaxBatch {
		return nil, fmt.Errorf("google embeddings: %d texts exceeds the batch limit of %d", len(texts), googleMaxBatch)
	}

	contents := make([]*genai.Content, 0, len(texts))
	for _, t := range texts {
		contents = append(contents, genai.NewContentFromText(t, genai.RoleUser))
	}

	cfg := &genai.EmbedContentConfig{TaskType: googleTaskType(kind)}
	// embedding-001 predates the parameter and rejects it outright.
	if g.space.Model != "embedding-001" {
		dim := int32(g.space.Dim)
		cfg.OutputDimensionality = &dim
	}

	resp, err := g.client.Models.EmbedContent(ctx, g.space.Model, contents, cfg)
	if err != nil {
		return nil, classifyGoogleErr(err)
	}

	out := make([]Vector, 0, len(texts))
	truncated := g.space.Dim < nativeDims[g.space.Model] // zero for unknown models
	for _, e := range resp.Embeddings {
		v := Vector(e.Values)
		if truncated {
			v = normalize(v)
		}
		out = append(out, v)
	}

	if err := checkResult(g.space, texts, out); err != nil {
		return nil, fmt.Errorf("google embeddings: %w", err)
	}
	return out, nil
}

func googleTaskType(k Kind) string {
	if k == KindQuery {
		return "RETRIEVAL_QUERY"
	}
	return "RETRIEVAL_DOCUMENT"
}

// normalize rescales a vector to unit length.
//
// Gemini normalises its full-width output but not a truncated one. Cosine
// distance is scale-invariant so pgvector's `<=>` would not care either way,
// but storing consistently normalised vectors keeps L2 and inner-product
// operators usable later without a full recompute.
func normalize(v Vector) Vector {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return v
	}
	out := make(Vector, len(v))
	for i, f := range v {
		out[i] = float32(float64(f) / norm)
	}
	return out
}

func classifyGoogleErr(err error) error {
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		wrapped := fmt.Errorf("google embeddings: api error %d: %w", apiErr.Code, err)
		if backoff.IsTransientStatus(apiErr.Code) {
			return backoff.Mark(wrapped, 0)
		}
		return wrapped
	}
	return backoff.MarkTransport(fmt.Errorf("google embeddings: %w", err))
}
