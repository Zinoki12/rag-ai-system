package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

// ProviderOpenAI is the value of EMBED_PROVIDER that selects this client.
//
// It is not tied to OpenAI the company. The request and response shapes below
// are the de-facto interface that most hosted inference services expose, so one
// implementation plus a base URL reaches all of them — which matters when the
// question is "which provider will give me a free key that works from here"
// rather than "which vendor do we standardise on".
const ProviderOpenAI = "openai"

// openAIMaxBatch is conservative on purpose: hosted services differ in what
// they accept, and the failure for an over-long batch is usually a 400 that
// costs the whole run rather than a clear message.
const openAIMaxBatch = 64

const maxOpenAIBody = 64 << 20

// openAIPrefixes mirrors ollamaPrefixes. The OpenAI embeddings API has no
// task_type field, but several open models served through it were trained to
// read the task from the text. Dropping Kind silently would be exactly the
// quality regression Kind exists to prevent, so the well-known families are
// handled by name.
var openAIPrefixes = map[string]taskPrefix{
	"multilingual-e5-large": {document: "passage: ", query: "query: "},
	"multilingual-e5-base":  {document: "passage: ", query: "query: "},
	"multilingual-e5-small": {document: "passage: ", query: "query: "},
	"bge-m3":                {},
	"nomic-embed-text-v1.5": {document: "search_document: ", query: "search_query: "},
}

type openAIProvider struct {
	baseURL string
	apiKey  string
	http    *http.Client
	model   string
	dim     int
}

// NewOpenAI builds an embedding provider for any OpenAI-compatible endpoint.
//
// baseURL is the part before /embeddings, normally ending in /v1.
func NewOpenAI(baseURL, apiKey, model string, dim int) (Provider, error) {
	if strings.TrimSpace(baseURL) == "" {
		return nil, errors.New("openai embeddings: base URL is empty (set EMBED_BASE_URL)")
	}
	if strings.TrimSpace(model) == "" {
		return nil, errors.New("openai embeddings: model is empty (set EMBED_MODEL)")
	}
	if dim <= 0 {
		return nil, fmt.Errorf("openai embeddings: dimension %d must be positive", dim)
	}
	return &openAIProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 2 * time.Minute},
		model:   model,
		dim:     dim,
	}, nil
}

func (o *openAIProvider) Space() Space {
	return Space{Provider: ProviderOpenAI, Model: o.model, Dim: o.dim}
}

func (o *openAIProvider) MaxBatch() int { return openAIMaxBatch }

type openAIEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func (o *openAIProvider) Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if len(texts) > openAIMaxBatch {
		return nil, fmt.Errorf("openai embeddings: batch of %d exceeds %d", len(texts), openAIMaxBatch)
	}

	prefixed := make([]string, len(texts))
	prefix := openAIPrefixes[o.model]
	for i, t := range texts {
		switch kind {
		case KindQuery:
			prefixed[i] = prefix.query + t
		default:
			prefixed[i] = prefix.document + t
		}
	}

	body, err := json.Marshal(openAIEmbedRequest{Model: o.model, Input: prefixed})
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		o.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if o.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.apiKey)
	}

	resp, err := o.http.Do(req)
	if err != nil {
		return nil, backoff.MarkTransport(fmt.Errorf("openai embeddings: %w", err))
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxOpenAIBody))
	if err != nil {
		return nil, backoff.MarkTransport(fmt.Errorf("openai embeddings: read body: %w", err))
	}

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("openai embeddings: %s: %s",
			resp.Status, strings.TrimSpace(truncate(string(raw), 400)))
		if backoff.IsTransientStatus(resp.StatusCode) {
			return nil, backoff.Mark(err, backoff.ParseRetryAfter(resp.Header.Get("Retry-After")))
		}
		return nil, err
	}

	var parsed openAIEmbedResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("openai embeddings: decode response: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("openai embeddings: %s", parsed.Error.Message)
	}
	if len(parsed.Data) != len(texts) {
		return nil, fmt.Errorf("openai embeddings: got %d vectors for %d inputs",
			len(parsed.Data), len(texts))
	}

	// The spec says data comes back in input order, but it also carries an
	// index, and a mis-ordered batch would attach every vector to the wrong
	// chunk without any error. Sorting costs nothing and removes the question.
	sort.Slice(parsed.Data, func(i, j int) bool {
		return parsed.Data[i].Index < parsed.Data[j].Index
	})

	out := make([]Vector, len(parsed.Data))
	for i, d := range parsed.Data {
		if len(d.Embedding) != o.dim {
			return nil, fmt.Errorf(
				"openai embeddings: model %q returned dimension %d, configured %d — set EMBED_DIM=%d",
				o.model, len(d.Embedding), o.dim, len(d.Embedding))
		}
		out[i] = Vector(d.Embedding)
	}
	return out, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
