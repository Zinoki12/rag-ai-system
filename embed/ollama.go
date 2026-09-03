package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

// ProviderOllama is the value of EMBED_PROVIDER that selects this client.
const ProviderOllama = "ollama"

// DefaultOllamaURL matches the port published by the ollama service in
// docker-compose.yml.
const DefaultOllamaURL = "http://127.0.0.1:11434"

// ollamaMaxBatch is a self-imposed limit. Ollama accepts an arbitrarily long
// input array but processes it on one machine's CPU, so a huge batch turns into
// one very long request that is easy to time out and expensive to retry.
const ollamaMaxBatch = 32

// maxOllamaBody caps how much of a response we will read. A local server is
// trusted, but a body of unbounded size read into memory is a denial of service
// waiting for a misconfiguration.
const maxOllamaBody = 64 << 20

// taskPrefix holds the strings a model expects to be prepended so it can tell a
// stored passage from a search query.
//
// Hosted APIs take this as a structured task_type field; open models were
// trained to read it from the text itself, and omitting it is the same silent
// quality regression Kind exists to prevent — the vectors come back looking
// perfectly valid and rank badly.
type taskPrefix struct{ document, query string }

var ollamaPrefixes = map[string]taskPrefix{
	"nomic-embed-text":  {document: "search_document: ", query: "search_query: "},
	"mxbai-embed-large": {query: "Represent this sentence for searching relevant passages: "},
}

type ollamaProvider struct {
	baseURL string
	http    *http.Client
	space   Space
	prefix  taskPrefix
}

// NewOllama builds an embedding provider backed by a local Ollama server.
func NewOllama(baseURL, model string, dim int) (Provider, error) {
	space := Space{Provider: ProviderOllama, Model: model, Dim: dim}
	if err := space.Validate(); err != nil {
		return nil, err
	}
	if baseURL == "" {
		baseURL = DefaultOllamaURL
	}

	return &ollamaProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		// A local model on CPU is slow, especially on the first call when the
		// weights are still being loaded into memory, so this is generous. It
		// is still a bound: without one a wedged server hangs the run forever.
		http:   &http.Client{Timeout: 5 * time.Minute},
		space:  space,
		prefix: ollamaPrefixes[baseModelName(model)],
	}, nil
}

// baseModelName strips the tag from "nomic-embed-text:latest".
func baseModelName(model string) string {
	name, _, _ := strings.Cut(model, ":")
	return name
}

func (o *ollamaProvider) Space() Space  { return o.space }
func (o *ollamaProvider) MaxBatch() int { return ollamaMaxBatch }

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (o *ollamaProvider) Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if len(texts) > ollamaMaxBatch {
		return nil, fmt.Errorf("ollama embeddings: %d texts exceeds the batch limit of %d", len(texts), ollamaMaxBatch)
	}

	prefix := o.prefix.document
	if kind == KindQuery {
		prefix = o.prefix.query
	}
	input := make([]string, len(texts))
	for i, t := range texts {
		input[i] = prefix + t
	}

	body, err := json.Marshal(ollamaEmbedRequest{Model: o.space.Model, Input: input})
	if err != nil {
		return nil, fmt.Errorf("ollama embeddings: encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama embeddings: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.http.Do(req)
	if err != nil {
		return nil, backoff.MarkTransport(fmt.Errorf("ollama embeddings: %w", err))
	}
	defer func() {
		// Drain before closing so the connection can go back to the keep-alive
		// pool instead of being torn down after every call.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxOllamaBody))
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		err := fmt.Errorf("ollama embeddings: http %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
		if backoff.IsTransientStatus(resp.StatusCode) {
			return nil, backoff.Mark(err, backoff.ParseRetryAfter(resp.Header.Get("Retry-After")))
		}
		if resp.StatusCode == http.StatusNotFound {
			return nil, fmt.Errorf("%w (is the model pulled? try: docker compose exec ollama ollama pull %s)", err, o.space.Model)
		}
		return nil, err
	}

	var decoded ollamaEmbedResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxOllamaBody)).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("ollama embeddings: decode response: %w", err)
	}
	if len(decoded.Embeddings) == 0 {
		return nil, errors.New("ollama embeddings: response contained no embeddings")
	}

	out := make([]Vector, len(decoded.Embeddings))
	for i, v := range decoded.Embeddings {
		out[i] = Vector(v)
	}
	if err := checkResult(o.space, texts, out); err != nil {
		return nil, fmt.Errorf("ollama embeddings: %w", err)
	}
	return out, nil
}
