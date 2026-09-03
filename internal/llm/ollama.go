package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Zinoki12/rag-ai-system/internal/backoff"
	"github.com/Zinoki12/rag-ai-system/internal/embed"
)

// ProviderOllama is the value of LLM_PROVIDER that selects this client.
const ProviderOllama = "ollama"

const maxOllamaBody = 32 << 20

type ollamaProvider struct {
	baseURL string
	http    *http.Client
	model   string
}

// NewOllama builds a generation provider backed by a local Ollama server.
func NewOllama(baseURL, model string) (Provider, error) {
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("ollama generation: model is empty")
	}
	if baseURL == "" {
		baseURL = embed.DefaultOllamaURL
	}
	return &ollamaProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		// Generating on CPU is slow, and the first call also pays for loading
		// the weights. Generous, but still bounded.
		http:  &http.Client{Timeout: 10 * time.Minute},
		model: model,
	}, nil
}

func (o *ollamaProvider) Model() string { return o.model }

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  map[string]any  `json:"options,omitempty"`
}

type ollamaChatResponse struct {
	Message ollamaMessage `json:"message"`
}

func (o *ollamaProvider) Generate(ctx context.Context, system, user string) (string, error) {
	msgs := make([]ollamaMessage, 0, 2)
	if system != "" {
		msgs = append(msgs, ollamaMessage{Role: "system", Content: system})
	}
	msgs = append(msgs, ollamaMessage{Role: "user", Content: user})

	// stream:false matters: with streaming on, Ollama replies with a sequence
	// of JSON objects rather than one, and a plain Decode would silently read
	// only the first token.
	body, err := json.Marshal(ollamaChatRequest{
		Model:    o.model,
		Messages: msgs,
		Stream:   false,
		Options:  map[string]any{"temperature": 0.1},
	})
	if err != nil {
		return "", fmt.Errorf("ollama generation: encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("ollama generation: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.http.Do(req)
	if err != nil {
		return "", backoff.MarkTransport(fmt.Errorf("ollama generation: %w", err))
	}
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxOllamaBody))
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		err := fmt.Errorf("ollama generation: http %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
		if backoff.IsTransientStatus(resp.StatusCode) {
			return "", backoff.Mark(err, backoff.ParseRetryAfter(resp.Header.Get("Retry-After")))
		}
		if resp.StatusCode == http.StatusNotFound {
			return "", fmt.Errorf("%w (is the model pulled? try: docker compose exec ollama ollama pull %s)", err, o.model)
		}
		return "", err
	}

	var decoded ollamaChatResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxOllamaBody)).Decode(&decoded); err != nil {
		return "", fmt.Errorf("ollama generation: decode response: %w", err)
	}

	text := strings.TrimSpace(decoded.Message.Content)
	if text == "" {
		return "", fmt.Errorf("ollama generation: model %s returned an empty answer", o.model)
	}
	return text, nil
}
