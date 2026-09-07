package llm

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

// ProviderOpenAI is the value of LLM_PROVIDER that selects this client.
//
// As in the embed package, this is the shape of the API rather than the vendor:
// one implementation plus a base URL reaches most hosted inference services.
const ProviderOpenAI = "openai"

const maxOpenAIBody = 32 << 20

type openAIProvider struct {
	baseURL string
	apiKey  string
	http    *http.Client
	model   string

	// extra carries per-service headers that are not part of the OpenAI shape,
	// such as the attribution pair OpenRouter reads. Kept as a map so adding a
	// broker does not mean adding a field and touching every constructor.
	extra map[string]string
}

// NewOpenAI builds a generation provider for any OpenAI-compatible endpoint.
// baseURL is the part before /chat/completions, normally ending in /v1.
func NewOpenAI(baseURL, apiKey, model string) (Provider, error) {
	return newOpenAIProvider(baseURL, apiKey, model, nil)
}

// newOpenAIProvider is the shared constructor. NewOpenAI and NewOpenRouter
// differ only in what they pass here.
func newOpenAIProvider(baseURL, apiKey, model string, extra map[string]string) (Provider, error) {
	if strings.TrimSpace(baseURL) == "" {
		return nil, errors.New("openai generation: base URL is empty (set LLM_BASE_URL)")
	}
	if strings.TrimSpace(model) == "" {
		return nil, errors.New("openai generation: model is empty (set LLM_MODEL)")
	}
	return &openAIProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		// Generous: a free tier under load can queue a request for a long time,
		// and failing early only turns a slow answer into no answer.
		http:  &http.Client{Timeout: 5 * time.Minute},
		model: model,
		extra: extra,
	}, nil
}

func (o *openAIProvider) Model() string { return o.model }

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChatRequest struct {
	Model    string          `json:"model"`
	Messages []openAIMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message      openAIMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func (o *openAIProvider) Generate(ctx context.Context, system, user string) (string, error) {
	messages := make([]openAIMessage, 0, 2)
	if strings.TrimSpace(system) != "" {
		messages = append(messages, openAIMessage{Role: "system", Content: system})
	}
	messages = append(messages, openAIMessage{Role: "user", Content: user})

	body, err := json.Marshal(openAIChatRequest{Model: o.model, Messages: messages})
	if err != nil {
		return "", fmt.Errorf("openai generation: encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		o.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("openai generation: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if o.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.apiKey)
	}
	for k, v := range o.extra {
		req.Header.Set(k, v)
	}

	resp, err := o.http.Do(req)
	if err != nil {
		return "", backoff.MarkTransport(fmt.Errorf("openai generation: %w", err))
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxOpenAIBody))
	if err != nil {
		return "", backoff.MarkTransport(fmt.Errorf("openai generation: read body: %w", err))
	}

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("openai generation: %s: %s",
			resp.Status, strings.TrimSpace(truncate(string(raw), 400)))
		if backoff.IsTransientStatus(resp.StatusCode) {
			return "", backoff.Mark(err, backoff.ParseRetryAfter(resp.Header.Get("Retry-After")))
		}
		return "", err
	}

	var parsed openAIChatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("openai generation: decode response: %w", err)
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("openai generation: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", errors.New("openai generation: response contained no choices")
	}

	text := strings.TrimSpace(parsed.Choices[0].Message.Content)
	if text == "" {
		// An empty answer with a stop reason is worth reporting rather than
		// passing up as a blank reply the operator has to explain.
		return "", fmt.Errorf("openai generation: empty answer (finish_reason=%q)",
			parsed.Choices[0].FinishReason)
	}
	return text, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
