package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/Zinoki12/rag-ai-system/internal/backoff"
)

// ProviderGoogle is the value of LLM_PROVIDER that selects this client.
const ProviderGoogle = "google"

type googleProvider struct {
	client *genai.Client
	model  string
}

// NewGoogle builds a generation provider backed by the Gemini API.
func NewGoogle(ctx context.Context, apiKey, model string) (Provider, error) {
	if apiKey == "" {
		return nil, errors.New("google generation: API key is empty (set GOOGLE_API_KEY)")
	}
	if strings.TrimSpace(model) == "" {
		return nil, errors.New("google generation: model is empty")
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
		// Generation is slower than embedding and the answer arrives in one
		// piece, so the ceiling is higher — but there is still a ceiling.
		HTTPClient: &http.Client{Timeout: 120 * time.Second},
	})
	if err != nil {
		return nil, fmt.Errorf("google generation: init client: %w", err)
	}
	return &googleProvider{client: client, model: model}, nil
}

func (g *googleProvider) Model() string { return g.model }

func (g *googleProvider) Generate(ctx context.Context, system, user string) (string, error) {
	cfg := &genai.GenerateContentConfig{
		// Near-zero temperature: the job is to restate what the retrieved
		// context says, and invention is the failure mode this whole pipeline
		// exists to avoid.
		Temperature: genai.Ptr(float32(0.1)),
	}
	if system != "" {
		cfg.SystemInstruction = genai.NewContentFromText(system, genai.RoleUser)
	}

	resp, err := g.client.Models.GenerateContent(ctx, g.model,
		[]*genai.Content{genai.NewContentFromText(user, genai.RoleUser)}, cfg)
	if err != nil {
		return "", classifyGoogleErr(err)
	}

	text := strings.TrimSpace(resp.Text())
	if text == "" {
		return "", fmt.Errorf("google generation: model %s returned an empty answer", g.model)
	}
	return text, nil
}

func classifyGoogleErr(err error) error {
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		wrapped := fmt.Errorf("google generation: api error %d: %w", apiErr.Code, err)
		if backoff.IsTransientStatus(apiErr.Code) {
			return backoff.Mark(wrapped, 0)
		}
		return wrapped
	}
	return backoff.MarkTransport(fmt.Errorf("google generation: %w", err))
}
