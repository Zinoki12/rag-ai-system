package embed

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"google.golang.org/genai"
)

type Client struct {
	client    *genai.Client
	modelName string
}

func (c *Client) Model() string {
	return c.modelName
}

func New(ctx context.Context, apiKey string) (*Client, error) {
	tx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	client, err := genai.NewClient(tx, &genai.ClientConfig{
		APIKey:     apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: &http.Client{Timeout: 15 * time.Second},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize genai client: %w", err)
	}
	return &Client{client: client,
		modelName: "text-embedding-004"}, nil
}

func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	tx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	var content []*genai.Content
	content = append(content, genai.NewContentFromText(text, genai.RoleUser))
	result, err := c.client.Models.EmbedContent(tx, c.modelName, content, &genai.EmbedContentConfig{TaskType: "RETRIEVAL_DOCUMENT"})
	if err != nil {
		return nil, fmt.Errorf("failed to embed content: %w", err)
	}
	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("embed with %q: response contained no embeddings", c.modelName)
	}
	return result.Embeddings[0].Values, nil
}
