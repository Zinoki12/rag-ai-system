// Package app wires the pieces together.
//
// Every command needs the same things in the same order — a pool, migrations,
// an embedding provider, the space that provider writes into — and the ask
// pipeline itself is identical whether it is driven from a CLI or an HTTP
// handler. Assembling that once here is what keeps cmd/* down to argument
// parsing and output formatting.
package app

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/Zinoki12/rag-ai-system/internal/embed"
	"github.com/Zinoki12/rag-ai-system/internal/llm"
	"github.com/Zinoki12/rag-ai-system/internal/migrate"
	"github.com/Zinoki12/rag-ai-system/internal/rag"
	"github.com/Zinoki12/rag-ai-system/internal/storage"
)

// App holds the assembled dependencies.
type App struct {
	Pool  *pgxpool.Pool
	Store *storage.Store
	Embed embed.Provider
	Space storage.SpaceRef
	LLM   llm.Provider // nil when Options.WithLLM is false
}

// Options selects which parts to build.
type Options struct {
	// WithLLM also builds a generation provider. Commands that only index or
	// search leave it off so a missing LLM_PROVIDER cannot fail their startup.
	WithLLM bool

	// Logger receives startup diagnostics, migrations included. Defaults to a
	// plain-text handler on stderr, which suits a CLI; the server passes a JSON
	// one.
	Logger *slog.Logger
}

// Open connects, migrates and builds the configured providers.
func Open(ctx context.Context, opts Options) (*App, error) {
	if opts.Logger == nil {
		opts.Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	}

	pool, err := storage.NewPool(ctx)
	if err != nil {
		return nil, err
	}

	// From here on any failure must release the pool, or a command that exits
	// on a bad config leaves connections open until the process dies.
	ok := false
	defer func() {
		if !ok {
			pool.Close()
		}
	}()

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.Ping(pingCtx); err != nil {
		return nil, fmt.Errorf("ping database: %w", err)
	}

	if err := migrate.Up(ctx, pool, opts.Logger); err != nil {
		return nil, err
	}

	embedCfg, err := embed.ConfigFromEnv()
	if err != nil {
		return nil, err
	}
	embedder, err := embed.New(ctx, embedCfg)
	if err != nil {
		return nil, err
	}

	store := storage.New(pool)
	space, err := store.EnsureSpace(ctx, embedder.Space())
	if err != nil {
		return nil, err
	}

	a := &App{Pool: pool, Store: store, Embed: embedder, Space: space}

	if opts.WithLLM {
		llmCfg, err := llm.ConfigFromEnv()
		if err != nil {
			return nil, err
		}
		if a.LLM, err = llm.New(ctx, llmCfg); err != nil {
			return nil, err
		}
	}

	ok = true
	return a, nil
}

// Close releases the connection pool.
func (a *App) Close() {
	if a.Pool != nil {
		a.Pool.Close()
	}
}

// Stats reports index coverage for the active space.
func (a *App) Stats(ctx context.Context) (storage.EmbeddingStats, error) {
	return a.Store.Stats(ctx, a.Space)
}

// SpaceName names the active embedding space, for logs and responses.
func (a *App) SpaceName() string { return a.Space.Space.String() }

// Search embeds the query and returns the closest chunks.
func (a *App) Search(ctx context.Context, query string, topK int) ([]storage.Hit, error) {
	// KindQuery, not KindDocument. Embedding a question the way a stored
	// passage is embedded costs result quality and reports nothing.
	vectors, err := a.Embed.Embed(ctx, []string{query}, embed.KindQuery)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("embed query: got %d vectors for 1 input", len(vectors))
	}
	return a.Store.Search(ctx, a.Space, vectors[0], topK)
}

// Answer is a generated response with the material it was shown.
type Answer struct {
	Text     string
	Model    string
	Passages []rag.Passage
	Sources  []string

	// TopScore is the similarity of the best retrieved chunk.
	//
	// It is reported rather than acted on. A relevance floor below which the
	// question is answered "not in the knowledge base" without calling the
	// model would be a real improvement, but the cutoff differs per embedding
	// model and per corpus, and picking one against a five-note test vault
	// would be fitting a constant to a fixture. Surfacing the number lets a
	// reader judge, and lets the threshold be calibrated later against real
	// content.
	TopScore float64
}

// Ask runs the full pipeline: embed the question, retrieve, prompt, generate.
func (a *App) Ask(ctx context.Context, question string, topK int) (Answer, error) {
	if a.LLM == nil {
		return Answer{}, fmt.Errorf("ask: no generation provider configured")
	}

	hits, err := a.Search(ctx, question, topK)
	if err != nil {
		return Answer{}, err
	}

	passages := Passages(hits)
	prompt := rag.BuildUserPrompt(question, passages, rag.DefaultMaxContextRunes)

	text, err := a.LLM.Generate(ctx, rag.SystemPrompt, prompt)
	if err != nil {
		return Answer{}, err
	}

	answer := Answer{
		Text:     text,
		Model:    a.LLM.Model(),
		Passages: passages,
		Sources:  rag.Sources(passages),
	}
	if len(hits) > 0 {
		answer.TopScore = hits[0].Score // Search returns best first
	}
	return answer, nil
}

// Passages converts search hits into the prompt package's own type.
func Passages(hits []storage.Hit) []rag.Passage {
	out := make([]rag.Passage, 0, len(hits))
	for _, h := range hits {
		out = append(out, rag.Passage{
			Source: h.NotePath,
			Title:  h.NoteName,
			Index:  h.ChunkIndex,
			Text:   h.Text,
		})
	}
	return out
}
