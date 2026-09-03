// Command server exposes the knowledge base over HTTP.
//
//	GET  /health          index coverage and the active embedding space
//	POST /search          {"query": "...", "k": 5}      -> ranked chunks
//	POST /ask             {"question": "...", "k": 5}   -> answer + sources
//	POST /reindex         start a background indexing pass -> 202
//	GET  /reindex/status  progress of the last or current pass
//
// Set RAG_API_TOKEN to require "Authorization: Bearer <token>" on everything
// except /health. Without it the server refuses to bind anywhere but loopback,
// because /search and /ask hand out the contents of the knowledge base and
// /reindex spends money on the embedding provider.
package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/Zinoki12/rag-ai-system/internal/config"
	"github.com/Zinoki12/rag-ai-system/internal/httpapi"
	"github.com/Zinoki12/rag-ai-system/knowledge"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx); err != nil {
		log.Fatalf("server: %v", err)
	}
}

func run(ctx context.Context) error {
	// JSON to stderr: a server's logs are read by machines first and people
	// second, unlike the ingest CLI whose stdout is its user interface.
	logger := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	cfg, err := knowledge.ConfigFromEnv()
	if err != nil {
		return err
	}
	cfg.Logger = logger

	kb, err := knowledge.Open(ctx, cfg)
	if err != nil {
		return err
	}
	defer kb.Close()

	httpCfg := httpapi.DefaultConfig
	httpCfg.Addr = config.String("HTTP_ADDR", httpCfg.Addr)
	httpCfg.AuthToken = config.String("RAG_API_TOKEN", "")

	allowOpen, err := config.Bool("RAG_ALLOW_UNAUTHENTICATED", false)
	if err != nil {
		return err
	}
	httpCfg.AllowUnauthenticated = allowOpen

	return httpapi.Run(ctx, kb, kb.Space().String(), logger, httpCfg)
}
