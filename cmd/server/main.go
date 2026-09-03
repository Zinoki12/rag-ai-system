// Command server exposes the knowledge base over HTTP.
//
//	GET  /health   index coverage and the active embedding space
//	POST /search   {"query": "...", "k": 5}      -> ranked chunks
//	POST /ask      {"question": "...", "k": 5}   -> generated answer + sources
package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/Zinoki12/rag-ai-system/internal/app"
	"github.com/Zinoki12/rag-ai-system/internal/config"
	"github.com/Zinoki12/rag-ai-system/internal/httpapi"
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

	a, err := app.Open(ctx, app.Options{WithLLM: true, Logger: logger})
	if err != nil {
		return err
	}
	defer a.Close()

	cfg := httpapi.DefaultConfig
	cfg.Addr = config.String("HTTP_ADDR", cfg.Addr)

	return httpapi.Run(ctx, a, logger, cfg)
}
