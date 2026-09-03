// Command server exposes the knowledge base over HTTP.
//
//	GET  /health          index coverage and the active embedding space
//	POST /search          {"query": "...", "k": 5}      -> ranked chunks
//	POST /ask             {"question": "...", "k": 5}   -> answer + sources
//	POST /reindex         start a background indexing pass -> 202
//	GET  /reindex/status  progress of the last or current pass
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

	return httpapi.Run(ctx, kb, kb.Space().String(), logger, httpCfg)
}
