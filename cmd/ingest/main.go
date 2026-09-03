// Command ingest builds the search index from a markdown vault.
//
// It is a thin wrapper: everything it does is one call to knowledge.Index. The
// same call is what an application embedding this repository as a library
// makes, and what POST /reindex runs in the background.
package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/Zinoki12/rag-ai-system/knowledge"
)

func main() {
	// A cancellable context is what makes "resumable" true rather than
	// aspirational: Ctrl+C stops between batches and everything already
	// committed stays committed.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx); err != nil {
		log.Fatalf("ingest: %v", err)
	}
}

func run(ctx context.Context) error {
	cfg, err := knowledge.ConfigFromEnv()
	if err != nil {
		return err
	}
	if cfg.VaultPath == "" {
		return fmt.Errorf("VAULT_PATH is not set")
	}
	// Indexing never generates text, so a missing or misconfigured
	// LLM_PROVIDER must not be able to fail this command.
	cfg.Generator = nil
	cfg.Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	kb, err := knowledge.Open(ctx, cfg)
	if err != nil {
		return err
	}
	defer kb.Close()

	before, err := kb.Stats(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("Пространство %s: заметок %d, чанков %d, векторов %d\n",
		before.Space, before.Notes, before.Chunks, before.Embedded)

	res, indexErr := kb.Index(ctx)

	fmt.Printf("\nЗаметки: записано %d, без изменений %d, удалено %d, с ошибками %d\n",
		res.NotesWritten, res.NotesUnchanged, res.NotesDeleted, res.NotesFailed)
	fmt.Printf("Векторов посчитано: %d\n", res.ChunksEmbedded)

	after, err := kb.Stats(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("\nИтог: заметок %d, чанков %d, векторов %d в пространстве %s\n",
		after.Notes, after.Chunks, after.Embedded, after.Space)

	// Reported last so partial failures are visible but do not hide the counts.
	return indexErr
}
