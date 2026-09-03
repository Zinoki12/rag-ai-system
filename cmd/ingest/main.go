// Command ingest builds the search index from a markdown vault.
//
// It runs two stages. The first walks the vault and writes notes and their
// chunks, skipping files whose content hash is unchanged. The second fills in
// embeddings for whatever chunks are missing one in the configured vector
// space. The stages are deliberately independent: the second asks the database
// what is missing rather than acting on what the first just wrote, so switching
// embedding models recomputes everything without touching a single file, and an
// interrupted run resumes where it stopped.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Zinoki12/rag-ai-system/internal/chunk"
	"github.com/Zinoki12/rag-ai-system/internal/config"
	"github.com/Zinoki12/rag-ai-system/internal/embed"
	"github.com/Zinoki12/rag-ai-system/internal/migrate"
	"github.com/Zinoki12/rag-ai-system/internal/storage"
	"github.com/Zinoki12/rag-ai-system/internal/vault"
)

// chunkSize is in runes, not bytes and not the vector dimension. The two
// numbers are unrelated and 800 is kept clear of 768 so they cannot be confused
// for each other.
const chunkSize = 800

// pendingBatch is how many chunks are claimed from the database per round. The
// provider splits this further according to its own batch limit.
const pendingBatch = 128

func main() {
	// A cancellable context is what makes "resumable" true rather than
	// aspirational: Ctrl+C stops the loop between batches, and everything
	// already committed stays committed.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx); err != nil {
		log.Fatalf("ingest: %v", err)
	}
}

func run(ctx context.Context) error {
	vaultPath, err := config.Required("VAULT_PATH")
	if err != nil {
		return err
	}

	pool, err := storage.NewPool(ctx)
	if err != nil {
		return err
	}
	defer pool.Close()

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.Ping(pingCtx); err != nil {
		return fmt.Errorf("ping database: %w", err)
	}

	if err := migrate.Up(ctx, pool); err != nil {
		return err
	}

	store := storage.New(pool)

	if err := ingestNotes(ctx, store, vaultPath); err != nil {
		return err
	}
	return embedPending(ctx, store)
}

// ingestNotes writes every markdown file in the vault as a note plus its chunks.
func ingestNotes(ctx context.Context, store *storage.Store, vaultPath string) error {
	files, err := vault.Scan(vaultPath)
	if err != nil {
		// Scan reports two different things through one error: files it had to
		// skip, and a failure that stopped the walk. Only the second is fatal.
		if !errors.Is(err, vault.ErrUnclosedFrontmatter) {
			return fmt.Errorf("scan vault %s: %w", vaultPath, err)
		}
		log.Printf("пропущены файлы с битым фронтматтером:\n%v", err)
	}

	var (
		written, skipped int
		noteErrs         []error
	)
	for _, file := range files {
		if err := ctx.Err(); err != nil {
			return err
		}

		id, ok, err := store.SaveNoteWithChunks(ctx, file, func() ([]string, error) {
			return chunk.Cut(file.Text, chunkSize)
		})
		if err != nil {
			// One unwritable note must not cost the whole run. The failures are
			// collected and reported at the end so the exit code still says the
			// run was not clean.
			noteErrs = append(noteErrs, fmt.Errorf("note %s: %w", file.Path, err))
			continue
		}

		if ok {
			written++
			fmt.Printf("записана заметка [%s], id %d\n", file.Path, id)
		} else {
			skipped++
		}
	}

	fmt.Printf("\nЗаметки: записано %d, без изменений %d, с ошибками %d\n", written, skipped, len(noteErrs))
	return errors.Join(noteErrs...)
}

// embedPending fills in vectors for chunks that do not have one yet.
func embedPending(ctx context.Context, store *storage.Store) error {
	cfg, err := embed.ConfigFromEnv()
	if err != nil {
		return err
	}

	provider, err := embed.New(ctx, cfg)
	if err != nil {
		return err
	}

	ref, err := store.EnsureSpace(ctx, provider.Space())
	if err != nil {
		return err
	}

	before, err := store.Stats(ctx, ref)
	if err != nil {
		return err
	}
	fmt.Printf("\nПространство %s: %d чанков, векторов уже есть %d\n",
		ref.Space, before.Chunks, before.Embedded)

	done := 0
	for {
		pending, err := store.PendingChunks(ctx, ref, pendingBatch)
		if err != nil {
			return err
		}
		if len(pending) == 0 {
			break
		}

		texts := make([]string, len(pending))
		for i, c := range pending {
			texts[i] = c.Text
		}

		vectors, err := embed.Batched(ctx, provider, texts, embed.KindDocument)
		if err != nil {
			return fmt.Errorf("embed %d chunks: %w", len(texts), err)
		}

		batch := make([]storage.ChunkVector, len(pending))
		for i, c := range pending {
			batch[i] = storage.ChunkVector{ChunkID: c.ID, Vector: vectors[i]}
		}
		if err := store.SaveEmbeddings(ctx, ref, batch); err != nil {
			return err
		}

		done += len(pending)
		fmt.Printf("векторов посчитано %d из %d\n", before.Embedded+done, before.Chunks)
	}

	after, err := store.Stats(ctx, ref)
	if err != nil {
		return err
	}
	fmt.Printf("\nИтог: %d чанков, %d векторов в пространстве %s\n", after.Chunks, after.Embedded, ref.Space)
	return nil
}
