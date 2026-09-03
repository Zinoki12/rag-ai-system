package knowledge_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/Zinoki12/rag-ai-system/embed"
	"github.com/Zinoki12/rag-ai-system/internal/testdb"
	"github.com/Zinoki12/rag-ai-system/knowledge"
)

// openKB builds a knowledge base on a throwaway database with the offline
// embedder, so the whole pipeline runs with no API key and no downloaded model.
func openKB(t *testing.T, vault string) *knowledge.Knowledge {
	t.Helper()

	_, dsn := testdb.New(t)
	kb, err := knowledge.Open(context.Background(), knowledge.Config{
		DSN:       dsn,
		VaultPath: vault,
		Embedder:  embed.Config{Provider: embed.ProviderFake, Dim: 16},
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(kb.Close)
	return kb
}

func writeNote(t *testing.T, root, name, content string) string {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return path
}

// Breaking a note's frontmatter must not delete it from the index.
//
// The chain that used to do exactly that: Scan skips the unparseable file, the
// skipped file is absent from the paths handed to the prune, the prune reads
// that absence as "the file is gone", and the note leaves along with its chunks
// and vectors — while the file sits untouched in the vault and Index returns a
// nil error. Deleting the closing --- is the likeliest hand edit there is, and
// the vault is open in Obsidian.
func TestIndexKeepsNotesWhoseFrontmatterBroke(t *testing.T) {
	vault := t.TempDir()
	writeNote(t, vault, "a.md", "---\ntitle: Первая\n---\nТело первой заметки.")
	writeNote(t, vault, "b.md", "---\ntitle: Вторая\n---\nТело второй заметки.")

	kb := openKB(t, vault)
	ctx := context.Background()

	first, err := kb.Index(ctx)
	if err != nil {
		t.Fatalf("first Index: %v", err)
	}
	if first.NotesWritten != 2 || first.NotesDeleted != 0 {
		t.Fatalf("first pass = %+v, want 2 written and 0 deleted", first)
	}

	// The edit: the closing --- disappears.
	writeNote(t, vault, "b.md", "---\ntitle: Вторая\nТело второй заметки.")

	second, err := kb.Index(ctx)
	if err != nil {
		t.Fatalf("second Index: %v", err)
	}
	if second.NotesDeleted != 0 {
		t.Errorf("second pass deleted %d notes; the file is still in the vault", second.NotesDeleted)
	}
	if second.NotesSkipped != 1 {
		t.Errorf("NotesSkipped = %d, want 1: the caller has to be able to see this", second.NotesSkipped)
	}

	stats, err := kb.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if stats.Notes != 2 {
		t.Errorf("index holds %d notes, want 2: the broken note keeps its last good version", stats.Notes)
	}

	// And the previously indexed content is still searchable.
	hits, err := kb.Search(ctx, "Тело второй заметки", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	var found bool
	for _, h := range hits {
		if h.Source == "b.md" {
			found = true
		}
	}
	if !found {
		t.Error("the broken note is no longer searchable")
	}
}

// A note whose file really is gone must leave, along with its vectors.
func TestIndexDeletesNotesWhoseFileIsGone(t *testing.T) {
	vault := t.TempDir()
	writeNote(t, vault, "a.md", "---\ntitle: Первая\n---\nТело.")
	path := writeNote(t, vault, "b.md", "---\ntitle: Вторая\n---\nТело.")

	kb := openKB(t, vault)
	ctx := context.Background()

	if _, err := kb.Index(ctx); err != nil {
		t.Fatalf("first Index: %v", err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatalf("remove: %v", err)
	}

	res, err := kb.Index(ctx)
	if err != nil {
		t.Fatalf("second Index: %v", err)
	}
	if res.NotesDeleted != 1 {
		t.Errorf("NotesDeleted = %d, want 1", res.NotesDeleted)
	}

	stats, err := kb.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if stats.Notes != 1 {
		t.Errorf("index holds %d notes, want 1", stats.Notes)
	}
	if stats.Chunks != stats.Embedded {
		t.Errorf("chunks=%d embedded=%d: vectors and chunks drifted apart", stats.Chunks, stats.Embedded)
	}
}

func TestIndexIsIdempotent(t *testing.T) {
	vault := t.TempDir()
	writeNote(t, vault, "a.md", "---\ntitle: Первая\n---\nТело первой заметки.")
	writeNote(t, vault, "nested/b.md", "---\ntitle: Вторая\n---\nТело второй заметки.")

	kb := openKB(t, vault)
	ctx := context.Background()

	first, err := kb.Index(ctx)
	if err != nil {
		t.Fatalf("first Index: %v", err)
	}
	second, err := kb.Index(ctx)
	if err != nil {
		t.Fatalf("second Index: %v", err)
	}

	if second.NotesWritten != 0 || second.NotesUnchanged != first.NotesWritten {
		t.Errorf("second pass = %+v, want everything unchanged", second)
	}
	// Not one call to the embedding provider on an unchanged vault. On a paid
	// provider this is the difference between a re-run costing nothing and
	// costing the whole corpus again.
	if second.ChunksEmbedded != 0 {
		t.Errorf("second pass embedded %d chunks, want 0", second.ChunksEmbedded)
	}
}

func TestAskWithoutGeneratorIsAnExplicitError(t *testing.T) {
	kb := openKB(t, t.TempDir())
	if _, err := kb.Ask(context.Background(), "вопрос", 5); err != knowledge.ErrNoGenerator {
		t.Errorf("Ask = %v, want ErrNoGenerator", err)
	}
}

// Migrations must be safe to replay: an existing installation re-runs them once
// when the version table moves, and every command in the project migrates on
// startup.
func TestOpenTwiceOnTheSameDatabase(t *testing.T) {
	_, dsn := testdb.New(t)
	cfg := knowledge.Config{DSN: dsn, VaultPath: t.TempDir(), Embedder: embed.Config{Provider: embed.ProviderFake, Dim: 16}}

	for i := range 2 {
		kb, err := knowledge.Open(context.Background(), cfg)
		if err != nil {
			t.Fatalf("Open #%d: %v", i+1, err)
		}
		kb.Close()
	}
}
