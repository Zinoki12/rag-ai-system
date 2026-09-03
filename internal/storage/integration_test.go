package storage_test

import (
	"context"
	"crypto/sha256"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/Zinoki12/rag-ai-system/embed"
	"github.com/Zinoki12/rag-ai-system/internal/model"
	"github.com/Zinoki12/rag-ai-system/internal/storage"
	"github.com/Zinoki12/rag-ai-system/internal/testdb"
)

func newStore(t *testing.T) *storage.Store {
	t.Helper()
	pool, _ := testdb.New(t)
	return storage.New(pool)
}

func note(path, name, text string) *model.Note {
	return &model.Note{Path: path, Name: name, Text: text, Hash: sha256.Sum256([]byte(text))}
}

func chunker(parts ...string) func() ([]string, error) {
	return func() ([]string, error) { return parts, nil }
}

// unit is a vector pointing at one axis, so cosine similarity between two of
// them is 1 when the axis matches and 0 otherwise. That makes ranking
// assertions exact instead of approximate.
func unit(dim, axis int) embed.Vector {
	v := make(embed.Vector, dim)
	v[axis] = 1
	return v
}

func TestEnsureSpaceIsIdempotentAndIsolated(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	a := embed.Space{Provider: "fake", Model: "one", Dim: 4}
	b := embed.Space{Provider: "fake", Model: "two", Dim: 8}

	refA, err := s.EnsureSpace(ctx, a)
	if err != nil {
		t.Fatalf("EnsureSpace(a): %v", err)
	}
	again, err := s.EnsureSpace(ctx, a)
	if err != nil {
		t.Fatalf("EnsureSpace(a) twice: %v", err)
	}
	if again.ID != refA.ID || again.Table() != refA.Table() {
		t.Errorf("second EnsureSpace returned %v, want the same as %v", again, refA)
	}

	refB, err := s.EnsureSpace(ctx, b)
	if err != nil {
		t.Fatalf("EnsureSpace(b): %v", err)
	}
	// Two models must not share a table. This is the whole architecture: it is
	// what makes comparing vectors from different models structurally
	// impossible rather than a rule someone has to remember.
	if refB.Table() == refA.Table() {
		t.Fatalf("two spaces share the table %s", refA.Table())
	}

	spaces, err := s.Spaces(ctx)
	if err != nil {
		t.Fatalf("Spaces: %v", err)
	}
	if len(spaces) != 2 {
		t.Errorf("got %d registered spaces, want 2", len(spaces))
	}
}

func TestEnsureSpaceRejectsInvalid(t *testing.T) {
	s := newStore(t)
	if _, err := s.EnsureSpace(context.Background(), embed.Space{Provider: "fake", Model: "m", Dim: 0}); err == nil {
		t.Error("EnsureSpace accepted dimension 0")
	}
}

func TestPendingChunksFindsWhatIsMissing(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	space := embed.Space{Provider: "fake", Model: "m", Dim: 4}
	ref, err := s.EnsureSpace(ctx, space)
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}

	if _, _, err := s.SaveNoteWithChunks(ctx, note("a.md", "A", "тело"), chunker("один", "два", "три")); err != nil {
		t.Fatalf("SaveNoteWithChunks: %v", err)
	}

	pending, err := s.PendingChunks(ctx, ref, 100)
	if err != nil {
		t.Fatalf("PendingChunks: %v", err)
	}
	if len(pending) != 3 {
		t.Fatalf("got %d pending chunks, want 3", len(pending))
	}

	// Embed two of the three; the third must still be pending, and the two must
	// not come back. This is what makes an interrupted pass resumable.
	batch := []storage.ChunkVector{
		{ChunkID: pending[0].ID, Vector: unit(4, 0)},
		{ChunkID: pending[1].ID, Vector: unit(4, 1)},
	}
	if err := s.SaveEmbeddings(ctx, ref, batch); err != nil {
		t.Fatalf("SaveEmbeddings: %v", err)
	}

	pending, err = s.PendingChunks(ctx, ref, 100)
	if err != nil {
		t.Fatalf("PendingChunks after saving: %v", err)
	}
	if len(pending) != 1 {
		t.Fatalf("got %d pending chunks, want 1", len(pending))
	}

	// A second space sees everything as missing, with no command to remember.
	other, err := s.EnsureSpace(ctx, embed.Space{Provider: "fake", Model: "other", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace(other): %v", err)
	}
	pendingOther, err := s.PendingChunks(ctx, other, 100)
	if err != nil {
		t.Fatalf("PendingChunks(other): %v", err)
	}
	if len(pendingOther) != 3 {
		t.Errorf("a new space has %d pending chunks, want all 3 recomputed", len(pendingOther))
	}
}

func TestSaveEmbeddingsOverwrites(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	ref, err := s.EnsureSpace(ctx, embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	if _, _, err := s.SaveNoteWithChunks(ctx, note("a.md", "A", "тело"), chunker("текст")); err != nil {
		t.Fatalf("SaveNoteWithChunks: %v", err)
	}
	pending, err := s.PendingChunks(ctx, ref, 10)
	if err != nil || len(pending) != 1 {
		t.Fatalf("PendingChunks = %v, %v", pending, err)
	}
	id := pending[0].ID

	if err := s.SaveEmbeddings(ctx, ref, []storage.ChunkVector{{ChunkID: id, Vector: unit(4, 0)}}); err != nil {
		t.Fatalf("first SaveEmbeddings: %v", err)
	}
	// Writing the same chunk again must update rather than fail on the primary
	// key: a run that died mid-batch has to be safe to repeat.
	if err := s.SaveEmbeddings(ctx, ref, []storage.ChunkVector{{ChunkID: id, Vector: unit(4, 2)}}); err != nil {
		t.Fatalf("second SaveEmbeddings: %v", err)
	}

	hits, err := s.Search(ctx, ref, unit(4, 2), 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || math.Abs(hits[0].Score-1) > 1e-6 {
		t.Errorf("hits = %+v, want the overwritten vector to match exactly", hits)
	}
}

func TestSaveEmbeddingsEmptyBatch(t *testing.T) {
	s := newStore(t)
	ref, err := s.EnsureSpace(context.Background(), embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	if err := s.SaveEmbeddings(context.Background(), ref, nil); err != nil {
		t.Errorf("SaveEmbeddings(nil) = %v, want no error", err)
	}
}

func TestSearchRanksAndJoins(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	ref, err := s.EnsureSpace(ctx, embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	if _, _, err := s.SaveNoteWithChunks(ctx, note("dir/a.md", "Заголовок A", "тело"), chunker("ноль", "один")); err != nil {
		t.Fatalf("SaveNoteWithChunks: %v", err)
	}

	pending, err := s.PendingChunks(ctx, ref, 10)
	if err != nil {
		t.Fatalf("PendingChunks: %v", err)
	}
	for i, c := range pending {
		if err := s.SaveEmbeddings(ctx, ref, []storage.ChunkVector{{ChunkID: c.ID, Vector: unit(4, i)}}); err != nil {
			t.Fatalf("SaveEmbeddings: %v", err)
		}
	}

	hits, err := s.Search(ctx, ref, unit(4, 1), 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2", len(hits))
	}
	if hits[0].Text != "один" {
		t.Errorf("best hit = %q, want %q", hits[0].Text, "один")
	}
	if hits[0].NotePath != "dir/a.md" || hits[0].NoteName != "Заголовок A" {
		t.Errorf("hit did not carry its note: %+v", hits[0])
	}
	if hits[0].ChunkIndex != 1 {
		t.Errorf("ChunkIndex = %d, want 1", hits[0].ChunkIndex)
	}
	// Similarity, not distance: 1 for the same direction, 0 for orthogonal.
	if math.Abs(hits[0].Score-1) > 1e-6 || math.Abs(hits[1].Score) > 1e-6 {
		t.Errorf("scores = %v, %v; want 1 and 0", hits[0].Score, hits[1].Score)
	}
}

func TestSearchRejectsWrongDimension(t *testing.T) {
	s := newStore(t)
	ref, err := s.EnsureSpace(context.Background(), embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	if _, err := s.Search(context.Background(), ref, unit(8, 0), 5); err == nil {
		t.Error("Search accepted a query vector of the wrong dimension")
	}
}

// A note with no title stores NULL in notes.name; scanning that into a string
// would fail, so search results must survive it.
func TestSearchHandlesUntitledNote(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	ref, err := s.EnsureSpace(ctx, embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	if _, _, err := s.SaveNoteWithChunks(ctx, note("untitled.md", "", "тело"), chunker("текст")); err != nil {
		t.Fatalf("SaveNoteWithChunks: %v", err)
	}
	pending, _ := s.PendingChunks(ctx, ref, 10)
	if err := s.SaveEmbeddings(ctx, ref, []storage.ChunkVector{{ChunkID: pending[0].ID, Vector: unit(4, 0)}}); err != nil {
		t.Fatalf("SaveEmbeddings: %v", err)
	}

	hits, err := s.Search(ctx, ref, unit(4, 0), 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].NoteName != "" {
		t.Errorf("hits = %+v, want one hit with an empty name", hits)
	}
}

func TestSaveNoteWithChunksSkipsUnchanged(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	n := note("a.md", "A", "тело")
	calls := 0
	counting := func() ([]string, error) {
		calls++
		return []string{"один", "два"}, nil
	}

	id, written, err := s.SaveNoteWithChunks(ctx, n, counting)
	if err != nil || !written {
		t.Fatalf("first save: id=%d written=%v err=%v", id, written, err)
	}

	again, written, err := s.SaveNoteWithChunks(ctx, n, counting)
	if err != nil {
		t.Fatalf("second save: %v", err)
	}
	if written {
		t.Error("an unchanged note was rewritten")
	}
	if again != id {
		t.Errorf("note id changed from %d to %d", id, again)
	}
	// Chunking an unchanged note is wasted work on every pass, and on a large
	// vault it is the pass.
	if calls != 1 {
		t.Errorf("chunker ran %d times, want 1: an unchanged note must not be re-chunked", calls)
	}
}

func TestSaveNoteWithChunksReplacesChunks(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	ref, err := s.EnsureSpace(ctx, embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	if _, _, err := s.SaveNoteWithChunks(ctx, note("a.md", "A", "старое"), chunker("один", "два", "три")); err != nil {
		t.Fatalf("first save: %v", err)
	}
	if _, _, err := s.SaveNoteWithChunks(ctx, note("a.md", "A", "новое"), chunker("другое")); err != nil {
		t.Fatalf("second save: %v", err)
	}

	st, err := s.Stats(ctx, ref)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if st.Notes != 1 || st.Chunks != 1 {
		t.Errorf("Stats = %+v, want 1 note and 1 chunk", st)
	}
}

func TestSaveNoteWithChunksPropagatesChunkerError(t *testing.T) {
	s := newStore(t)
	sentinel := errors.New("не смог разрезать")
	_, _, err := s.SaveNoteWithChunks(context.Background(), note("a.md", "A", "тело"),
		func() ([]string, error) { return nil, sentinel })
	if !errors.Is(err, sentinel) {
		t.Errorf("err = %v, want it to wrap the chunker's error", err)
	}
}

func TestDeleteMissingNotesTakesVectorsWithIt(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	ref, err := s.EnsureSpace(ctx, embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	for _, p := range []string{"a.md", "b.md"} {
		if _, _, err := s.SaveNoteWithChunks(ctx, note(p, p, "тело "+p), chunker("один", "два")); err != nil {
			t.Fatalf("SaveNoteWithChunks(%s): %v", p, err)
		}
	}
	pending, _ := s.PendingChunks(ctx, ref, 10)
	for i, c := range pending {
		if err := s.SaveEmbeddings(ctx, ref, []storage.ChunkVector{{ChunkID: c.ID, Vector: unit(4, i%4)}}); err != nil {
			t.Fatalf("SaveEmbeddings: %v", err)
		}
	}

	deleted, err := s.DeleteMissingNotes(ctx, []string{"a.md"})
	if err != nil {
		t.Fatalf("DeleteMissingNotes: %v", err)
	}
	if deleted != 1 {
		t.Fatalf("deleted %d notes, want 1", deleted)
	}

	st, err := s.Stats(ctx, ref)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	// The vectors must go too. A vector left behind by a deleted note keeps
	// surfacing in search results that cite a path nobody can open.
	if st.Notes != 1 || st.Chunks != 2 || st.Embedded != 2 {
		t.Errorf("Stats = %+v, want 1 note, 2 chunks, 2 vectors", st)
	}
}

// A mistyped VAULT_PATH scans an empty directory perfectly successfully. The
// difference between that and a genuinely emptied vault is not visible from
// here, so the safe reading is the one that does not erase the index.
func TestDeleteMissingNotesRefusesEmptyKeepList(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	if _, _, err := s.SaveNoteWithChunks(ctx, note("a.md", "A", "тело"), chunker("один")); err != nil {
		t.Fatalf("SaveNoteWithChunks: %v", err)
	}
	deleted, err := s.DeleteMissingNotes(ctx, nil)
	if err != nil {
		t.Fatalf("DeleteMissingNotes: %v", err)
	}
	if deleted != 0 {
		t.Errorf("deleted %d notes on an empty keep list, want 0", deleted)
	}
}

// Two passages with the same text embed to the same vector and sit at exactly
// the same distance. With one sort key their order is whatever the executor
// happened to produce, so the same query can answer differently on consecutive
// runs; the chunk id breaks the tie.
func TestSearchOrderIsStableAcrossTies(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	ref, err := s.EnsureSpace(ctx, embed.Space{Provider: "fake", Model: "m", Dim: 4})
	if err != nil {
		t.Fatalf("EnsureSpace: %v", err)
	}
	for _, p := range []string{"a.md", "b.md", "c.md"} {
		if _, _, err := s.SaveNoteWithChunks(ctx, note(p, p, "тело "+p), chunker("одинаковый текст")); err != nil {
			t.Fatalf("SaveNoteWithChunks(%s): %v", p, err)
		}
	}
	pending, _ := s.PendingChunks(ctx, ref, 10)
	for _, c := range pending {
		if err := s.SaveEmbeddings(ctx, ref, []storage.ChunkVector{{ChunkID: c.ID, Vector: unit(4, 0)}}); err != nil {
			t.Fatalf("SaveEmbeddings: %v", err)
		}
	}

	var first []int
	for run := range 5 {
		hits, err := s.Search(ctx, ref, unit(4, 0), 3)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		ids := make([]int, len(hits))
		for i, h := range hits {
			ids[i] = h.ChunkID
		}
		if run == 0 {
			first = ids
			if !slices.IsSorted(ids) {
				t.Errorf("tied hits came back as %v, want them ordered by chunk id", ids)
			}
			continue
		}
		if !slices.Equal(ids, first) {
			t.Fatalf("run %d returned %v, run 1 returned %v", run+1, ids, first)
		}
	}
}
