package storage

import (
	"regexp"
	"strings"
	"testing"

	"github.com/pgvector/pgvector-go"

	"github.com/Zinoki12/rag-ai-system/embed"
)

// testRef builds a SpaceRef without a database. The table field is unexported
// precisely so production code cannot do this; a test in the same package can,
// which is the point of keeping the type here.
func testRef(dim int) SpaceRef {
	return SpaceRef{
		ID:    1,
		Space: embed.Space{Provider: "fake", Model: "hashing-trick", Dim: dim},
		table: "emb_fake__hashing_trick__4_abcd1234",
	}
}

func TestBuildEmbeddingInsertNumbersPlaceholdersInOrder(t *testing.T) {
	ref := testRef(2)
	batch := []ChunkVector{
		{ChunkID: 7, Vector: embed.Vector{0.1, 0.2}},
		{ChunkID: 8, Vector: embed.Vector{0.3, 0.4}},
		{ChunkID: 9, Vector: embed.Vector{0.5, 0.6}},
	}

	q, args, err := buildEmbeddingInsert(ref, batch)
	if err != nil {
		t.Fatalf("buildEmbeddingInsert: %v", err)
	}

	// The failure this guards against is silent: placeholders that drift by one
	// still produce a statement Postgres accepts, and it stores each vector
	// against the neighbouring chunk. Nothing errors; search just quietly
	// returns the wrong passages.
	wantValues := "($1, $2), ($3, $4), ($5, $6)"
	if !strings.Contains(q, wantValues) {
		t.Errorf("statement = %q, want it to contain %q", q, wantValues)
	}
	if len(args) != 6 {
		t.Fatalf("got %d args, want 6", len(args))
	}
	for i, cv := range batch {
		if args[i*2] != cv.ChunkID {
			t.Errorf("arg %d = %v, want chunk id %d", i*2+1, args[i*2], cv.ChunkID)
		}
		vec, ok := args[i*2+1].(pgvector.Vector)
		if !ok {
			t.Fatalf("arg %d is %T, want pgvector.Vector", i*2+2, args[i*2+1])
		}
		if got := vec.Slice(); len(got) != 2 || got[0] != cv.Vector[0] || got[1] != cv.Vector[1] {
			t.Errorf("arg %d = %v, want %v", i*2+2, got, cv.Vector)
		}
	}

	// Re-running after a partial failure must overwrite, not skip.
	if !strings.Contains(q, "ON CONFLICT (chunk_id) DO UPDATE") {
		t.Errorf("statement %q does not upsert", q)
	}
	// The table name reaches the statement quoted.
	if !strings.Contains(q, `"emb_fake__hashing_trick__4_abcd1234"`) {
		t.Errorf("statement %q does not use the quoted table name", q)
	}
}

func TestBuildEmbeddingInsertSingleRow(t *testing.T) {
	q, args, err := buildEmbeddingInsert(testRef(2), []ChunkVector{{ChunkID: 1, Vector: embed.Vector{0, 1}}})
	if err != nil {
		t.Fatalf("buildEmbeddingInsert: %v", err)
	}
	if strings.Contains(q, "), (") {
		t.Errorf("statement %q has a stray row separator", q)
	}
	if len(args) != 2 {
		t.Errorf("got %d args, want 2", len(args))
	}
}

// A vector of the wrong length must be refused before it reaches the database.
// Postgres would reject it too, but only after the whole batch is on the wire,
// and its message names neither the chunk nor the expected dimension.
func TestBuildEmbeddingInsertRejectsWrongDimension(t *testing.T) {
	_, _, err := buildEmbeddingInsert(testRef(4), []ChunkVector{
		{ChunkID: 1, Vector: embed.Vector{0, 1, 2, 3}},
		{ChunkID: 2, Vector: embed.Vector{0, 1}},
	})
	if err == nil {
		t.Fatal("accepted a vector of the wrong dimension")
	}
	for _, want := range []string{"chunk 2", "dimension 2", "want 4"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not contain %q", err, want)
		}
	}
}

// Every identifier that reaches a statement comes from SpaceRef.Table, so this
// is the one place where quoting has to hold.
func TestSpaceRefTableIsQuoted(t *testing.T) {
	cases := []struct {
		name  string
		table string
		want  string
	}{
		{"обычное имя", "emb_google__gemini_001__768", `"emb_google__gemini_001__768"`},
		{"кавычка внутри", `emb_a"b`, `"emb_a""b"`},
		{"попытка инъекции", `x"; DROP TABLE notes; --`, `"x""; DROP TABLE notes; --"`},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ref := SpaceRef{table: c.table}
			if got := ref.Table(); got != c.want {
				t.Errorf("Table() = %s, want %s", got, c.want)
			}
		})
	}
}

// Slug is what actually produces those names, and it is the real defence: the
// quoting above is the second line. Nothing outside [a-z0-9_] may survive it.
func TestSlugIsSafeForDDL(t *testing.T) {
	safe := regexp.MustCompile(`^emb_[a-z0-9_]+$`)
	for _, s := range []embed.Space{
		{Provider: `x"; DROP TABLE notes; --`, Model: "m", Dim: 8},
		{Provider: "google", Model: "модель с пробелами", Dim: 768},
		{Provider: "ollama", Model: strings.Repeat("very-long-model-name", 10), Dim: 1024},
		{Provider: "a'b", Model: "c\\d", Dim: 1},
	} {
		if got := s.Slug(); !safe.MatchString(got) {
			t.Errorf("Slug() for %v = %q, which is not a plain identifier", s, got)
		}
	}
}
