package storage

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pgvector/pgvector-go"

	"github.com/Zinoki12/rag-ai-system/internal/embed"
)

// Chunk is a stored piece of a note awaiting or carrying an embedding.
type Chunk struct {
	ID   int
	Text string
}

// ChunkVector pairs a chunk with its computed embedding.
type ChunkVector struct {
	ChunkID int
	Vector  embed.Vector
}

// PendingChunks returns chunks that have no vector in this space yet, oldest
// first, at most limit of them.
//
// Work is found by asking what is missing, not by remembering what was just
// written. That distinction is the whole design: the ingest pass skips notes
// whose file hash is unchanged, so an embedding stage triggered by "a note was
// written" would leave every untouched note carrying vectors from whatever
// model produced them — a silent mismatch across the index that no query
// reports. Asking the database what is missing makes a model switch recompute
// everything on its own, and makes an interrupted run resumable.
func (s *Store) PendingChunks(ctx context.Context, ref SpaceRef, limit int) ([]Chunk, error) {
	if limit < 1 {
		limit = 1
	}

	q := fmt.Sprintf(`
		SELECT c.id, c.chunk_text
		FROM chunks c
		LEFT JOIN %s e ON e.chunk_id = c.id
		WHERE e.chunk_id IS NULL
		ORDER BY c.id
		LIMIT $1`, ref.Table())

	rows, err := s.db.Query(ctx, q, limit)
	if err != nil {
		return nil, fmt.Errorf("pending chunks for %s: %w", ref, err)
	}
	defer rows.Close()

	var out []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.Text); err != nil {
			return nil, fmt.Errorf("pending chunks for %s: scan: %w", ref, err)
		}
		out = append(out, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pending chunks for %s: %w", ref, err)
	}
	return out, nil
}

// SaveEmbeddings writes a batch of vectors in a single statement.
//
// One multi-row INSERT rather than a loop: the loop costs a network round trip
// per chunk, which dominates everything else once the vault is more than a
// handful of notes. It is not pgx.Batch because that carries its own trap --
// the BatchResults must be closed before the transaction commits, and the first
// queued error only surfaces from that Close -- for a saving that a single
// statement already delivers.
func (s *Store) SaveEmbeddings(ctx context.Context, ref SpaceRef, batch []ChunkVector) error {
	if len(batch) == 0 {
		return nil
	}

	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	var (
		sb   strings.Builder
		args = make([]any, 0, len(batch)*2)
	)
	fmt.Fprintf(&sb, "INSERT INTO %s (chunk_id, embedding) VALUES ", ref.Table())
	for i, cv := range batch {
		if len(cv.Vector) != ref.Space.Dim {
			return fmt.Errorf("save embeddings for %s: chunk %d has dimension %d, want %d",
				ref, cv.ChunkID, len(cv.Vector), ref.Space.Dim)
		}
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(&sb, "($%d, $%d)", len(args)+1, len(args)+2)
		args = append(args, cv.ChunkID, pgvector.NewVector(cv.Vector))
	}
	// DO UPDATE rather than DO NOTHING: re-running after a partial failure
	// should overwrite, and a chunk whose text changed keeps its id.
	sb.WriteString(" ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, created_at = now()")

	if _, err := s.db.Exec(ctx, sb.String(), args...); err != nil {
		return fmt.Errorf("save %d embeddings for %s: %w", len(batch), ref, err)
	}
	return nil
}

// EmbeddingStats reports how much of the corpus this space covers.
type EmbeddingStats struct {
	Chunks   int
	Embedded int
}

// Stats counts chunks and how many of them have a vector in this space.
func (s *Store) Stats(ctx context.Context, ref SpaceRef) (EmbeddingStats, error) {
	q := fmt.Sprintf(
		"SELECT (SELECT count(*) FROM chunks), (SELECT count(*) FROM %s)", ref.Table())

	var st EmbeddingStats
	if err := s.db.QueryRow(ctx, q).Scan(&st.Chunks, &st.Embedded); err != nil {
		return EmbeddingStats{}, fmt.Errorf("stats for %s: %w", ref, err)
	}
	return st, nil
}
