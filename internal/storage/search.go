package storage

import (
	"context"
	"fmt"

	"github.com/pgvector/pgvector-go"

	"github.com/Zinoki12/rag-ai-system/internal/embed"
)

// Hit is one retrieved chunk with the note it came from.
type Hit struct {
	ChunkID    int
	ChunkIndex int
	Text       string
	NotePath   string
	NoteName   string
	Score      float64 // cosine similarity in [-1, 1]; higher is closer
}

// Search returns the topK chunks closest to v within one embedding space.
//
// The space is part of the query rather than a filter applied afterwards
// because vectors live in per-space tables: there is no way to accidentally
// compare a vector against one produced by a different model, which would
// return a confident ranking of noise.
func (s *Store) Search(ctx context.Context, ref SpaceRef, v embed.Vector, topK int) ([]Hit, error) {
	if len(v) != ref.Space.Dim {
		return nil, fmt.Errorf("search %s: query vector has dimension %d, want %d", ref, len(v), ref.Space.Dim)
	}
	if topK < 1 {
		topK = 1
	}

	// `<=>` is cosine distance in pgvector: 0 for identical direction, 2 for
	// opposite. Subtracting from 1 turns it back into the similarity people
	// expect to read. ORDER BY uses the raw distance so the hnsw index applies.
	q := fmt.Sprintf(`
		SELECT c.id, c.chunk_index, c.chunk_text, n.paths, n.name,
		       1 - (e.embedding <=> $1) AS score
		FROM %s e
		JOIN chunks c ON c.id = e.chunk_id
		JOIN notes  n ON n.id = c.note_id
		ORDER BY e.embedding <=> $1
		LIMIT $2`, ref.Table())

	rows, err := s.db.Query(ctx, q, pgvector.NewVector(v), topK)
	if err != nil {
		return nil, fmt.Errorf("search %s: %w", ref, err)
	}
	defer rows.Close()

	var out []Hit
	for rows.Next() {
		var h Hit
		var name *string // notes.name is nullable when the note had no title
		if err := rows.Scan(&h.ChunkID, &h.ChunkIndex, &h.Text, &h.NotePath, &name, &h.Score); err != nil {
			return nil, fmt.Errorf("search %s: scan: %w", ref, err)
		}
		if name != nil {
			h.NoteName = *name
		}
		out = append(out, h)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search %s: %w", ref, err)
	}
	return out, nil
}
