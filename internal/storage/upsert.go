package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/Zinoki12/rag-ai-system/internal/model"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	db *pgxpool.Pool
}

func New(pool *pgxpool.Pool) *Store {
	return &Store{db: pool}
}

func (s *Store) UpsertNote(ctx context.Context, info *model.Note) (int, bool, error) {
	queryCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	query := `WITH inserted_row AS (
		INSERT INTO notes (paths, name, text, hash)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (paths)
		DO UPDATE SET
			name = EXCLUDED.name,
			text = EXCLUDED.text,
			hash = EXCLUDED.hash,
			updated = now()
		WHERE notes.hash IS DISTINCT FROM EXCLUDED.hash
		RETURNING id
	)
	SELECT id, true AS written FROM inserted_row
	UNION ALL 
	SELECT id, false AS written FROM notes WHERE paths = $1
	ORDER BY written DESC
	LIMIT 1;`

	var (
		id      int
		written bool
	)

	err := s.db.QueryRow(queryCtx, query, info.Path, info.Name, info.Text, info.Hash[:]).Scan(&id, &written)
	if err != nil {
		return 0, false, fmt.Errorf("insert failed: %w", err)
	}
	return id, written, nil
}
