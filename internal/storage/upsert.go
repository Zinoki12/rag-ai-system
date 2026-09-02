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

func (s *Store) SaveNoteWithChunks(
	ctx context.Context,
	info *model.Note,
	chunker func() ([]string, error),
) (int, bool, error) {
	txCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()

	tx, err := s.db.Begin(txCtx)
	if err != nil {
		return 0, false, fmt.Errorf("cannot start transaction: %w", err)
	}

	defer func() {
		rollbackCtx, rollbackCancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer rollbackCancel()
		_ = tx.Rollback(rollbackCtx)
	}()

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

	err = tx.QueryRow(txCtx, query, info.Path, info.Name, info.Text, info.Hash[:]).Scan(&id, &written)
	if err != nil {
		return 0, false, fmt.Errorf("upsert note failed: %w", err)
	}

	if !written {
		if err := tx.Commit(txCtx); err != nil {
			return 0, false, fmt.Errorf("commit tx: %w", err)
		}
		return id, false, nil
	}

	chunks, err := chunker()
	if err != nil {
		return 0, false, fmt.Errorf("chunking failed for %s: %w", info.Path, err)
	}

	_, err = tx.Exec(txCtx, "DELETE FROM chunks WHERE note_id = $1", id)
	if err != nil {
		return 0, false, fmt.Errorf("failed to delete old chunks: %w", err)
	}

	for i, chunkText := range chunks {
		_, err = tx.Exec(txCtx,
			"INSERT INTO chunks (note_id, chunk_index, chunk_text) VALUES ($1, $2, $3)",
			id, i, chunkText,
		)
		if err != nil {
			return 0, false, fmt.Errorf("failed to insert chunk %d: %w", i, err)
		}
	}

	if err := tx.Commit(txCtx); err != nil {
		return 0, false, fmt.Errorf("commit full tx: %w", err)
	}

	return id, true, nil
}
