package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/Zinoki12/rag-ai-system/internal/embed"
)

// SpaceRef is a registered embedding space together with the physical table
// holding its vectors.
//
// The table name is unexported so it can only originate from EnsureSpace, which
// is the single place that validates it. A caller cannot hand a hand-built
// SpaceRef to a query.
type SpaceRef struct {
	ID    int16
	Space embed.Space
	table string
}

// Table returns the space's table name, already quoted for use in a statement.
func (r SpaceRef) Table() string { return pgx.Identifier{r.table}.Sanitize() }

func (r SpaceRef) String() string { return fmt.Sprintf("%s (%s)", r.Space, r.table) }

// EnsureSpace registers an embedding space and creates its storage.
//
// This is the only place in the project that puts a runtime-derived name into
// SQL, so it is worth being explicit about why that is safe here. The name
// comes from embed.Space.Slug, which emits nothing outside [a-z0-9_] and caps
// the length; it is then quoted through pgx.Identifier.Sanitize. Neither the
// model name nor anything else user-supplied can reach the statement intact.
// The dimension is an int checked against pgvector's limits before it is
// formatted.
func (s *Store) EnsureSpace(ctx context.Context, space embed.Space) (SpaceRef, error) {
	if err := space.Validate(); err != nil {
		return SpaceRef{}, err
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	tx, err := s.db.Begin(ctx)
	if err != nil {
		return SpaceRef{}, fmt.Errorf("ensure space %s: begin: %w", space, err)
	}
	defer func() {
		rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Second)
		defer cancel()
		_ = tx.Rollback(rollbackCtx)
	}()

	// Two commands starting at once is the normal case here, not an exotic one:
	// ingest and server both call this at boot. The lock is released when the
	// transaction ends, whichever way it ends.
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock(hashtext($1))", space.Slug()); err != nil {
		return SpaceRef{}, fmt.Errorf("ensure space %s: lock: %w", space, err)
	}

	// The no-op DO UPDATE exists only so RETURNING fires on the conflict path
	// too; DO NOTHING returns no row when the space already exists.
	const register = `
		INSERT INTO embedding_spaces (provider, model, dim, table_name)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (provider, model, dim)
		DO UPDATE SET table_name = embedding_spaces.table_name
		RETURNING id, table_name`

	var ref SpaceRef
	ref.Space = space
	err = tx.QueryRow(ctx, register, space.Provider, space.Model, space.Dim, space.Slug()).
		Scan(&ref.ID, &ref.table)
	if err != nil {
		return SpaceRef{}, fmt.Errorf("ensure space %s: register: %w", space, err)
	}

	quoted := pgx.Identifier{ref.table}.Sanitize()
	createTable := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			chunk_id   INT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
			embedding  vector(%d) NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`, quoted, space.Dim)
	if _, err := tx.Exec(ctx, createTable); err != nil {
		return SpaceRef{}, fmt.Errorf("ensure space %s: create table: %w", space, err)
	}

	// Cosine is the operator every query in this project uses, so the index has
	// to be built for it: an hnsw index on vector_l2_ops would simply not be
	// used by an ORDER BY <=> and the planner would fall back to a seq scan
	// without complaining.
	createIndex := fmt.Sprintf(
		"CREATE INDEX IF NOT EXISTS %s ON %s USING hnsw (embedding vector_cosine_ops)",
		pgx.Identifier{ref.table + "_hnsw"}.Sanitize(), quoted,
	)
	if _, err := tx.Exec(ctx, createIndex); err != nil {
		return SpaceRef{}, fmt.Errorf("ensure space %s: create index: %w", space, err)
	}

	if err := tx.Commit(ctx); err != nil {
		return SpaceRef{}, fmt.Errorf("ensure space %s: commit: %w", space, err)
	}
	return ref, nil
}

// Spaces lists every registered embedding space, newest first.
func (s *Store) Spaces(ctx context.Context) ([]SpaceRef, error) {
	const q = `
		SELECT id, provider, model, dim, table_name
		FROM embedding_spaces
		ORDER BY created_at DESC, id DESC`

	rows, err := s.db.Query(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("list embedding spaces: %w", err)
	}
	defer rows.Close()

	var out []SpaceRef
	for rows.Next() {
		var r SpaceRef
		if err := rows.Scan(&r.ID, &r.Space.Provider, &r.Space.Model, &r.Space.Dim, &r.table); err != nil {
			return nil, fmt.Errorf("list embedding spaces: scan: %w", err)
		}
		out = append(out, r)
	}
	// rows.Err reports failures that happened mid-stream, after Query already
	// returned success. Without this check a connection dropped halfway through
	// looks exactly like a short result set.
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list embedding spaces: %w", err)
	}
	return out, nil
}
