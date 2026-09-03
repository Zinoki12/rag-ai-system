// Package migrate owns the database schema.
//
// The schema used to live in postgres/schema.sql, mounted into the container's
// docker-entrypoint-initdb.d. That directory only runs on an empty data volume,
// so editing the file did nothing to a database that already had data — every
// schema change meant either hand-written ALTERs in psql or destroying the
// volume. The migrations here are embedded in the binary and applied at startup,
// so the schema a build expects always travels with that build.
//
// Everything here goes through goose's instance API (goose.NewProvider) and
// never through its package-level functions. That is not a style preference.
// goose.SetTableName, goose.SetBaseFS, goose.SetDialect and goose.SetLogger all
// write to package variables shared by every caller in the process, so a
// library that used them would silently take over — or be taken over by — the
// migrations of the program that imported it. See VersionTable.
package migrate

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

// VersionTable is where this package records which of its migrations have run.
//
// Not goose's default "goose_db_version": the host program may run its own
// goose migrations against the same database, and sharing one version table
// means each set of migrations reads the other's version number as its own. The
// failure is silent and total — goose sees "version 3" written by someone else,
// concludes its own 1..3 are applied, runs nothing, and reports success. The
// missing tables then surface much later as an unrelated-looking error.
//
// A separate table costs nothing and makes the two histories independent.
const VersionTable = "rag_db_version"

// advisoryLockID guards the migration run. Every command in this project
// migrates on startup, so two of them racing at boot is normal rather than
// exotic; without the lock both would try to apply the same migration and one
// would fail on a duplicate object. The value is arbitrary but must be stable.
const advisoryLockID int64 = 0x7261676169 // "ragai" in ASCII

// Up applies every pending migration. It is safe to call concurrently from
// several processes: the second one blocks on the advisory lock and then finds
// nothing left to do.
func Up(ctx context.Context, pool *pgxpool.Pool, log *slog.Logger) error {
	// Hold the lock on one dedicated connection for the whole run. Advisory
	// locks are session-scoped, so it must be the same connection from
	// pg_advisory_lock to pg_advisory_unlock — Acquire, not a pool call.
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire connection for migration lock: %w", err)
	}
	defer conn.Release()

	if _, err := conn.Exec(ctx, "SELECT pg_advisory_lock($1)", advisoryLockID); err != nil {
		return fmt.Errorf("take migration lock: %w", err)
	}
	defer func() {
		// Releasing the connection would drop the lock anyway, but doing it
		// explicitly keeps the lock's lifetime visible in the code. A fresh
		// context with its own deadline: ctx may already be cancelled by the
		// time this runs, and an unlock that blocks forever in a defer is worse
		// than one that gives up.
		unlockCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_, _ = conn.Exec(unlockCtx, "SELECT pg_advisory_unlock($1)", advisoryLockID)
	}()

	// goose speaks database/sql. This adapter borrows connections from the same
	// pgxpool rather than opening a second set, and closing it leaves the pool
	// itself untouched.
	db := stdlib.OpenDBFromPool(pool)
	defer db.Close()

	// goose reads migrations from the root of the FS it is given, so hand it
	// the subdirectory rather than the whole embedded tree.
	sub, err := fs.Sub(migrationsFS, "migrations")
	if err != nil {
		return fmt.Errorf("open embedded migrations: %w", err)
	}

	opts := []goose.ProviderOption{
		goose.WithTableName(VersionTable),
		// Go migrations registered by the host program through goose's global
		// registry are not ours to run.
		goose.WithDisableGlobalRegistry(true),
	}
	if log != nil {
		opts = append(opts, goose.WithSlog(log))
	}

	provider, err := goose.NewProvider(goose.DialectPostgres, db, sub, opts...)
	if err != nil {
		return fmt.Errorf("configure migrations: %w", err)
	}

	if _, err := provider.Up(ctx); err != nil {
		return fmt.Errorf("apply migrations: %w", err)
	}
	return nil
}
