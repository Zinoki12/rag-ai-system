// Package migrate owns the database schema.
//
// The schema used to live in postgres/schema.sql, mounted into the container's
// docker-entrypoint-initdb.d. That directory only runs on an empty data volume,
// so editing the file did nothing to a database that already had data — every
// schema change meant either hand-written ALTERs in psql or destroying the
// volume. The migrations here are embedded in the binary and applied at startup,
// so the schema a build expects always travels with that build.
package migrate

import (
	"context"
	"embed"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

// advisoryLockID guards the migration run. Every command in this project
// migrates on startup, so two of them racing at boot is normal rather than
// exotic; without the lock both would try to apply the same migration and one
// would fail on a duplicate object. The value is arbitrary but must be stable.
const advisoryLockID int64 = 0x7261676169 // "ragai" in ASCII

// gooseLogger adapts goose's Printf/Fatalf logger to slog, so migration output
// lands in the same stream and format as everything else the process logs
// rather than as stray plain text among JSON lines.
type gooseLogger struct{ log *slog.Logger }

func (g gooseLogger) Printf(format string, v ...any) {
	g.log.Info(strings.TrimSpace(fmt.Sprintf(format, v...)))
}

// Fatalf must not exit the process: goose calls it on errors that Up already
// returns, and killing the program here would skip every deferred cleanup.
func (g gooseLogger) Fatalf(format string, v ...any) {
	g.log.Error(strings.TrimSpace(fmt.Sprintf(format, v...)))
}

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

	goose.SetBaseFS(migrationsFS)
	if log != nil {
		goose.SetLogger(gooseLogger{log: log})
	}
	if err := goose.SetDialect("postgres"); err != nil {
		return fmt.Errorf("set goose dialect: %w", err)
	}

	if err := goose.UpContext(ctx, db, "migrations"); err != nil {
		return fmt.Errorf("apply migrations: %w", err)
	}
	return nil
}
