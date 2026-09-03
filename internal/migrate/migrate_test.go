package migrate_test

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"testing/fstest"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"

	"github.com/Zinoki12/rag-ai-system/internal/migrate"
	"github.com/Zinoki12/rag-ai-system/internal/testdb"
)

func silent() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func tableExists(t *testing.T, pool *pgxpool.Pool, name string) bool {
	t.Helper()
	var exists bool
	err := pool.QueryRow(context.Background(),
		"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = $1)",
		name).Scan(&exists)
	if err != nil {
		t.Fatalf("check table %s: %v", name, err)
	}
	return exists
}

// hostMigrations stands in for the migrations of the program that imports this
// library — the Журнал, in practice.
var hostMigrations = fstest.MapFS{
	"00001_host.sql": &fstest.MapFile{Data: []byte(
		"-- +goose Up\nCREATE TABLE host_objects (id INT PRIMARY KEY);\n")},
	"00002_host.sql": &fstest.MapFile{Data: []byte(
		"-- +goose Up\nCREATE TABLE host_sheets (id INT PRIMARY KEY);\n")},
	"00003_host.sql": &fstest.MapFile{Data: []byte(
		"-- +goose Up\nCREATE TABLE host_entries (id INT PRIMARY KEY);\n")},
}

// applyHostMigrations migrates the host's own schema the way a host would: with
// goose's package-level API, which is what most programs use.
func applyHostMigrations(t *testing.T, pool *pgxpool.Pool) {
	t.Helper()

	db := stdlib.OpenDBFromPool(pool)
	defer db.Close()

	goose.SetTableName("journal_db_version")
	goose.SetBaseFS(hostMigrations)
	goose.SetLogger(goose.NopLogger())
	if err := goose.SetDialect("postgres"); err != nil {
		t.Fatalf("host SetDialect: %v", err)
	}
	if err := goose.UpContext(context.Background(), db, "."); err != nil {
		t.Fatalf("host migrations: %v", err)
	}
	// Deliberately not restored. The point of the test is that this library
	// must survive whatever the host left in goose's package variables.
}

// The failure this reproduces was silent and total. goose keeps the version
// table name, the migration filesystem, the dialect and the logger in package
// variables. A library that used goose's package-level API would read whichever
// version table the host program had configured, see the host's "version 3",
// conclude its own three migrations were already applied, run nothing, and
// report success. The missing tables then surfaced much later as an unrelated
// pgvector error.
func TestUpIgnoresTheHostsGooseGlobals(t *testing.T) {
	pool, _ := testdb.NewRaw(t)
	ctx := context.Background()

	applyHostMigrations(t, pool)

	if err := migrate.Up(ctx, pool, silent()); err != nil {
		t.Fatalf("Up after the host configured goose: %v", err)
	}

	// The schema this library needs is actually there.
	for _, name := range []string{"notes", "chunks", "embedding_spaces"} {
		if !tableExists(t, pool, name) {
			t.Errorf("table %s was not created", name)
		}
	}

	// Its history went into its own table.
	if !tableExists(t, pool, migrate.VersionTable) {
		t.Errorf("version table %s does not exist", migrate.VersionTable)
	}
	var applied int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM "+migrate.VersionTable).Scan(&applied); err != nil {
		t.Fatalf("count %s: %v", migrate.VersionTable, err)
	}
	if applied == 0 {
		t.Errorf("%s is empty; the migrations were not recorded", migrate.VersionTable)
	}

	// And the host's history is untouched: same table, same three versions,
	// same tables.
	var hostVersions int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM journal_db_version WHERE version_id > 0").Scan(&hostVersions); err != nil {
		t.Fatalf("count journal_db_version: %v", err)
	}
	if hostVersions != 3 {
		t.Errorf("host version table holds %d migrations, want its own 3", hostVersions)
	}
	for _, name := range []string{"host_objects", "host_sheets", "host_entries"} {
		if !tableExists(t, pool, name) {
			t.Errorf("host table %s disappeared", name)
		}
	}
}

// The mirror image: this library must not write its history into the table the
// host is using, on a database of its own.
func TestUpDoesNotWriteIntoTheHostsVersionTable(t *testing.T) {
	pool, _ := testdb.NewRaw(t)

	// The host configured goose but migrates a different database, so this one
	// starts empty. Only the globals are set.
	goose.SetTableName("journal_db_version")
	goose.SetBaseFS(hostMigrations)

	if err := migrate.Up(context.Background(), pool, silent()); err != nil {
		t.Fatalf("Up: %v", err)
	}

	if tableExists(t, pool, "journal_db_version") {
		t.Error("the library recorded its migrations in the host's version table")
	}
	if !tableExists(t, pool, migrate.VersionTable) {
		t.Errorf("version table %s does not exist", migrate.VersionTable)
	}
	// The host's migration files must not have been run either.
	if tableExists(t, pool, "host_objects") {
		t.Error("the library ran the host's migrations from goose's global base FS")
	}
}

// Every command in this project migrates on startup, so a second run against an
// already-migrated database is the normal case, not an edge one. It also covers
// the replay an existing installation performs when the version table moves.
func TestUpIsIdempotent(t *testing.T) {
	pool, _ := testdb.NewRaw(t)
	ctx := context.Background()

	for i := range 3 {
		if err := migrate.Up(ctx, pool, silent()); err != nil {
			t.Fatalf("Up #%d: %v", i+1, err)
		}
	}
}

// The real replay case: a database migrated under goose's default table name,
// as installations before this change were. Migrations run again against a
// schema that already matches, and must not fail on an object that exists.
func TestUpReplaysOntoAnExistingSchema(t *testing.T) {
	pool, _ := testdb.NewRaw(t)
	ctx := context.Background()

	if err := migrate.Up(ctx, pool, silent()); err != nil {
		t.Fatalf("first Up: %v", err)
	}
	// Forget the history, keep the schema.
	if _, err := pool.Exec(ctx, "DROP TABLE "+migrate.VersionTable); err != nil {
		t.Fatalf("drop version table: %v", err)
	}

	if err := migrate.Up(ctx, pool, silent()); err != nil {
		t.Fatalf("replay onto an existing schema: %v", err)
	}
	if !tableExists(t, pool, "embedding_spaces") {
		t.Error("embedding_spaces disappeared during the replay")
	}
}
