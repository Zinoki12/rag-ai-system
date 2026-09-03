// Package testdb creates throwaway databases for integration tests.
//
// It is only imported from _test files. Tests that touch a real Postgres are
// the only way to check the parts of this project that are SQL rather than Go —
// the dynamic DDL, the LEFT JOIN that finds unembedded chunks, the CASCADE that
// takes vectors with a deleted note. Mocking the database there would test the
// mock.
package testdb

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/Zinoki12/rag-ai-system/internal/migrate"
)

// EnvVar names the connection string tests connect through. Its database is
// never touched: it is only the entry point from which a fresh one is created.
const EnvVar = "RAG_TEST_DSN"

// New returns a pool on a freshly created, migrated database, and its DSN.
//
// The test is skipped when EnvVar is unset, so `go test ./...` stays green on a
// machine with no Postgres. A fresh database per test rather than a shared one
// means tests can count rows without caring what ran before them, and a failing
// test leaves nothing behind for the next one to trip over.
func New(t *testing.T) (*pgxpool.Pool, string) {
	t.Helper()

	pool, dsn := NewRaw(t)
	silent := slog.New(slog.NewTextHandler(io.Discard, nil))
	if err := migrate.Up(context.Background(), pool, silent); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return pool, dsn
}

// NewRaw is New without the migrations, for tests that need to arrange the
// database's state before this project's schema is applied to it.
func NewRaw(t *testing.T) (*pgxpool.Pool, string) {
	t.Helper()

	adminDSN := os.Getenv(EnvVar)
	if adminDSN == "" {
		t.Skipf("%s is not set; skipping the tests that need a real Postgres", EnvVar)
	}

	ctx := context.Background()
	admin, err := pgx.Connect(ctx, adminDSN)
	if err != nil {
		t.Fatalf("connect with %s: %v", EnvVar, err)
	}
	defer admin.Close(ctx)

	var suffix [6]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		t.Fatalf("random name: %v", err)
	}
	name := "ragtest_" + hex.EncodeToString(suffix[:])

	// CREATE DATABASE takes no parameters, so the name is interpolated. It is
	// hex from crypto/rand behind a fixed prefix, and quoted on the way in.
	quoted := pgx.Identifier{name}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+quoted); err != nil {
		t.Fatalf("create database %s: %v", name, err)
	}

	dsn, err := replaceDatabase(adminDSN, name)
	if err != nil {
		t.Fatalf("build dsn: %v", err)
	}

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("connect to %s: %v", name, err)
	}

	t.Cleanup(func() {
		pool.Close()

		// A separate connection: the pool is closed, and DROP DATABASE cannot
		// run from a session connected to the database being dropped.
		dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		conn, err := pgx.Connect(dropCtx, adminDSN)
		if err != nil {
			t.Logf("cleanup: connect: %v", err)
			return
		}
		defer conn.Close(dropCtx)
		if _, err := conn.Exec(dropCtx, "DROP DATABASE IF EXISTS "+quoted+" WITH (FORCE)"); err != nil {
			t.Logf("cleanup: drop %s: %v", name, err)
		}
	})

	return pool, dsn
}

// replaceDatabase swaps the database name in a connection URL, leaving every
// other component — including a password with characters that need escaping —
// exactly as it was.
func replaceDatabase(dsn, database string) (string, error) {
	u, err := url.Parse(dsn)
	if err != nil {
		return "", fmt.Errorf("parse dsn: %w", err)
	}
	u.Path = "/" + database
	return u.String(), nil
}
