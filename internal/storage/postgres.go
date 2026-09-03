package storage

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

// envInfo holds the POSTGRES_* variables.
type envInfo struct {
	pgPassword string
	pgUser     string
	pgDB       string
	pgHost     string
	pgPort     string
}

func readDBEnv() (envInfo, error) {
	env := envInfo{
		pgPassword: os.Getenv("POSTGRES_PASSWORD"),
		pgUser:     os.Getenv("POSTGRES_USER"),
		pgDB:       os.Getenv("POSTGRES_DB"),
		pgHost:     os.Getenv("POSTGRES_HOST"),
		pgPort:     os.Getenv("POSTGRES_PORT"),
	}

	// Report every missing variable at once. Reporting only the first turns
	// filling in a fresh .env into a run-fix-run loop.
	var missing []string
	for name, value := range map[string]string{
		"POSTGRES_PASSWORD": env.pgPassword,
		"POSTGRES_USER":     env.pgUser,
		"POSTGRES_DB":       env.pgDB,
		"POSTGRES_HOST":     env.pgHost,
		"POSTGRES_PORT":     env.pgPort,
	} {
		if strings.TrimSpace(value) == "" {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		slices.Sort(missing)
		return env, fmt.Errorf("database configuration: %s not set", strings.Join(missing, ", "))
	}
	return env, nil
}

func buildDSN(env envInfo) string {
	u := url.URL{
		Scheme:   "postgres",
		User:     url.UserPassword(env.pgUser, env.pgPassword),
		Host:     env.pgHost + ":" + env.pgPort,
		Path:     env.pgDB,
		RawQuery: "sslmode=disable",
	}
	return u.String()
}

// DSNFromEnv assembles a connection string from the POSTGRES_* variables.
//
// Separate from NewPool so that a program embedding this package as a library
// can pass its own DSN — from a config file, a secret manager, an existing
// connection string — instead of being forced to set process-wide environment
// variables just to open a pool.
func DSNFromEnv() (string, error) {
	env, err := readDBEnv()
	if err != nil {
		return "", err
	}
	return buildDSN(env), nil
}

// NewPool opens a connection pool for dsn.
//
// pgxpool connects lazily, so a bad address surfaces on first use rather than
// here; callers should Ping before reporting success.
func NewPool(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	if strings.TrimSpace(dsn) == "" {
		return nil, errors.New("database DSN is empty")
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("connect to database: %w", err)
	}
	return pool, nil
}
