package storage

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
)

type EnvInfo struct {
	pgPassword string
	pgUser     string
	pgDB       string
	pgHost     string
	pgPort     string
}

func readDbEnv() (EnvInfo, error) {
	env := EnvInfo{
		pgPassword: os.Getenv("POSTGRES_PASSWORD"),
		pgUser:     os.Getenv("POSTGRES_USER"),
		pgDB:       os.Getenv("POSTGRES_DB"),
		pgHost:     os.Getenv("POSTGRES_HOST"),
		pgPort:     os.Getenv("POSTGRES_PORT"),
	}

	switch {
	case env.pgPassword == "":
		return env, errors.New("no POSTGRES_PASSWORD has been set")
	case env.pgUser == "":
		return env, errors.New("no POSTGRES_USER has been set")
	case env.pgDB == "":
		return env, errors.New("no POSTGRES_DB has been set")
	case env.pgHost == "":
		return env, errors.New("no POSTGRES_HOST has been set")
	case env.pgPort == "":
		return env, errors.New("no POSTGRES_PORT has been set")
	}

	return env, nil
}

func buildDSN(env EnvInfo) string {
	user := url.UserPassword(env.pgUser, env.pgPassword)
	u := url.URL{
		Scheme:   "postgres",
		User:     user,
		Host:     env.pgHost + ":" + env.pgPort,
		Path:     env.pgDB,
		RawQuery: "sslmode=disable",
	}
	return u.String()
}

func NewPool(ctx context.Context) (*pgxpool.Pool, error) {
	env, err := readDbEnv()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize application: %w", err)
	}

	dsn := buildDSN(env)
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("can't connect to db: %w", err)
	}

	return pool, nil
}
