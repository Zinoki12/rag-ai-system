-- +goose Up
-- Baseline: the schema as it was created by postgres/schema.sql via
-- docker-entrypoint-initdb.d. Everything is IF NOT EXISTS so this migration is a
-- no-op on databases that already have the tables, and creates them from scratch
-- on a fresh volume.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS notes (
    id      INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    paths   VARCHAR(4096) UNIQUE NOT NULL,
    name    VARCHAR(255),
    text    TEXT,
    hash    BYTEA,
    created TIMESTAMPTZ DEFAULT now(),
    updated TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    note_id         INTEGER REFERENCES notes (id) ON DELETE CASCADE,
    chunk_index     SMALLINT,
    chunk_text      TEXT,
    embedding       vector(768),
    embedding_model VARCHAR(255)
);

-- +goose Down
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS notes;
