-- +goose Up

-- 1. Tighten `chunks`.
--
-- note_id and chunk_index were nullable, which let a chunk exist detached from
-- its note. Nothing ever wrote NULL there, so this is safe to enforce now.
ALTER TABLE chunks ALTER COLUMN note_id     SET NOT NULL;
ALTER TABLE chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE chunks ALTER COLUMN chunk_text  SET NOT NULL;

-- The unique index doubles as the lookup index on note_id: a btree on
-- (note_id, chunk_index) can serve any query with a note_id predicate because
-- note_id is the leading column. A separate index on note_id alone would be
-- redundant write amplification.
ALTER TABLE chunks ADD CONSTRAINT chunks_note_id_chunk_index_key
    UNIQUE (note_id, chunk_index);

-- 2. A long YAML `title` currently fails the insert with a value-too-long error.
-- There is no reason to cap it: TEXT and VARCHAR(n) are the same storage in
-- Postgres, the limit only buys a runtime error.
ALTER TABLE notes ALTER COLUMN name TYPE TEXT;
ALTER TABLE notes ALTER COLUMN text SET NOT NULL;
ALTER TABLE notes ALTER COLUMN hash SET NOT NULL;

-- 3. Vectors move out of `chunks` into one table per embedding space.
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_model;

-- 4. The registry of embedding spaces.
--
-- A "space" is one (provider, model, dim) triple. Vectors produced by different
-- models are not comparable — cosine distance between them computes fine and
-- means nothing — so each space owns a separate physical table, created on
-- demand by internal/storage.EnsureSpace. Keeping the dimension in the column
-- type is what makes an hnsw index possible; a bare `vector` column accepts any
-- dimension but cannot be indexed.
CREATE TABLE IF NOT EXISTS embedding_spaces (
    id         SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    dim        INT  NOT NULL CHECK (dim BETWEEN 1 AND 16000),
    table_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, model, dim)
);

-- +goose Down
DROP TABLE IF EXISTS embedding_spaces;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(768);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(255);
ALTER TABLE notes ALTER COLUMN hash DROP NOT NULL;
ALTER TABLE notes ALTER COLUMN text DROP NOT NULL;
ALTER TABLE notes ALTER COLUMN name TYPE VARCHAR(255);
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_note_id_chunk_index_key;
ALTER TABLE chunks ALTER COLUMN chunk_text  DROP NOT NULL;
ALTER TABLE chunks ALTER COLUMN chunk_index DROP NOT NULL;
ALTER TABLE chunks ALTER COLUMN note_id     DROP NOT NULL;
