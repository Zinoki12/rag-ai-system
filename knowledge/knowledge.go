// Package knowledge is the library entry point: a searchable knowledge base
// built from a directory of markdown files.
//
// Use it when you want the whole pipeline — indexing, semantic search, grounded
// answers — inside your own program:
//
//	kb, err := knowledge.Open(ctx, knowledge.Config{
//		DSN:       "postgres://user:pass@127.0.0.1:5433/ragdb?sslmode=disable",
//		VaultPath: "/srv/vault",
//		Embedder:  embed.Config{Provider: embed.ProviderOllama},
//	})
//	if err != nil { return err }
//	defer kb.Close()
//
//	if _, err := kb.Index(ctx); err != nil { return err }
//	hits, err := kb.Search(ctx, "как устроен чанкинг", 5)
//
// It is a library in the sense net/http is one: you call it, it never calls
// you, it owns no goroutines you did not ask for and no signal handlers. Open
// gives you a value, Close releases it, and everything in between is a method
// with a context.
//
// The pieces are usable separately too: embed and llm are standalone connectors
// with no dependency on this package or on a database.
package knowledge

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"time"

	"github.com/Zinoki12/rag-ai-system/embed"
	"github.com/Zinoki12/rag-ai-system/internal/chunk"
	"github.com/Zinoki12/rag-ai-system/internal/migrate"
	"github.com/Zinoki12/rag-ai-system/internal/storage"
	"github.com/Zinoki12/rag-ai-system/internal/vault"
	"github.com/Zinoki12/rag-ai-system/llm"
	"github.com/Zinoki12/rag-ai-system/rag"
)

// DefaultChunkSize is in runes. It is kept clear of any vector dimension so the
// two numbers cannot be mistaken for each other.
const DefaultChunkSize = 800

// pendingBatch is how many chunks are claimed from the database per round; the
// embedding provider splits this further by its own batch limit.
const pendingBatch = 128

// Config describes a knowledge base.
type Config struct {
	// DSN is the Postgres connection string. Either this or Postgres is
	// required; DSN wins when both are set.
	DSN string

	// Postgres builds the connection string from parts, escaping each one
	// correctly. Prefer it over hand-writing DSN unless you already have a
	// connection string from somewhere else.
	Postgres Postgres

	// VaultPath is the markdown directory to index. Required only for Index.
	VaultPath string

	// ChunkSize in runes; zero means DefaultChunkSize.
	ChunkSize int

	// Embedder selects the embedding provider. Required — it determines which
	// vector space this knowledge base reads and writes.
	Embedder embed.Config

	// Generator selects the text generation provider. Leave nil for a
	// search-only knowledge base; Ask then returns ErrNoGenerator, while
	// Index and Search work normally.
	Generator *llm.Config

	// Logger receives startup and migration diagnostics. Nil discards them:
	// a library should be silent unless asked to speak.
	Logger *slog.Logger

	// Migrate applies pending schema migrations on Open. Defaults to true.
	// Set SkipMigrate if the schema is managed elsewhere.
	SkipMigrate bool
}

// ErrNoGenerator is returned by Ask when Config.Generator was nil.
var ErrNoGenerator = errors.New("knowledge: no generation provider configured")

// Knowledge is an open knowledge base. It is safe for concurrent use.
type Knowledge struct {
	cfg    Config
	log    *slog.Logger
	store  *storage.Store
	pool   interface{ Close() }
	embed  embed.Provider
	llm    llm.Provider
	space  storage.SpaceRef
	chunkN int
}

// Open connects, applies migrations, builds the configured providers and
// registers the embedding space they write into.
func Open(ctx context.Context, cfg Config) (*Knowledge, error) {
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if cfg.ChunkSize <= 0 {
		cfg.ChunkSize = DefaultChunkSize
	}

	dsn := cfg.DSN
	if dsn == "" {
		if err := cfg.Postgres.validate(); err != nil {
			return nil, err
		}
		dsn = cfg.Postgres.DSN()
	}

	pool, err := storage.NewPool(ctx, dsn)
	if err != nil {
		return nil, err
	}

	// Any failure past this point must release the pool, or a program that
	// exits on a bad configuration leaves connections open until it dies.
	ok := false
	defer func() {
		if !ok {
			pool.Close()
		}
	}()

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.Ping(pingCtx); err != nil {
		return nil, fmt.Errorf("ping database: %w", err)
	}

	if !cfg.SkipMigrate {
		if err := migrate.Up(ctx, pool, cfg.Logger); err != nil {
			return nil, err
		}
	}

	embedder, err := embed.New(ctx, cfg.Embedder)
	if err != nil {
		return nil, err
	}

	store := storage.New(pool)
	space, err := store.EnsureSpace(ctx, embedder.Space())
	if err != nil {
		return nil, err
	}

	k := &Knowledge{
		cfg:    cfg,
		log:    cfg.Logger,
		store:  store,
		pool:   pool,
		embed:  embedder,
		space:  space,
		chunkN: cfg.ChunkSize,
	}

	if cfg.Generator != nil {
		if k.llm, err = llm.New(ctx, *cfg.Generator); err != nil {
			return nil, err
		}
	}

	ok = true
	return k, nil
}

// Close releases the connection pool. The Knowledge must not be used after.
func (k *Knowledge) Close() { k.pool.Close() }

// Space reports the vector space this knowledge base reads and writes.
func (k *Knowledge) Space() embed.Space { return k.space.Space }

// Stats reports index coverage.
type Stats struct {
	Notes    int         `json:"notes"`
	Chunks   int         `json:"chunks"`
	Embedded int         `json:"embedded"`
	Space    embed.Space `json:"space"`
}

// Stats counts what is indexed.
func (k *Knowledge) Stats(ctx context.Context) (Stats, error) {
	st, err := k.store.Stats(ctx, k.space)
	if err != nil {
		return Stats{}, err
	}
	return Stats{Notes: st.Notes, Chunks: st.Chunks, Embedded: st.Embedded, Space: k.space.Space}, nil
}

// IndexResult summarises one indexing pass.
type IndexResult struct {
	NotesWritten   int `json:"notes_written"`   // content changed, or new
	NotesUnchanged int `json:"notes_unchanged"` // file hash matched, nothing rewritten
	NotesDeleted   int `json:"notes_deleted"`   // file gone from the vault
	NotesSkipped   int `json:"notes_skipped"`   // file unparseable; left in the index as it was
	NotesFailed    int `json:"notes_failed"`    // could not be stored; see the returned error
	ChunksEmbedded int `json:"chunks_embedded"`
}

// Index brings the database in line with the vault.
//
// It is safe to call as often as you like: notes whose file hash is unchanged
// are skipped, and embeddings are computed only for chunks that do not have one
// in this space yet. That also makes it resumable — an interrupted pass picks
// up where it stopped — and makes switching embedding models recompute
// everything on its own, with no separate command to remember.
//
// A note that cannot be stored does not stop the pass: the rest are indexed and
// the failures come back in the error, so a caller can log them and still have
// a usable index. A returned error therefore does not mean nothing happened;
// check IndexResult too.
//
// Notes whose file has disappeared from the vault are removed along with their
// chunks and vectors. A file that is present but cannot be parsed is not a
// disappearance: it keeps whatever it had in the index, and is counted in
// IndexResult.NotesSkipped.
func (k *Knowledge) Index(ctx context.Context) (IndexResult, error) {
	var res IndexResult

	if k.cfg.VaultPath == "" {
		return res, errors.New("knowledge: Index needs Config.VaultPath")
	}

	scan, err := vault.Scan(k.cfg.VaultPath)
	if err != nil {
		// Scan reports two different things through one error: files it had to
		// skip, and a failure that stopped the walk. Only the second is fatal.
		if !errors.Is(err, vault.ErrUnclosedFrontmatter) {
			return res, fmt.Errorf("scan vault %s: %w", k.cfg.VaultPath, err)
		}
		k.log.Warn("skipped files with malformed frontmatter", slog.String("detail", err.Error()))
	}
	res.NotesSkipped = len(scan.Skipped)

	var noteErrs []error
	paths := make([]string, 0, len(scan.Notes)+len(scan.Skipped))

	// A file that could not be parsed still exists, so it counts as present for
	// the prune below. Otherwise a stray edit to the frontmatter — deleting the
	// closing --- is the likeliest one — would drop the note, its chunks and its
	// vectors while the file sat untouched in the vault, and Index would report
	// no error at all. Keeping the previously indexed version is strictly better
	// than deleting it: the text on disk has not gone anywhere, and the next
	// clean scan overwrites it.
	paths = append(paths, scan.Skipped...)

	for _, file := range scan.Notes {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		paths = append(paths, file.Path)

		_, written, err := k.store.SaveNoteWithChunks(ctx, file, func() ([]string, error) {
			return chunk.Cut(file.Text, k.chunkN)
		})
		if err != nil {
			res.NotesFailed++
			noteErrs = append(noteErrs, fmt.Errorf("note %s: %w", file.Path, err))
			continue
		}
		if written {
			res.NotesWritten++
		} else {
			res.NotesUnchanged++
		}
	}

	// Only prune when every file was accounted for. Deleting on the strength of
	// a partial scan would drop notes whose files are present but unreadable.
	if len(noteErrs) == 0 {
		if res.NotesDeleted, err = k.store.DeleteMissingNotes(ctx, paths); err != nil {
			return res, err
		}
	}

	embedded, err := k.embedPending(ctx)
	res.ChunksEmbedded = embedded
	if err != nil {
		return res, err
	}
	return res, errors.Join(noteErrs...)
}

// embedPending fills in vectors for chunks that do not have one in this space.
func (k *Knowledge) embedPending(ctx context.Context) (int, error) {
	done := 0
	for {
		pending, err := k.store.PendingChunks(ctx, k.space, pendingBatch)
		if err != nil {
			return done, err
		}
		if len(pending) == 0 {
			return done, nil
		}

		texts := make([]string, len(pending))
		for i, c := range pending {
			texts[i] = c.Text
		}

		vectors, err := embed.Batched(ctx, k.embed, texts, embed.KindDocument)
		if err != nil {
			return done, fmt.Errorf("embed %d chunks: %w", len(texts), err)
		}

		batch := make([]storage.ChunkVector, len(pending))
		for i, c := range pending {
			batch[i] = storage.ChunkVector{ChunkID: c.ID, Vector: vectors[i]}
		}
		if err := k.store.SaveEmbeddings(ctx, k.space, batch); err != nil {
			return done, err
		}

		done += len(pending)
		k.log.Debug("embedded a batch", slog.Int("chunks", len(pending)), slog.Int("total", done))
	}
}

// Hit is one retrieved chunk.
type Hit struct {
	Source string  `json:"source"` // note path, relative to the vault root
	Title  string  `json:"title"`  // note title from the frontmatter; may be empty
	Index  int     `json:"index"`  // which chunk of that note
	Text   string  `json:"text"`   // the chunk itself
	Score  float64 `json:"score"`  // cosine similarity in [-1, 1]; higher is closer
}

// Search returns the topK chunks closest in meaning to query.
func (k *Knowledge) Search(ctx context.Context, query string, topK int) ([]Hit, error) {
	// KindQuery, not KindDocument. Embedding a question the way a stored
	// passage is embedded costs result quality and reports nothing.
	vectors, err := k.embed.Embed(ctx, []string{query}, embed.KindQuery)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("embed query: got %d vectors for 1 input", len(vectors))
	}

	found, err := k.store.Search(ctx, k.space, vectors[0], topK)
	if err != nil {
		return nil, err
	}

	hits := make([]Hit, 0, len(found))
	for _, h := range found {
		hits = append(hits, Hit{
			Source: h.NotePath,
			Title:  h.NoteName,
			Index:  h.ChunkIndex,
			Text:   h.Text,
			Score:  h.Score,
		})
	}
	return hits, nil
}

// Answer is a generated response with the material it was shown.
type Answer struct {
	Text  string `json:"text"`
	Model string `json:"model"`

	// Hits are the chunks that went into the prompt, best match first.
	Hits []Hit `json:"hits"`

	// Sources lists each note in Hits once, in first-seen order. These are the
	// notes shown to the model, not a claim about what the answer rests on: the
	// model may well have replied that the knowledge base has no answer.
	Sources []string `json:"sources"`

	// TopScore is the similarity of the best hit, so a caller can judge how
	// much the retrieval was worth. There is deliberately no relevance
	// threshold here: the cutoff differs per embedding model and per corpus,
	// and one picked without real content would be a guess wearing a number.
	TopScore float64 `json:"top_score"`
}

// Ask retrieves context for the question and has the model answer from it.
//
// The model is instructed to answer only from the retrieved passages and to say
// so when they do not contain the answer. Returns ErrNoGenerator when the
// knowledge base was opened without a generation provider.
func (k *Knowledge) Ask(ctx context.Context, question string, topK int) (Answer, error) {
	if k.llm == nil {
		return Answer{}, ErrNoGenerator
	}

	hits, err := k.Search(ctx, question, topK)
	if err != nil {
		return Answer{}, err
	}

	passages := make([]rag.Passage, 0, len(hits))
	for _, h := range hits {
		passages = append(passages, rag.Passage{
			Source: h.Source, Title: h.Title, Index: h.Index, Text: h.Text,
		})
	}

	text, err := k.llm.Generate(ctx, rag.SystemPrompt,
		rag.BuildUserPrompt(question, passages, rag.DefaultMaxContextRunes))
	if err != nil {
		return Answer{}, err
	}

	answer := Answer{
		Text:    text,
		Model:   k.llm.Model(),
		Hits:    hits,
		Sources: rag.Sources(passages),
	}
	if len(hits) > 0 {
		answer.TopScore = hits[0].Score // Search returns best first
	}
	return answer, nil
}
