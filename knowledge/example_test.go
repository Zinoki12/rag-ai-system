package knowledge_test

import (
	"context"
	"fmt"
	"log"

	"github.com/Zinoki12/rag-ai-system/embed"
	"github.com/Zinoki12/rag-ai-system/knowledge"
	"github.com/Zinoki12/rag-ai-system/llm"
)

// These examples have no "// Output:" comment, so the toolchain compiles them
// without running them. That is the point: they need a database and a model to
// run, but they must never be allowed to stop compiling, because documentation
// that has quietly drifted from the API is worse than none.

// The whole pipeline in one place: index a vault, then answer from it.
func Example() {
	ctx := context.Background()

	kb, err := knowledge.Open(ctx, knowledge.Config{
		Postgres: knowledge.Postgres{
			Host: "127.0.0.1", Port: "5433",
			User: "rag", Password: "secret", Database: "ragdb",
		},
		VaultPath: "/srv/vault",
		Embedder:  embed.Config{Provider: embed.ProviderOllama},
		Generator: &llm.Config{Provider: llm.ProviderOllama, Model: "llama3.2"},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer kb.Close()

	if _, err := kb.Index(ctx); err != nil {
		log.Fatal(err)
	}

	answer, err := kb.Ask(ctx, "как устроен чанкинг", 5)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer.Text)
	for _, source := range answer.Sources {
		fmt.Println(" -", source)
	}
}

// A knowledge base opened without a generator does semantic search only. Ask
// returns ErrNoGenerator; Index and Search work as usual.
func ExampleKnowledge_Search() {
	ctx := context.Background()

	kb, err := knowledge.Open(ctx, knowledge.Config{
		DSN:      "postgres://rag:secret@127.0.0.1:5433/ragdb?sslmode=disable",
		Embedder: embed.Config{Provider: embed.ProviderGoogle, APIKey: "..."},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer kb.Close()

	hits, err := kb.Search(ctx, "что писали про эту стену", 10)
	if err != nil {
		log.Fatal(err)
	}
	for _, h := range hits {
		fmt.Printf("%.3f %s #%d\n", h.Score, h.Source, h.Index)
	}
}

// After writing new markdown into the vault, call Index to pick it up. It only
// touches what changed, so calling it often is cheap.
func ExampleKnowledge_Index() {
	ctx := context.Background()

	kb, err := knowledge.Open(ctx, knowledge.Config{
		DSN:       "postgres://rag:secret@127.0.0.1:5433/ragdb?sslmode=disable",
		VaultPath: "/srv/vault",
		Embedder:  embed.Config{Provider: embed.ProviderOllama},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer kb.Close()

	res, err := kb.Index(ctx)
	// A note that could not be stored does not stop the pass, so the counts are
	// worth reading even when err is non-nil.
	fmt.Printf("written %d, unchanged %d, deleted %d, embedded %d\n",
		res.NotesWritten, res.NotesUnchanged, res.NotesDeleted, res.ChunksEmbedded)
	if err != nil {
		log.Printf("some notes failed: %v", err)
	}
}
