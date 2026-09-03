package embed_test

import (
	"context"
	"fmt"
	"log"

	"github.com/Zinoki12/rag-ai-system/embed"
)

// The embedding connector is usable on its own, with no database and no
// dependency on the rest of this repository.
func Example() {
	ctx := context.Background()

	// One line switches this between a local model and a hosted API.
	provider, err := embed.New(ctx, embed.Config{
		Provider: embed.ProviderOllama,
		Model:    "nomic-embed-text",
		Dim:      768,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Batched splits the input across as many calls as the provider's batch
	// limit requires and keeps the order.
	vectors, err := embed.Batched(ctx, provider,
		[]string{"первый документ", "второй документ"}, embed.KindDocument)
	if err != nil {
		log.Fatal(err)
	}

	// KindQuery for a search string, KindDocument for stored text: providers
	// place them differently, and using the wrong one costs result quality
	// without reporting anything.
	query, err := provider.Embed(ctx, []string{"что за документы"}, embed.KindQuery)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(len(vectors), len(query[0]), provider.Space())
}
