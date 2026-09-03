// Command ask answers a question from the indexed vault.
//
//	go run ./cmd/ask "как устроен чанкинг"
//	go run ./cmd/ask -k 8 -search "чанкинг"
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/Zinoki12/rag-ai-system/knowledge"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx); err != nil {
		log.Fatalf("ask: %v", err)
	}
}

func run(ctx context.Context) error {
	topK := flag.Int("k", 5, "сколько фрагментов доставать из базы")
	searchOnly := flag.Bool("search", false, "только поиск, без обращения к модели")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Использование: ask [флаги] <вопрос>\n\nФлаги:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	question := strings.TrimSpace(strings.Join(flag.Args(), " "))
	if question == "" {
		flag.Usage()
		return errors.New("не задан вопрос")
	}

	cfg, err := knowledge.ConfigFromEnv()
	if err != nil {
		return err
	}
	if *searchOnly {
		// Nil generator: searching must not fail on a misconfigured model.
		cfg.Generator = nil
	}

	kb, err := knowledge.Open(ctx, cfg)
	if err != nil {
		return err
	}
	defer kb.Close()

	if *searchOnly {
		return printSearch(ctx, kb, question, *topK)
	}
	return printAnswer(ctx, kb, question, *topK)
}

func printSearch(ctx context.Context, kb *knowledge.Knowledge, question string, topK int) error {
	hits, err := kb.Search(ctx, question, topK)
	if err != nil {
		return err
	}
	if len(hits) == 0 {
		fmt.Println("Ничего не найдено. Индекс пуст? Запусти: go run ./cmd/ingest")
		return nil
	}

	fmt.Printf("Пространство: %s\n\n", kb.Space())
	for i, h := range hits {
		title := h.Title
		if title == "" {
			title = "(без заголовка)"
		}
		fmt.Printf("%d. [%.3f] %s — %s #%d\n", i+1, h.Score, title, h.Source, h.Index)
		fmt.Printf("   %s\n\n", preview(h.Text, 240))
	}
	return nil
}

func printAnswer(ctx context.Context, kb *knowledge.Knowledge, question string, topK int) error {
	answer, err := kb.Ask(ctx, question, topK)
	if err != nil {
		return err
	}

	fmt.Println(answer.Text)

	// "Показано модели", not "Источники": these are the fragments that went
	// into the prompt. Whether the answer rests on them is not something this
	// program knows, and calling them sources when the model has just said the
	// base has no answer would be a claim it cannot support.
	if len(answer.Sources) > 0 {
		fmt.Println("\nПоказано модели:")
		for _, s := range answer.Sources {
			fmt.Printf("  - %s\n", s)
		}
	}
	fmt.Printf("\n(модель: %s, пространство: %s, фрагментов: %d, лучшая близость: %.3f)\n",
		answer.Model, kb.Space(), len(answer.Hits), answer.TopScore)
	return nil
}

// preview collapses a chunk to a single readable line.
func preview(s string, limit int) string {
	s = strings.Join(strings.Fields(s), " ")
	runes := []rune(s)
	if len(runes) <= limit {
		return s
	}
	return string(runes[:limit]) + "…"
}
