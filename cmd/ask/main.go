// Command ask answers a question from the indexed vault.
//
//	go run ./cmd/ask "как устроен чанкинг"
//	go run ./cmd/ask -k 8 -search "чанкинг"
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/Zinoki12/rag-ai-system/internal/app"
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
		return fmt.Errorf("не задан вопрос")
	}

	a, err := app.Open(ctx, app.Options{WithLLM: !*searchOnly})
	if err != nil {
		return err
	}
	defer a.Close()

	if *searchOnly {
		return printSearch(ctx, a, question, *topK)
	}
	return printAnswer(ctx, a, question, *topK)
}

func printSearch(ctx context.Context, a *app.App, question string, topK int) error {
	hits, err := a.Search(ctx, question, topK)
	if err != nil {
		return err
	}
	if len(hits) == 0 {
		fmt.Println("Ничего не найдено. Индекс пуст? Запусти: go run ./cmd/ingest")
		return nil
	}

	fmt.Printf("Пространство: %s\n\n", a.Space.Space)
	for i, h := range hits {
		title := h.NoteName
		if title == "" {
			title = "(без заголовка)"
		}
		fmt.Printf("%d. [%.3f] %s — %s #%d\n", i+1, h.Score, title, h.NotePath, h.ChunkIndex)
		fmt.Printf("   %s\n\n", preview(h.Text, 240))
	}
	return nil
}

func printAnswer(ctx context.Context, a *app.App, question string, topK int) error {
	answer, err := a.Ask(ctx, question, topK)
	if err != nil {
		return err
	}

	fmt.Println(answer.Text)

	if len(answer.Sources) > 0 {
		fmt.Println("\nИсточники:")
		for _, s := range answer.Sources {
			fmt.Printf("  - %s\n", s)
		}
	}
	fmt.Printf("\n(модель: %s, пространство: %s, фрагментов: %d)\n",
		answer.Model, a.Space.Space, len(answer.Passages))
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
