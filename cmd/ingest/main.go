package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/Zinoki12/rag-ai-system/internal/storage"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("Critical error: %v", err)
	}
}

func run() error {
	ctx := context.Background()
	pool, err := storage.NewPool(ctx)
	if err != nil {
		return fmt.Errorf("error at new pool: %w", err)
	}
	defer pool.Close()

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.Ping(pingCtx); err != nil {
		return fmt.Errorf("error at ping: %w", err)
	}

	fmt.Println("Connection to db is successefull")
	return nil
}
