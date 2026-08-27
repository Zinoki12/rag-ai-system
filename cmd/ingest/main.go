package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/Zinoki12/rag-ai-system/internal/storage"
	"github.com/Zinoki12/rag-ai-system/internal/vault"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("Critical error: %v", err)
	}
}

func run() error {
	path := os.Getenv("VAULT_PATH")
	if path == "" {
		return errors.New("no VAULT_PATH has been set")
	}

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

	files, err := vault.Scan(path)
	if err != nil {
		if !errors.Is(err, vault.ErrUnclosedFrontmatter) {
			return fmt.Errorf("critical error at scanning vault at %s: %w", path, err)
		}
		log.Printf("Warnings:\n%v", err)
	}

	for i, file := range files {
		fmt.Printf("File %d: path:%s hex:%x title:%s\n", i+1, file.Path, file.Hash, file.Name)
	}
	return nil
}
