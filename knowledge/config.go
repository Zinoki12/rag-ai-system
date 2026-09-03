package knowledge

import (
	"os"

	"github.com/Zinoki12/rag-ai-system/embed"
	"github.com/Zinoki12/rag-ai-system/internal/config"
	"github.com/Zinoki12/rag-ai-system/internal/storage"
	"github.com/Zinoki12/rag-ai-system/llm"
)

// ConfigFromEnv assembles a Config from environment variables.
//
// A convenience for programs that are happy to be configured that way — the
// commands in this repository are. Anything embedding this package as a library
// can ignore it and fill in Config directly; nothing else here reads the
// environment.
//
//	POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT
//	VAULT_PATH      markdown directory to index
//	CHUNK_SIZE      runes per chunk (default 800)
//	EMBED_*         see embed.ConfigFromEnv
//	LLM_*           see llm.ConfigFromEnv
//
// The generator is always populated; set Config.Generator to nil for a
// search-only knowledge base.
func ConfigFromEnv() (Config, error) {
	dsn, err := storage.DSNFromEnv()
	if err != nil {
		return Config{}, err
	}

	embedCfg, err := embed.ConfigFromEnv()
	if err != nil {
		return Config{}, err
	}

	llmCfg, err := llm.ConfigFromEnv()
	if err != nil {
		return Config{}, err
	}

	chunkSize, err := config.Int("CHUNK_SIZE", DefaultChunkSize)
	if err != nil {
		return Config{}, err
	}

	return Config{
		DSN:       dsn,
		VaultPath: os.Getenv("VAULT_PATH"),
		ChunkSize: chunkSize,
		Embedder:  embedCfg,
		Generator: &llmCfg,
	}, nil
}
