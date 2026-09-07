package embed

import (
	"strings"
	"testing"
)

// Setting one broker for both halves is the natural mistake, and the generic
// "want one of ..." would send the reader hunting for a typo that is not there.
func TestOpenRouterAsAnEmbedderExplainsItself(t *testing.T) {
	t.Setenv("EMBED_PROVIDER", "openrouter")

	_, err := ConfigFromEnv()
	if err == nil {
		t.Fatal("openrouter принят как провайдер эмбеддингов")
	}
	msg := err.Error()
	// The message must say what is wrong and what to do instead; either half
	// alone leaves the reader stuck.
	for _, want := range []string{
		"no embeddings endpoint",
		"LLM_PROVIDER=openrouter",
		"EMBED_PROVIDER=ollama",
		"EMBED_PROVIDER=google",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("в сообщении нет %q:\n%s", want, msg)
		}
	}
}
