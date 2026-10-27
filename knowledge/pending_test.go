package knowledge

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/Zinoki12/rag-ai-system/embed"
	"github.com/Zinoki12/rag-ai-system/internal/testdb"
)

// quotaProvider lets a set number of calls through and refuses the rest, the
// way a per-minute quota does once it is spent.
type quotaProvider struct {
	embed.Provider
	batch int
	allow int
	calls int
}

func (q *quotaProvider) MaxBatch() int { return q.batch }

func (q *quotaProvider) Embed(ctx context.Context, texts []string, kind embed.Kind) ([]embed.Vector, error) {
	q.calls++
	if q.calls > q.allow {
		return nil, errors.New("api error 429: quota exhausted")
	}
	return q.Provider.Embed(ctx, texts, kind)
}

// A pass that hits the quota part-way must keep what it embedded before the
// refusal, and the next pass must finish the rest rather than start over.
//
// The defect this pins: one round claimed 128 chunks, the provider split them
// into calls of 100 and 28, and nothing was saved unless both succeeded. Under
// the Gemini free tier the first call spends the minute's quota, the second is
// refused, and the round is thrown away — every pass, forever, with the index
// stuck at whatever it held before.
func TestIndexKeepsWhatItEmbeddedBeforeTheQuotaRanOut(t *testing.T) {
	_, dsn := testdb.New(t)
	vault := t.TempDir()
	for i := range 5 {
		body := fmt.Sprintf("---\ntitle: Заметка %d\n---\nТекст заметки номер %d.", i, i)
		if err := os.WriteFile(filepath.Join(vault, fmt.Sprintf("n%d.md", i)), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	ctx := context.Background()
	kb, err := Open(ctx, Config{DSN: dsn, VaultPath: vault, Embedder: embed.Config{Provider: embed.ProviderFake, Dim: 16}})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(kb.Close)

	fake := kb.embed
	kb.embed = &quotaProvider{Provider: fake, batch: 2, allow: 2}

	res, err := kb.Index(ctx)
	if err == nil {
		t.Fatal("Index reported no error although the provider refused a call")
	}
	st, serr := kb.Stats(ctx)
	if serr != nil {
		t.Fatal(serr)
	}
	if st.Chunks != 5 {
		t.Fatalf("chunks = %d, want 5 (one per note)", st.Chunks)
	}
	if res.ChunksEmbedded != 4 || st.Embedded != 4 {
		t.Fatalf("after the refusal: reported %d, stored %d; want the 4 from the two calls that went through",
			res.ChunksEmbedded, st.Embedded)
	}

	// The quota refills; the next pass embeds only what is still missing.
	q := &quotaProvider{Provider: fake, batch: 2, allow: 100}
	kb.embed = q
	res, err = kb.Index(ctx)
	if err != nil {
		t.Fatalf("second Index: %v", err)
	}
	if st, _ = kb.Stats(ctx); st.Embedded != 5 || res.ChunksEmbedded != 1 || q.calls != 1 {
		t.Fatalf("second pass: stored %d, embedded %d in %d call(s); want 5 stored, 1 embedded in 1 call",
			st.Embedded, res.ChunksEmbedded, q.calls)
	}
}
