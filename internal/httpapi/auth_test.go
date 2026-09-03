package httpapi

import (
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Zinoki12/rag-ai-system/knowledge"
)

const testToken = "s3cret-token"

func newAuthedHandler(svc Service) http.Handler {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return Handler(context.Background(), svc, testSpace, log, Config{
		RequestTimeout: 5 * time.Second,
		AuthToken:      testToken,
	})
}

func TestAuthGuardsEveryPathThatReturnsContent(t *testing.T) {
	// Each of these either returns stored text or starts work that costs money
	// upstream. None may answer an anonymous caller.
	protected := []struct{ method, path, body string }{
		{http.MethodPost, "/search", `{"query":"секрет"}`},
		{http.MethodPost, "/ask", `{"question":"секрет"}`},
		{http.MethodPost, "/reindex", ""},
		{http.MethodGet, "/reindex/status", ""},
	}

	for _, p := range protected {
		t.Run(p.method+" "+p.path, func(t *testing.T) {
			svc := &stubService{
				hits:   []knowledge.Hit{{Source: "секретная-заметка.md", Text: "содержимое"}},
				answer: knowledge.Answer{Text: "ответ"},
			}
			h := newAuthedHandler(svc)

			w := do(t, h, p.method, p.path, p.body)
			if w.Code != http.StatusUnauthorized {
				t.Fatalf("anonymous request: status = %d, want 401", w.Code)
			}
			if got := w.Header().Get("WWW-Authenticate"); got == "" {
				t.Error("401 without a WWW-Authenticate header")
			}
			if strings.Contains(w.Body.String(), "содержимое") {
				t.Error("the 401 body leaked note content")
			}
			if svc.indexRuns.Load() != 0 {
				t.Error("an anonymous request started an indexing pass")
			}

			// The same request with the token goes through.
			r := httptest.NewRequest(p.method, p.path, strings.NewReader(p.body))
			r.Header.Set("Authorization", "Bearer "+testToken)
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, r)
			if rec.Code == http.StatusUnauthorized {
				t.Fatalf("authenticated request was rejected: %s", rec.Body)
			}
		})
	}
}

func TestAuthRejectsWrongCredentials(t *testing.T) {
	cases := []struct {
		name   string
		header string
	}{
		{"пустой заголовок", ""},
		{"без схемы", testToken},
		{"чужая схема", "Basic " + testToken},
		{"неверный токен", "Bearer wrong-token"},
		{"верный токен как префикс", "Bearer " + testToken[:5]},
		{"верный токен с хвостом", "Bearer " + testToken + "x"},
	}

	h := newAuthedHandler(&stubService{})
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/search", strings.NewReader(`{"query":"x"}`))
			if c.header != "" {
				r.Header.Set("Authorization", c.header)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != http.StatusUnauthorized {
				t.Fatalf("status = %d, want 401", w.Code)
			}
		})
	}
}

// The scheme name is case-insensitive per RFC 7235; the token is not.
func TestAuthAcceptsAnyCaseScheme(t *testing.T) {
	h := newAuthedHandler(&stubService{})
	for _, scheme := range []string{"Bearer", "bearer", "BEARER", "BeArEr"} {
		r := httptest.NewRequest(http.MethodPost, "/search", strings.NewReader(`{"query":"x"}`))
		r.Header.Set("Authorization", scheme+" "+testToken)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code == http.StatusUnauthorized {
			t.Errorf("scheme %q was rejected", scheme)
		}
	}
}

// Monitoring probes usually cannot be given a credential, so /health stays open
// — and must therefore stay free of note content.
func TestHealthStaysPublic(t *testing.T) {
	svc := &stubService{stats: knowledge.Stats{Notes: 5, Chunks: 16, Embedded: 16}}
	w := do(t, newAuthedHandler(svc), http.MethodGet, "/health", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestAuthDisabledWhenNoTokenConfigured(t *testing.T) {
	w := do(t, newTestHandler(&stubService{}), http.MethodPost, "/search", `{"query":"x"}`)
	if w.Code == http.StatusUnauthorized {
		t.Fatal("no token configured, yet the request was rejected")
	}
}

// Binding to a non-loopback address with no token would publish the knowledge
// base to the network. Serve must refuse rather than document the hole.
func TestServeRefusesOpenNetworkListener(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := DefaultConfig

	t.Run("сеть без токена — отказ", func(t *testing.T) {
		ln, err := net.Listen("tcp", "0.0.0.0:0")
		if err != nil {
			t.Skipf("cannot bind 0.0.0.0: %v", err)
		}
		defer ln.Close()

		// A deadline, not context.Background(): if the check regresses, Serve
		// starts serving and would otherwise hang the whole test binary until
		// go test's own timeout. Two seconds turns that into a clear failure.
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		err = Serve(ctx, ln, &stubService{}, testSpace, log, cfg)
		if err == nil {
			t.Fatal("Serve accepted an open network listener with no token")
		}
		if !strings.Contains(err.Error(), "authentication") {
			t.Errorf("error = %v, want it to say why", err)
		}
	})

	t.Run("loopback без токена — разрешено", func(t *testing.T) {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		defer ln.Close()

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // shut down immediately; we only care that it got past the check
		if err := Serve(ctx, ln, &stubService{}, testSpace, log, cfg); err != nil {
			t.Fatalf("Serve on loopback: %v", err)
		}
	})

	t.Run("сеть с токеном — разрешено", func(t *testing.T) {
		ln, err := net.Listen("tcp", "0.0.0.0:0")
		if err != nil {
			t.Skipf("cannot bind 0.0.0.0: %v", err)
		}
		defer ln.Close()

		authed := cfg
		authed.AuthToken = testToken

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := Serve(ctx, ln, &stubService{}, testSpace, log, authed); err != nil {
			t.Fatalf("Serve with a token: %v", err)
		}
	})
}
