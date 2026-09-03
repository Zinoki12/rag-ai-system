package httpapi

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Zinoki12/rag-ai-system/knowledge"
)

// blockingService reports when a request reaches it, waits to be released, and
// fails if its context is cancelled first.
type blockingService struct {
	started chan struct{}
	release chan struct{}
}

func (b *blockingService) Stats(context.Context) (knowledge.Stats, error) {
	return knowledge.Stats{}, nil
}

func (b *blockingService) Ask(context.Context, string, int) (knowledge.Answer, error) {
	return knowledge.Answer{}, nil
}

func (b *blockingService) Index(context.Context) (knowledge.IndexResult, error) {
	return knowledge.IndexResult{}, nil
}

func (b *blockingService) Search(ctx context.Context, _ string, _ int) ([]knowledge.Hit, error) {
	close(b.started)
	select {
	case <-b.release:
		return []knowledge.Hit{{Source: "a.md", Text: "готово"}}, nil
	case <-ctx.Done():
		// This is the failure the test is looking for: the handler was told to
		// give up while the server was supposed to be draining it.
		return nil, errors.New("context cancelled mid-request")
	}
}

// Graceful shutdown must let a request that is already running finish.
//
// The way to get this wrong is to hand http.Server a BaseContext derived from
// the same context that signals shutdown: every in-flight handler is then
// cancelled the instant SIGTERM lands, and Shutdown politely waits for
// handlers that have already been told to abandon their work.
func TestGracefulShutdownLetsInFlightRequestFinish(t *testing.T) {
	svc := &blockingService{started: make(chan struct{}), release: make(chan struct{})}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := Config{RequestTimeout: 10 * time.Second, ShutdownGrace: 10 * time.Second}

	serveErr := make(chan error, 1)
	go func() { serveErr <- Serve(ctx, ln, svc, testSpace, log, cfg) }()

	respCh := make(chan *http.Response, 1)
	reqErr := make(chan error, 1)
	go func() {
		resp, err := http.Post("http://"+addr+"/search", "application/json",
			strings.NewReader(`{"query":"q"}`))
		if err != nil {
			reqErr <- err
			return
		}
		respCh <- resp
	}()

	select {
	case <-svc.started:
	case err := <-reqErr:
		t.Fatalf("request failed before reaching the handler: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("request never reached the handler")
	}

	// Shutdown starts while the handler is still working.
	cancel()
	time.Sleep(50 * time.Millisecond)
	close(svc.release)

	select {
	case resp := <-respCh:
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("in-flight request got status %d during shutdown, want 200 (body: %s)",
				resp.StatusCode, body)
		}
	case err := <-reqErr:
		t.Errorf("in-flight request was dropped during shutdown: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("in-flight request never completed")
	}

	select {
	case err := <-serveErr:
		if err != nil {
			t.Errorf("Serve() returned %v, want nil after a clean shutdown", err)
		}
	case <-time.After(5 * time.Second):
		t.Error("Serve() did not return after shutdown")
	}
}

// A port already in use must surface as an error, not a server that silently
// serves nothing.
func TestRunReportsListenFailure(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := DefaultConfig
	cfg.Addr = ln.Addr().String()

	if err := Run(context.Background(), &stubService{}, testSpace, log, cfg); err == nil {
		t.Error("Run() on an occupied port returned no error")
	}
}
