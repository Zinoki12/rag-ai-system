package httpapi

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"
)

// Config tunes the HTTP server.
type Config struct {
	Addr           string
	RequestTimeout time.Duration
	ShutdownGrace  time.Duration
}

// DefaultConfig is the configuration used when nothing is overridden.
var DefaultConfig = Config{
	Addr: ":8081",
	// Long enough for a local model to generate an answer, short enough that a
	// wedged upstream cannot pin a worker indefinitely.
	RequestTimeout: 2 * time.Minute,
	ShutdownGrace:  20 * time.Second,
}

// Handler builds the routed, wrapped handler.
//
// ctx bounds the lifetime of background work started from a request — indexing
// in particular, which must outlive the request that asked for it. space names
// the embedding space, read once at startup because it cannot change while the
// process runs.
//
// The order of the wrappers is deliberate. Request-id is outermost so every
// later layer, including the panic log, has an id to report. Recover sits above
// logging so a panicking handler still produces a log line with its status.
func Handler(ctx context.Context, a Service, space string, log *slog.Logger, cfg Config) http.Handler {
	ix := newIndexer(ctx, a, log)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handleHealth(a))
	mux.HandleFunc("POST /search", handleSearch(a, space, log))
	mux.HandleFunc("POST /ask", handleAsk(a, space, log))
	mux.HandleFunc("POST /reindex", handleReindex(ix))
	mux.HandleFunc("GET /reindex/status", handleReindexStatus(ix))

	var h http.Handler = mux
	h = withTimeout(cfg.RequestTimeout, h)
	h = withLogging(log, h)
	h = withRecover(log, h)
	h = withRequestID(h)
	return h
}

// Run listens on cfg.Addr and serves until ctx is cancelled.
func Run(ctx context.Context, a Service, space string, log *slog.Logger, cfg Config) error {
	ln, err := net.Listen("tcp", cfg.Addr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", cfg.Addr, err)
	}
	return Serve(ctx, ln, a, space, log, cfg)
}

// Serve runs on an already-open listener, then drains in-flight requests.
//
// Split out from Run so a test can listen on port 0 and still know where to
// send a request; shutdown behaviour is exactly the kind of thing that has to
// be exercised against a real socket to mean anything.
func Serve(ctx context.Context, ln net.Listener, a Service, space string, log *slog.Logger, cfg Config) error {
	srv := &http.Server{
		// The signal context, not a detached one: a background indexing pass
		// should stop when the server is shutting down. It is resumable by
		// construction, so a cancelled pass costs nothing but the batch in
		// flight. Requests get their own detached base context below.
		Handler: Handler(ctx, a, space, log, cfg),

		// These four are not tuning knobs, they are the difference between a
		// server and an open socket. Without ReadHeaderTimeout a client can
		// hold a connection by dribbling one header byte at a time until the
		// listener runs out of file descriptors (Slowloris); the others bound
		// the body, the response and the idle keep-alive respectively.
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      cfg.RequestTimeout + 30*time.Second,
		IdleTimeout:       120 * time.Second,

		// WithoutCancel, not ctx itself. Requests must inherit ctx's values but
		// not its cancellation: deriving them from the signal context would
		// cancel every in-flight handler the instant SIGTERM arrives, so
		// Shutdown would dutifully wait for handlers that had already been told
		// to give up — a graceful shutdown that drops exactly the requests it
		// exists to protect.
		BaseContext: func(net.Listener) context.Context { return context.WithoutCancel(ctx) },
	}

	errCh := make(chan error, 1)
	go func() {
		log.Info("listening", slog.String("addr", ln.Addr().String()), slog.String("space", space))
		// ErrServerClosed is what Shutdown causes; it is the success path here.
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- fmt.Errorf("serve on %s: %w", ln.Addr(), err)
			return
		}
		errCh <- nil
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}

	log.Info("shutting down", slog.Duration("grace", cfg.ShutdownGrace))

	// A fresh context: the one that triggered the shutdown is already cancelled,
	// and passing it to Shutdown would abandon in-flight requests immediately
	// instead of giving them their grace period.
	shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cfg.ShutdownGrace)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("graceful shutdown: %w", err)
	}
	return <-errCh
}
