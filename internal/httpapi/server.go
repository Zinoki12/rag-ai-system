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
// The order of the wrappers is deliberate. Request-id is outermost so every
// later layer, including the panic log, has an id to report. Recover sits above
// logging so a panicking handler still produces a log line with its status.
func Handler(a Service, log *slog.Logger, cfg Config) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handleHealth(a))
	mux.HandleFunc("POST /search", handleSearch(a, log))
	mux.HandleFunc("POST /ask", handleAsk(a, log))

	var h http.Handler = mux
	h = withTimeout(cfg.RequestTimeout, h)
	h = withLogging(log, h)
	h = withRecover(log, h)
	h = withRequestID(h)
	return h
}

// Run serves until ctx is cancelled, then drains in-flight requests.
func Run(ctx context.Context, a Service, log *slog.Logger, cfg Config) error {
	srv := &http.Server{
		Addr:    cfg.Addr,
		Handler: Handler(a, log, cfg),

		// These four are not tuning knobs, they are the difference between a
		// server and an open socket. Without ReadHeaderTimeout a client can
		// hold a connection by dribbling one header byte at a time until the
		// listener runs out of file descriptors (Slowloris); the others bound
		// the body, the response and the idle keep-alive respectively.
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      cfg.RequestTimeout + 30*time.Second,
		IdleTimeout:       120 * time.Second,

		BaseContext: func(net.Listener) context.Context { return ctx },
	}

	errCh := make(chan error, 1)
	go func() {
		log.Info("listening", slog.String("addr", cfg.Addr), slog.String("space", a.SpaceName()))
		// ErrServerClosed is what Shutdown causes; it is the success path here.
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- fmt.Errorf("listen on %s: %w", cfg.Addr, err)
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
